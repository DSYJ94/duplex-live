"""Actual semantic backend tool execution over episode-local read-only records.

The record values stay outside the model prompt until a validated tool call.
Application revision guards reject late cancelled or superseded results. This
backend component is a hybrid bridge, not proof of learned speech-model routing.
"""
import json
import os
import time


class WorkerContractError(RuntimeError):
    def __init__(self,message,result):
        super().__init__(message)
        self.result = result


class RecordWorld:
    def __init__(self, records):
        self.records = records
        self.tasks = {}
        self.receipts = []

    def reserve(self, task_id):
        revision = self.tasks.get(task_id, {}).get("revision", 0) + 1
        self.tasks[task_id] = {"revision":revision,"cancelled":False}
        return revision

    def read(self, task_id, project, field):
        if project not in self.records or field not in self.records[project]:
            raise ValueError("Tool requested an unknown project or record field")
        revision = self.reserve(task_id)
        receipt = {"task_id":task_id,"revision":revision,"tool":"lookup_record", "project":project,"field":field,
            "value":self.records[project][field],"read_only":True,"executed":True}
        self.receipts.append(receipt)
        return receipt

    def cancel(self, task_id):
        if task_id not in self.tasks or self.tasks[task_id]["cancelled"]:
            raise ValueError("No active task to cancel")
        self.tasks[task_id]["cancelled"] = True
        return {"task_id":task_id,"revision":self.tasks[task_id]["revision"],"tool":"cancel_lookup","cancelled":True}

    def deliverable(self, receipt):
        state = self.tasks.get(receipt["task_id"])
        return bool(state and not state["cancelled"] and state["revision"] == receipt["revision"])


def tool(name,description,properties):
    return {"type":"function","name":name,"description":description,"strict":True,
        "parameters":{"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}}


def run_worker(context,world,task_id,model="gpt-6-luna"):
    import requests
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Backend API credential unavailable")
    tools = [tool("lookup_record","Read an identified private project field. Ask for missing identification before calling.",
        {"project":{"type":"string"},"field":{"type":"string"}}),
        tool("cancel_lookup","Cancel an identified active lookup and reject its future result.",{"task_id":{"type":"string"}})]
    schema = {"type":"object","properties":{"route":{"type":"string","enum":["answer","clarify","delegate","cancel"]},
        "content":{"type":"string"}},"required":["route","content"],"additionalProperties":False}
    instructions = ("Help a live voice assistant with the latest request in its transcript. "
        "Answer simple general knowledge without tools. Private records require lookup_record; never invent their contents. "
        "Clarify missing project or field. Honor negation and corrections. Cancel only an identified active task. "
        "Use route delegate after a record tool, cancel after cancellation, otherwise answer or clarify. "
        "Do not say a lookup or cancellation completed until its tool receipt is present. Reply in at most 60 words. "
        "Available project/field names and active tasks: " + json.dumps({"records":{p:list(fields) for p,fields in world.records.items()},
            "tasks":{name:state for name,state in world.tasks.items() if not state["cancelled"]}}))
    base = {"model":model,"instructions":instructions,"tools":tools,"parallel_tool_calls":False,
        "max_output_tokens":2048,"reasoning":{"effort":"low"},
        "text":{"format":{"type":"json_schema","name":"worker_result","strict":True,"schema":schema}}}
    events = []
    def request(payload):
        start = time.monotonic()
        response = requests.post("https://api.openai.com/v1/responses",headers={"Authorization":"Bearer "+key},json=payload,timeout=90)
        if response.status_code != 200:
            raise RuntimeError(f"Backend HTTP {response.status_code}")
        data = response.json()
        if data.get("status") != "completed":
            raise RuntimeError("Backend did not complete within the token budget")
        events.append({"response_id":data["id"],"model":data.get("model"),"usage":data.get("usage"),
            "seconds":time.monotonic()-start,"output":data.get("output",[])})
        return data
    response = request({**base,"input":json.dumps(context)})
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    receipts = []
    if calls:
        if len(calls) != 1:
            raise RuntimeError("Worker exceeded the single-task tool contract")
        call = calls[0]
        arguments = json.loads(call["arguments"])
        if call["name"] == "lookup_record":
            receipt = world.read(task_id,**arguments)
        elif call["name"] == "cancel_lookup":
            receipt = world.cancel(**arguments)
        else:
            raise RuntimeError("Unknown backend tool")
        receipts.append(receipt)
        response = request({**base,"tool_choice":"none","previous_response_id":response["id"],"input":[
            {"type":"function_call_output","call_id":call["call_id"],"output":json.dumps(receipt)}]})
    content = " ".join(part["text"] for item in response["output"] if item["type"] == "message"
        for part in item.get("content",[]) if part["type"] == "output_text")
    parsed = json.loads(content)
    result = {**parsed,"receipts":receipts,"events":events,"task_id":task_id,"speech_model_routing":False}
    if parsed["route"] == "delegate" and not any(r["tool"] == "lookup_record" for r in receipts):
        raise WorkerContractError("Delegation reported without actual record execution",result)
    if parsed["route"] == "cancel" and not any(r["tool"] == "cancel_lookup" for r in receipts):
        raise WorkerContractError("Cancellation reported without actual task cancellation",result)
    return result
