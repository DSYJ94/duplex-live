"""Actual semantic backend tool execution over episode-local read-only records.

The record values stay outside the model prompt until a validated tool call.
Application revision guards reject late cancelled or superseded results. This
backend component is a hybrid bridge, not proof of learned speech-model routing.
"""
import json
import os
import time
import threading
import re


class WorkerContractError(RuntimeError):
    def __init__(self,message,result):
        super().__init__(message)
        self.result = result


class WorkerSuperseded(RuntimeError):
    pass


def project_identified_by_user(context, project):
    """Available record names and assistant guesses do not identify a project."""
    explicit = " ".join(e.get("text", "") for e in context if e.get("role") == "user")
    # GPT-Live supplies text deltas, including their original whitespace.
    deltas = "".join(e.get("text", "") for e in context if e.get("speaker") == "user")
    normalize = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.casefold()))
    needle = normalize(project)
    return bool(needle and any(" "+needle+" " in " "+normalize(text)+" " for text in (explicit, deltas)))


class RecordWorld:
    def __init__(self, records):
        self.records = records
        self.tasks = {}
        self.receipts = []
        self.lock = threading.RLock()

    def reserve(self, task_id, context=None):
        with self.lock:
            revision = self.tasks.get(task_id, {}).get("revision", 0) + 1
            self.tasks[task_id] = {"revision":revision,"cancelled":False,"closed":False,"phase":"pending","context":context}
            return revision

    def read(self, task_id, project, field, expected_revision=None):
        with self.lock:
            if project not in self.records or field not in self.records[project]:
                raise ValueError("Tool requested an unknown project or record field")
            if expected_revision is None:
                revision = self.reserve(task_id)
            else:
                state = self.tasks.get(task_id)
                if not state or state["cancelled"] or state.get("closed",False) or state["revision"] != expected_revision:
                    raise WorkerSuperseded("Task was cancelled or replaced before record execution")
                revision = expected_revision
            receipt = {"task_id":task_id,"revision":revision,"tool":"lookup_record", "project":project,"field":field,
                "value":self.records[project][field],"read_only":True,"executed":True}
            self.tasks[task_id].update(operation="lookup_record",phase="running",project=project,field=field)
            self.receipts.append(receipt)
            return receipt

    def cancel(self, task_id):
        with self.lock:
            if task_id not in self.tasks or self.tasks[task_id]["cancelled"] or self.tasks[task_id].get("closed",False):
                raise ValueError("No active task to cancel")
            self.tasks[task_id]["cancelled"] = True
            return {"task_id":task_id,"revision":self.tasks[task_id]["revision"],"tool":"cancel_lookup","cancelled":True}

    def deliverable(self, receipt):
        with self.lock:
            state = self.tasks.get(receipt["task_id"])
            return bool(state and not state["cancelled"] and not state.get("closed",False) and state["revision"] == receipt["revision"])

    def cancel_many(self,task_ids):
        with self.lock:
            if not isinstance(task_ids,list) or not task_ids or any(not isinstance(t,str) for t in task_ids) or len(set(task_ids)) != len(task_ids):
                raise ValueError("Cancellation requires distinct existing task IDs")
            if any(t not in self.tasks or self.tasks[t]["cancelled"] or self.tasks[t].get("closed",False) for t in task_ids):
                raise ValueError("Every cancellation target must be active; nothing was changed")
            receipts = [self.cancel(t) for t in task_ids]
            return {"tool":"cancel_lookup","cancelled":True,"task_ids":task_ids,"receipts":receipts}

    def finish(self,task_id,revision):
        """Close a delivered or non-lookup request without claiming cancellation."""
        with self.lock:
            state=self.tasks.get(task_id)
            if not state or state['revision']!=revision or state['cancelled'] or state.get('closed',False):
                raise WorkerSuperseded('Cannot finish cancelled, replaced or already closed work')
            state.update(closed=True,phase='completed')

    def mark_ready(self,receipt):
        with self.lock:
            if not self.deliverable(receipt):
                raise WorkerSuperseded("Cancelled or replaced lookup cannot become ready")
            self.tasks[receipt["task_id"]]["phase"] = "result_ready"


def tool(name,description,properties):
    return {"type":"function","name":name,"description":description,"strict":True,
        "parameters":{"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}}


def run_worker(context,world,task_id,model="gpt-6-luna",expected_revision=None,dispatch_protocol="legacy",pending_speech=None):
    if dispatch_protocol not in ("legacy","required-action","state-aware"):raise ValueError("Unknown controller dispatch protocol")
    import requests
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Backend API credential unavailable")
    tools = [tool("lookup_record","Read an identified private project field. Ask for missing identification before calling.",
        {"project":{"type":"string"},"field":{"type":"string"}}),
        tool("cancel_lookup","Cancel the identified existing lookups. Include all related duplicates, excluding unrelated work.",
            {"task_ids":{"type":"array","items":{"type":"string"},"minItems":1}})]
    schema = {"type":"object","properties":{"route":{"type":"string","enum":["answer","clarify","delegate","cancel"]},
        "content":{"type":"string"}},"required":["route","content"],"additionalProperties":False}
    pending_speech=[] if pending_speech is None else pending_speech
    if (not isinstance(pending_speech,list) or any(set(p)!={'id','request_text','text'} for p in pending_speech)
            or any(not isinstance(p[k],str) or not p[k].strip() for p in pending_speech for k in p)
            or len({p['id'] for p in pending_speech})!=len(pending_speech)):
        raise ValueError('Invalid actual pending-speech snapshot')
    if dispatch_protocol!='state-aware' and pending_speech:
        raise ValueError('Pending-speech actions require explicit state-aware protocol')
    if dispatch_protocol=='state-aware':
        schema['properties']['consume_pending_speech_ids']={'type':'array','items':{'type':'string'}}
        schema['required'].append('consume_pending_speech_ids')
    with world.lock:
        available = {"records":{p:list(fields) for p,fields in world.records.items()},
            "tasks":{name:state.copy() for name,state in world.tasks.items() if not state["cancelled"] and not state.get("closed",False) and name != task_id}}
    instructions = ("Help a live voice assistant with only the latest USER request in its chronological transcript. "
        "Do not repeat older pending or completed lookups merely because the voice assistant has not spoken their results. "
        "Answer simple general knowledge without tools. Private records require lookup_record; never invent their contents. "
        "Clarify missing project or field. Honor negation and corrections. Cancel only identified existing tasks. "
        "Available record names are not user identification; never select a project the user has not identified. "
        "The current dispatch is not a cancellation target. When duplicate active lookups belong to the cancelled request, "
        "select all their IDs in one cancellation call; preserve unrelated work. Assistant speech is not a verified tool receipt. "
        "Use route delegate after a record tool, cancel after cancellation, otherwise answer or clarify. "
        "Do not say a lookup or cancellation completed until its tool receipt is present. Reply in at most 60 words. "
        "Available project/field names and active tasks: " + json.dumps(available))
    if dispatch_protocol=='state-aware':
        instructions+=(" Actual client speech state, separate from active lookup tasks: "+json.dumps(pending_speech)+
            ". These ordinary answers are owed and have not been fully played; an unknown prefix may have been heard. "
            "The client will resume each unless you list its ID in consume_pending_speech_ids. "
            "List an ID only if your current reply answers that exact request, or the latest user explicitly withdraws it. "
            "If you only confirm cancellation of a lookup, preserve unrelated ordinary answers by leaving their IDs out. "
            "Never repeat an owed answer in this reply and also leave it queued. Do not consume an answer merely because "
            "you cancelled an unrelated lookup. Pending speech is not a tool receipt or project identification.")
    base = {"model":model,"instructions":instructions,"tools":tools,"parallel_tool_calls":False,
        "max_output_tokens":2048,"reasoning":{"effort":"low"},
        "text":{"format":{"type":"json_schema","name":"worker_result","strict":True,"schema":schema}}}
    first = {**base,"input":json.dumps(context)}
    if dispatch_protocol in ("required-action","state-aware"):
        # Model selects a single semantic action, including an ordinary reply.
        # Required does not force a record read or cancellation: the reply action
        # is a no-side-effect output channel and carries no execution receipt.
        reply_properties={"route":{"type":"string","enum":["answer","clarify"]},"content":{"type":"string"}}
        if dispatch_protocol=='state-aware':
            reply_properties['consume_pending_speech_ids']=schema['properties']['consume_pending_speech_ids']
        reply_tool=tool("respond_to_user","Answer general knowledge or ask for missing identification; no external side effects.",reply_properties)
        first.pop("text")
        first.update(tools=tools+[reply_tool],tool_choice="required")
        first['instructions']+=(" Select exactly one action now. Use respond_to_user for an ordinary answer or clarification. "
            "Use lookup_record for an identified private field, or cancel_lookup for identified active tasks. "
            "Never return a preliminary JSON delegation/cancellation plan instead of the action. "
            "Calling respond_to_user does not execute a lookup or cancellation.")
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
            "seconds":time.monotonic()-start,"output":[{k:v for k,v in item.items() if k != "encrypted_content"}
                for item in data.get("output",[])]})
        return data
    response = request(first)
    direct_reply = None
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    if dispatch_protocol in ('required-action','state-aware') and len(calls)!=1:
        raise WorkerContractError('Required semantic action was not returned exactly once',{'events':events})
    selected_route = None
    if not calls:
        planning_text = " ".join(part["text"] for item in response["output"] if item["type"] == "message"
            for part in item.get("content",[]) if part["type"] == "output_text")
        plan = json.loads(planning_text)
        selected_route = plan["route"]
        required_tool = {"delegate":"lookup_record","cancel":"cancel_lookup"}.get(plan["route"])
        if required_tool:
            # Routing JSON is a plan, not a tool receipt. Execute the selected
            # phase explicitly; the model still supplies identifying arguments.
            response = request({**base,"previous_response_id":response["id"],
                "input":"Execute the selected tool now. Do not repeat the acknowledgement.",
                "tool_choice":{"type":"function","name":required_tool}})
            calls = [item for item in response["output"] if item["type"] == "function_call"]
            if not calls:
                raise WorkerContractError("Required execution phase returned no tool call",{"events":events,"plan":plan})
    receipts = []
    if calls:
        if len(calls) != 1:
            raise RuntimeError("Worker exceeded the single-task tool contract")
        call = calls[0]
        arguments = json.loads(call["arguments"])
        if call["name"]=="respond_to_user":
            reply_keys={'route','content'}|({'consume_pending_speech_ids'} if dispatch_protocol=='state-aware' else set())
            if (dispatch_protocol not in ("required-action","state-aware") or set(arguments)!=reply_keys
                    or arguments['route'] not in ('answer','clarify')
                    or not isinstance(arguments['content'],str) or not arguments['content'].strip()):
                raise WorkerContractError('Invalid no-side-effect reply action',{'events':events,'arguments':arguments})
            direct_reply=arguments;selected_route=arguments['route']
        elif call["name"] == "lookup_record":
            if not project_identified_by_user(context, arguments["project"]):
                raise WorkerContractError("Private project was not identified by the user", {"events":events,"arguments":arguments})
            try:
                receipt = world.read(task_id,**arguments,expected_revision=expected_revision)
            except ValueError as error:
                raise WorkerContractError('Private record request was rejected',
                    {'events':events,'arguments':arguments,'receipts':[],
                     'tool_error':str(error)}) from error
            selected_route = "delegate"
        elif call["name"] == "cancel_lookup":
            # Legacy single-ID captured events remain readable for regression
            # checks. New strict tool calls use the explicit array contract.
            targets = [arguments["task_id"]] if set(arguments) == {"task_id"} else arguments.get("task_ids")
            if not isinstance(targets,list) or task_id in targets or any(t not in available["tasks"] for t in targets):
                raise WorkerContractError("Cancellation must target existing work, not its own dispatch",{"events":events,"arguments":arguments})
            try:
                receipt = world.cancel_many(targets)
            except ValueError as error:
                raise WorkerContractError('Cancellation targets are no longer active',
                    {'events':events,'arguments':arguments,'receipts':[],
                     'tool_error':str(error)}) from error
            selected_route = "cancel"
        else:
            raise RuntimeError("Unknown backend tool")
        if direct_reply is None:
            receipts.append(receipt)
            response = request({**base,"tool_choice":"none","previous_response_id":response["id"],"input":[
                {"type":"function_call_output","call_id":call["call_id"],"output":json.dumps(receipt)}]})
    content = " ".join(part["text"] for item in response["output"] if item["type"] == "message"
        for part in item.get("content",[]) if part["type"] == "output_text")
    parsed = direct_reply if direct_reply is not None else json.loads(content)
    if dispatch_protocol=='state-aware':
        consumed=parsed.get('consume_pending_speech_ids')
        if (not isinstance(consumed,list) or any(not isinstance(identity,str) for identity in consumed)
                or len(set(consumed))!=len(consumed) or not set(consumed)<={p['id'] for p in pending_speech}
                or (parsed['route']=='clarify' and consumed)):
            raise WorkerContractError('Response cannot consume unknown, duplicate or unfulfilled speech IDs',
                {'events':events,'response':parsed,'pending_speech':pending_speech,'receipts':receipts})
    final_response_route = parsed["route"]
    # A worker may correctly call its post-tool factual response an answer.
    # Preserve the original dispatch decision separately from that final phase.
    parsed["route"] = selected_route or final_response_route
    result = {**parsed,"receipts":receipts,"events":events,"task_id":task_id,"speech_model_routing":False}
    result["final_response_route"] = final_response_route
    result["dispatch_protocol"] = dispatch_protocol
    if not isinstance(parsed.get("content"), str) or not parsed["content"].strip():
        raise WorkerContractError("Backend returned empty result text; do not inject an empty context span",result)
    if parsed["route"] == "delegate" and not any(r["tool"] == "lookup_record" for r in receipts):
        raise WorkerContractError("Delegation reported without actual record execution",result)
    if parsed["route"] == "cancel" and not any(r["tool"] == "cancel_lookup" for r in receipts):
        raise WorkerContractError("Cancellation reported without actual task cancellation",result)
    return result
