"""Project actual guarded tool execution into value-hidden conversation events.

These notices are client control metadata, not assistant speech or an invented
backend answer. Values remain withheld from a new foreground request while the
original private response is held by the client delivery policy.
"""
import hashlib
import json


def project_execution(call,receipt):
    if call.get('type')!='function_call' or not isinstance(call.get('call_id'),str):raise ValueError('Actual function-call identity required')
    arguments=json.loads(call['arguments']) if isinstance(call['arguments'],str) else call['arguments']
    name=call['name']
    expected={'lookup_record':{'project','field'},'cancel_lookup':{'task_ids'},'replace_lookup':{'task_ids','project','field'}}
    if name not in expected or not isinstance(arguments,dict) or set(arguments)!=expected[name]:raise ValueError('Unexpected execution arguments')
    if not isinstance(arguments,dict) or receipt.get('tool')!=name:raise ValueError('Call and actual receipt differ')
    notice=dict(kind='actual_execution_notice',tool=name,receipt_sha256=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    if name in ('lookup_record','replace_lookup'):
        read=receipt if name=='lookup_record' else receipt.get('lookup',{})
        if (not receipt.get('executed') or not read.get('executed') or not read.get('read_only') or read.get('tool')!='lookup_record'
                or read.get('project')!=arguments.get('project') or read.get('field')!=arguments.get('field')):
            raise ValueError('Read call lacks its actual matching receipt')
        notice.update(task_id=read['task_id'],revision=read['revision'],project=read['project'],field=read['field'],
            read_executed=True,value_withheld=True,delivery_state='client-held; do not reissue this read or invent its value')
    if name in ('cancel_lookup','replace_lookup'):
        cancellation=receipt if name=='cancel_lookup' else receipt.get('cancellation',{})
        if (cancellation.get('cancelled') is not True or cancellation.get('task_ids')!=arguments.get('task_ids')
                or {r['task_id'] for r in cancellation.get('receipts',[])}!=set(arguments.get('task_ids',[]))
                or any(r.get('cancelled') is not True or r.get('tool')!='cancel_lookup' for r in cancellation.get('receipts',[]))):
            raise ValueError('Cancellation lacks actual matching target receipts')
        notice.update(cancelled=True,task_ids=cancellation['task_ids'])
    if name not in ('lookup_record','cancel_lookup','replace_lookup'):raise ValueError('Only actual external operations enter control history')
    identity=call['call_id']
    return [dict(role='assistant',control_source='actual-tool-execution',tool_calls=[dict(id=identity,type='function',
        function=dict(name=name,arguments=arguments))]),
        dict(role='tool',control_source='actual-tool-execution',tool_call_id=identity,text=json.dumps(notice))]
