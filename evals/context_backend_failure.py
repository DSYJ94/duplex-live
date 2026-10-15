"""Expose a rejected backend request without executing it or claiming success."""


def notice(error, task_id):
    from evals.recovery_worker import WorkerContractError
    if not isinstance(error, WorkerContractError):
        raise TypeError('Only explicit backend contract rejections are handled here')
    if str(error).startswith('Private project was not identified by the user'):
        content = ('The backend cannot identify the project from this request. '
            'Ask the user which project they mean. No private read was authorized.')
    else:
        content = ('The backend did not provide a usable verified reply for this request. '
            'Do not claim it completed successfully or invent its answer.')
    return {'route': 'error', 'content': content, 'task_id': task_id,
        'backend_contract_error': str(error), 'original_failure': error.result,
        'receipts': error.result.get('receipts', []), 'speech_model_routing': False}
