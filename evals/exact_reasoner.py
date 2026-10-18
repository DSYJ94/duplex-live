"""Experimental Astra delegation with an optional terminal exact-count action.

One HTTP request per delegation, preserving existing request/output bounds.
When the model selects the terminal action, the actual computed receipt is
voiced; no guessed numeric answer is patched. Other answers remain unverified
model text. This profile is not enabled in the owner demo.
"""
import json
import os
from pathlib import Path
import time
from evals.exact_count_tool import definition, execute
from evals.moshi_model_download import sha256


def ask_exact_reasoner(question):
    import requests
    if not isinstance(question, str) or not question.strip() or len(question) > 16384:
        raise ValueError('Invalid bounded original reasoning question')
    started = time.monotonic()
    response = requests.post('https://api.openai.com/v1/responses',
        headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']},
        json=dict(model='gpt-6-astra', store=False, max_output_tokens=4096, parallel_tool_calls=False,
                  tools=[definition()], tool_choice='auto',
                  instructions='Solve the actual user problem carefully. For a request asking only for the count of subsets of distinct positive integers, select the terminal exact count action with the complete original list and target quoted faithfully. The application will compute and voice that exact result. Do not guess the count mentally. For other questions, answer in at most35 spoken words, using natural words for numbers, no Markdown, formulas, plans or internal metadata. Do not select a count action for another operation or when an explanation is required.',
                  input=[dict(role='user', content=question)]), timeout=180)
    if response.status_code != 200:
        raise RuntimeError('Exact reasoning backend HTTP ' + str(response.status_code))
    data = response.json()
    if data.get('status') != 'completed':
        raise RuntimeError('Exact reasoning backend did not complete within its token bound')
    calls = [x for x in data.get('output', []) if x['type'] == 'function_call']
    texts = [p['text'] for m in data.get('output', []) if m['type'] == 'message' for p in m['content'] if p['type'] == 'output_text']
    tool_receipts = []
    if calls:
        if len(calls) != 1 or calls[0]['name'] != 'answer_subset_count' or any(t.strip() for t in texts):
            raise ValueError('Terminal count requires one action and no competing final answer')
        receipt = execute(question, json.loads(calls[0]['arguments']))
        receipt.update(api_call_id=calls[0]['call_id'], producer_response_id=data['id'])
        tool_receipts.append(receipt)
        content, source = receipt['content'], 'actual-terminal-exact-count-receipt'
    else:
        content, source = ''.join(texts).strip(), 'unverified-model-text'
        if not content:
            raise ValueError('Reasoner returned neither a terminal action nor an answer')
    finished = time.monotonic()
    return dict(content=content, response_id=data['id'], model=data['model'], usage=data.get('usage'),
                seconds=finished-started, request_started_monotonic=started, request_finished_monotonic=finished,
                remote_api_used=True, actual_http_requests=1, exact_tool_receipts=tool_receipts, result_source=source,
                output_repaired=False, original_question_unchanged=True,
                exact_tool_code_sha256=sha256(Path(__file__).with_name('exact_count_tool.py')),
                reasoning_profile_code_sha256=sha256(Path(__file__)))
