"""Prepare a provenance-bound GSM8K TRAIN pool, not an approved native corpus.

Arithmetic annotation checks are partial evidence, never semantic certification.
"""
import argparse
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


def arithmetic(text):
    source=text.replace(',','')
    tree=ast.parse(source,mode='eval')
    if len(list(ast.walk(tree)))>80:raise ValueError('Bounded arithmetic only')
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            value=Fraction(ast.get_source_segment(source,node))
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            value=visit(node.operand)*(-1 if isinstance(node.op,ast.USub) else 1)
        elif isinstance(node,ast.BinOp):
            left,right=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):value=left+right
            elif isinstance(node.op,ast.Sub):value=left-right
            elif isinstance(node.op,ast.Mult):value=left*right
            elif isinstance(node.op,ast.Div):value=left/right
            elif isinstance(node.op,ast.Pow) and right.denominator==1 and abs(right)<=8:value=left**int(right)
            else:raise ValueError('Unsupported arithmetic operation')
        else:raise ValueError('Only numeric arithmetic; no executable names or calls')
        if abs(value)>10**18:raise ValueError('Arithmetic outside bounded source check')
        return value
    return visit(tree)


def inspect_answer(answer):
    parts=answer.rsplit('####',1)
    if len(parts)!=2:raise ValueError('Original final-answer delimiter required')
    final=parts[1].strip()
    if not re.fullmatch(r'-?[\d,]+(?:\.\d+)?',final):raise ValueError('Original numeric target required')
    final_value=arithmetic(final);checks=[]
    for annotation in re.findall(r'<<([^<>]+)>>',parts[0]):
        item=dict(annotation=annotation,exact=False)
        try:
            left,right=annotation.rsplit('=',1)
            item['exact']=arithmetic(left)==arithmetic(right)
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError) as error:
            item['diagnostic']=type(error).__name__
        checks.append(item)
    return dict(original_final=final,numeric_target=str(final_value),annotation_checks=checks,
        all_present_annotations_exact=bool(checks) and all(c['exact'] for c in checks),
        full_semantic_verification=False)


def main(a):
    if a.output.exists():raise ValueError('Preserve prior source-pool versions')
    if not re.fullmatch('[0-9a-f]{40}',a.revision):raise ValueError('Exact public source commit required')
    license_text=a.license.read_text()
    if not license_text.startswith('MIT License\n') or 'Copyright (c) 2021 OpenAI' not in license_text:
        raise ValueError('Original public repository license required')
    a.output.mkdir(parents=True)
    rows=[];seen=set()
    for index,line in enumerate(a.source.read_text().splitlines()):
        original=json.loads(line)
        if set(original)!={'question','answer'}:raise ValueError('Original GSM8K schema required')
        normalized=' '.join(original['question'].lower().split())
        identity=hashlib.sha256(normalized.encode()).hexdigest()
        if identity in seen:raise ValueError('Duplicate source question; diagnose before selection')
        seen.add(identity)
        rows.append(dict(id='gsm8k-train-'+identity[:20],source_row=index,source_split='train',
            question=original['question'],original_solution=original['answer'],
            **inspect_answer(original['answer']),training_approved=False))
    with (a.output/'pool.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(schema='dl-public-reasoning-source-pool-v1',repo='openai/grade-school-math',revision=a.revision,
        source_path='grade_school_math/data/train.jsonl',source_split='train',source_sha256=sha(a.source),
        license='MIT',license_file_sha256=sha(a.license),pool_sha256=sha(a.output/'pool.jsonl'),
        rows=len(rows),all_annotations_exact=sum(r['all_present_annotations_exact'] for r in rows),
        annotation_check_not_full_semantic_verification=True,official_test_split_downloaded=False,
        fresh_eval_decontamination_pending=True,native_audio_created=False,training_approved=False)
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','license','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--revision',required=True);main(p.parse_args())
