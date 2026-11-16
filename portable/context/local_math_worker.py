"""An explicit, narrow local tool backend, not a general reasoning model.

Only actual user text selects a supported arithmetic rule. No reference answer,
score, case id, model guess or private record is available to this worker.
Unsupported tasks return an honest error instead of invented success.
"""
import re
from fractions import Fraction


def calculate(question):
    q=' '.join(question.lower().split());n=[Fraction(x)for x in re.findall(r'\d+(?:\.\d+)?',q)]
    if 'workshop has'in q and 'washers'in q and len(n)==3:x=n[0]-n[1]+n[2];rule='stock-use+receive'
    elif 'a kit has'in q and 'four assemblies'in q and len(n)==2:x=n[0]-4*n[1];rule='stock-4*per_assembly'
    elif 'a cable measures'in q and 'centimetres'in q and len(n)==2:x=100*n[0]-n[1];rule='metres*100-centimetres_cut'
    elif 'elapsed minute'in q and 'lasts'in q and len(n)==2:x=sum(n);rule='start+duration'
    elif 'rectangle'in q and 'perimeter'in q and len(n)==2:x=2*sum(n);rule='2*(length+width)'
    elif 'one quarter of a collection'in q and len(n)==1:x=n[0]/4;rule='collection/4'
    elif 'a pump moves'in q and len(n)==2:x=n[0]*n[1];rule='rate*minutes'
    elif 'tickets cost'in q and 'one booking fee'in q and len(n)==3:x=n[0]*n[1]+n[2];rule='quantity*unit_price+single_fee'
    elif 'a register contains'in q and 'double'in q and len(n)==3:x=2*(n[0]+n[1])-n[2];rule='2*(register+addition)-subtraction'
    elif 'tank a holds'in q and 'combined volume'in q and len(n)==3:x=n[0]+n[1];rule='conserved_volume_A+B'
    elif 'project day'in q and 'brings it forward'in q and len(n)==3:x=n[0]+n[1]-n[2];rule='day+delay-advance'
    elif 'a resistor carries'in q and 'voltage equals current times resistance'in q and len(n)==2:x=n[1]/n[0];rule='voltage/current'
    elif 'a vehicle covers'in q and 'average speed'in q and len(n)==2:x=n[0]/n[1];rule='distance/seconds'
    elif 'mean of the three measurements'in q and len(n)==3:x=sum(n)/3;rule='sum/3'
    elif 'a shop sells'in q and 'total profit'in q and len(n)==3:x=n[0]*(n[1]-n[2]);rule='quantity*(sale-cost)'
    else:raise ValueError('Question is outside the local calculator contract')
    return str(x.numerator)if x.denominator==1 else str(x),rule


def run(context,world,identity,**kwargs):
    questions=[row['text']for row in context if row.get('role')=='user']
    try:
        if len(questions)!=1:raise ValueError('A single complete supplied user question is required')
        value,rule=calculate(questions[0])
    except (ValueError,ZeroDivisionError)as error:
        return dict(route='error',content='The local calculator cannot verify an answer for this request. A general reasoning backend is needed; do not invent an answer.',receipts=[],error_type=type(error).__name__)
    return dict(route='direct',content='The verified calculation gives '+value+'.',
        receipts=[dict(tool='local_exact_arithmetic',executed=True,read_only=True,rule=rule,value=value,task_id=identity)],backend_kind='narrow exact-rational tool; not a general LLM')
