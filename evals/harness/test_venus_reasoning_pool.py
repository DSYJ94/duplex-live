from fractions import Fraction
import pytest
from data.venus.reasoning_pool import arithmetic,inspect_answer


def test_source_equations_use_exact_arithmetic_not_python_execution():
    assert arithmetic('(1,200 - 200) / 4')==250
    assert arithmetic('1/3')==Fraction(1,3)
    assert arithmetic('0.10000000000000000001')==Fraction('0.10000000000000000001')
    for expression in ("__import__('os').getcwd()",'2**1000','name+2','True+1'):
        with pytest.raises(ValueError):arithmetic(expression)


def test_annotation_checks_do_not_certify_final_answers_or_silently_round_errors():
    good=inspect_answer('Three each: <<3*4=12>>12.\n#### 12')
    assert good['all_present_annotations_exact'] and not good['full_semantic_verification']
    bad=inspect_answer('Rounded: <<1/3=0.33>>.\n#### 1')
    assert not bad['all_present_annotations_exact']
    assert not inspect_answer('No calculation\n#### 12')['all_present_annotations_exact']
