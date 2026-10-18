import unittest
from evals.qwen_response_executor import validate


class LocalSchema(unittest.TestCase):
    def test_missing_extra_wrong_type_and_wrong_enum_are_rejected(self):
        schema=dict(type='object',properties=dict(route=dict(type='string',enum=['answer','clarify']),content=dict(type='string'),
            consume=dict(type='array',items=dict(type='string'))),required=['route','content','consume'],additionalProperties=False)
        valid=dict(route='answer',content='Three.',consume=[]);validate(valid,schema)
        for value in [dict(route='answer',content='Three.'),dict(valid,extra='unavailable'),dict(valid,route='delegate'),dict(valid,consume='other'),dict(valid,content=None)]:
            with self.subTest(value=value),self.assertRaises(ValueError):validate(value,schema)


if __name__=='__main__':unittest.main()
