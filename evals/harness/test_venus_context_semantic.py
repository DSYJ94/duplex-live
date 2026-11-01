import copy,json,unittest
from data.venus.context_semantic_review import admission,validate

class SemanticTests(unittest.TestCase):
 def setUp(self):
  self.body,self.grant=admission();self.verdicts={k:dict(semantic_correct=True,planned_causality_consistent=True,concern='') for k in self.body['tools'][0]['parameters']['properties']}
 def response(self,results):return dict(model='gpt-6-astra',status='completed',id='fixture',output=[dict(type='function_call',name='review_teacher_cohort',arguments=json.dumps(results))])
 def test_whole36_and_reservation_fit(self):
  self.assertEqual(len(validate(self.response(self.verdicts),self.body)),36);self.assertLess(self.grant['conservative_reservation_usd'],3)
 def test_no_subset_extra_or_wrong_model_acceptance(self):
  v=copy.deepcopy(self.verdicts);v.pop(next(iter(v)))
  with self.assertRaises(ValueError):validate(self.response(v),self.body)
  r=self.response(self.verdicts);r['model']='other'
  with self.assertRaises(ValueError):validate(r,self.body)
