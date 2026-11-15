import json,tempfile,unittest
from pathlib import Path
from training.personaplex.adaptive_evidence import training,development

class Evidence(unittest.TestCase):
    def test_full_epoch_and_incomplete_tail(self):
        with tempfile.TemporaryDirectory()as folder:
            paths=[]
            for rank in range(4):
                row=dict(step=1,rank=rank,world=4,actual_optimizer_updates=1,global_loss=2.,opal=dict(global_preclip_norm=.3,program=dict(update=1)),
                    expanded=[dict(id='train'+str(rank),valid=True,global_example_scale=1.,text_ce=2.,acoustic_ce=3.)],
                    retained=dict(exposure_sha256='a'*64,records=[dict(group='retain',id='old')]))
                path=Path(folder)/str(rank);path.write_text(json.dumps(row)+'\n'+ '{"step":');paths.append(path)
            args=(paths,1,1,{'train'+str(i)for i in range(4)},{('retain','old')})
            coverage,means,receipts=training(*args)
            self.assertEqual(coverage,dict(new_train=4,retained=1));self.assertEqual(means,dict(text=2.,audio=3.))
            self.assertTrue(all(r['prefix_bytes']<Path(r['path']).stat().st_size for r in receipts))
            with self.assertRaises(ValueError):training(paths[:-1],*args[1:])
            row=json.loads(paths[0].read_text().splitlines()[0]);row['expanded'][0]['id']='DEV-leak';paths[0].write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):training(*args)
    def test_dev_mean_and_exact_fold_membership(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'dev.json';report=dict(step=49,rows=[dict(id='d1',text=2.,audio=3.),dict(id='d2',text=4.,audio=5.)],mean=dict(text=3.,audio=4.))
            path.write_text(json.dumps(report));development(path,{'d1','d2'},49)
            for bad in (dict(report,mean=dict(text=1.,audio=4.)),dict(report,step=48),dict(report,rows=[report['rows'][0]]*2)):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):development(path,{'d1','d2'},49)

if __name__=='__main__':unittest.main()
