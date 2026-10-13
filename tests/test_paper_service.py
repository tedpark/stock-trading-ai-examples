"""Integration boundaries: accounting, causality, persistence and API contracts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from paper_service.engine import make_bars, train, signals, simulate, Risk
# Import-time default app stays in temporary storage during tests.
import os
_import_home=TemporaryDirectory()
_old_home=os.environ.get('PAPER_SERVICE_HOME')
os.environ['PAPER_SERVICE_HOME']=_import_home.name
from paper_service.app import create_app
if _old_home is None:os.environ.pop('PAPER_SERVICE_HOME',None)
else:os.environ['PAPER_SERVICE_HOME']=_old_home
from paper_service.market import rows
from paper_service.store import Store, Conflict

class PaperServiceTests(unittest.TestCase):
    def bars(self):
        return pd.DataFrame({'open':[100.,100.,80.],'close':[100.,80.,100.],'volume':[10]*3},
                            index=pd.date_range('2020-01-01',periods=3))
    def test_cash_and_next_open_fills(self):
        result=simulate(self.bars(),[1,1,0],Risk(1000,1,.5,0,0))
        self.assertEqual([x['equity'] for x in result['timeline']],[1000,800,1000])
        self.assertAlmostEqual(result['metrics']['max_drawdown'],.2)
        self.assertEqual(result['fills'][0]['timestamp'],self.bars().index[1].isoformat())
        self.assertEqual(result['timeline'][-1]['quantity'],0)
    def test_halt_latches_and_liquidates_next_open(self):
        result=simulate(self.bars(),[1,1,1],Risk(1000,1,.1,0,0))
        self.assertEqual(result['metrics']['final_equity'],800)
        self.assertTrue(result['metrics']['halted'])
        self.assertEqual(result['fills'][-1]['reason'],'risk_halt')
    def test_costs_and_cash_reserve(self):
        bars=self.bars();bars['open']=100.;bars['close']=100.
        result=simulate(bars,[1,1,1],Risk(1000,1,.5,10,0))
        self.assertAlmostEqual(result['metrics']['final_equity'],1000-result['metrics']['fees'])
        self.assertGreater(result['metrics']['fees'],0)
        self.assertTrue(all(x['cash']>=-1e-7 for x in result['timeline']))
    def test_training_and_filter_do_not_see_future(self):
        bars=make_bars();bundle,frame,split=train(bars)
        changed=bars.copy();changed.loc[frame.index[split+10]:,'close']*=2
        other,_,_=train(changed)
        np.testing.assert_allclose(bundle['model'].means_,other['model'].means_)
        np.testing.assert_allclose(bundle['scaler'].mean_,other['scaler'].mean_)
        before=signals(bundle,bars);after=signals(bundle,changed)
        np.testing.assert_allclose(before[2][:split+10],after[2][:split+10])
    def test_concurrent_claims_have_one_winner(self):
        with TemporaryDirectory() as d:
            store=Store(Path(d)/'runs.sqlite')
            def claim(_):
                try:store.claim('same','hash');return 'claimed'
                except Conflict:return 'conflict'
            with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(claim,range(4)))
            self.assertEqual(result.count('claimed'),1)
            self.assertEqual(result.count('conflict'),3)
    def test_api_repeat_restart_prediction_and_invalid_data(self):
        with TemporaryDirectory() as d:
            client=TestClient(create_app(d,tracking_uri=''))
            payload={'request_id':'demo'}
            response=client.post('/runs',json=payload);self.assertEqual(response.status_code,200,response.text)
            result=response.json();self.assertEqual(result['forecast_units'],'log_return_per_bar');self.assertEqual(result['tracking']['status'],'disabled')
            self.assertEqual(client.post('/runs',json=payload).json(),result)
            self.assertEqual(client.post('/runs',json={**payload,'seed':10}).status_code,409)
            restarted=TestClient(create_app(d,tracking_uri=''))
            self.assertEqual(restarted.get('/runs/demo').json()['result'],result)
            prediction=restarted.post('/runs/demo/predict',json={'bars':rows(make_bars())})
            self.assertEqual(prediction.status_code,200,prediction.text)
            self.assertEqual(prediction.json()['forecast_units'],'log_return_per_bar')
            self.assertAlmostEqual(sum(prediction.json()['state_probabilities']),1)
            model_file=Path(d)/'models'/'demo'/'model.joblib'
            model_file.write_bytes(b'\x80')  # Truncated pickle header.
            corrupt=restarted.post('/runs/demo/predict',json={'bars':rows(make_bars())})
            self.assertEqual(corrupt.status_code,503)
            self.assertEqual(restarted.get('/runs/demo').json()['status'],'completed')
            import joblib
            joblib.dump({'schema_version':999},model_file)
            wrong_schema=restarted.post('/runs/demo/predict',json={'bars':rows(make_bars())})
            self.assertEqual(wrong_schema.status_code,503)
            model_file.unlink()
            missing=restarted.post('/runs/demo/predict',json={'bars':rows(make_bars())})
            self.assertEqual(missing.status_code,503)
            self.assertNotIn(d,missing.text)
            bad=rows(make_bars());bad[1]['timestamp']=bad[0]['timestamp']
            self.assertEqual(client.post('/runs',json={'request_id':'bad','source':'uploaded','bars':bad}).status_code,422)
            self.assertEqual(client.post('/runs',json={'request_id':'../bad'}).status_code,422)
            self.assertEqual(client.post('/runs',json={'request_id':'bad','max_exposure':2}).status_code,422)
            self.assertEqual(client.get('/runs/missing').status_code,404)
if __name__=='__main__':unittest.main()
