"""Behavioral checks import the actual source modules used by manuscript excerpts."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
import torch
from quantlab import data, regime, kalman, statistics, risk, pairs, training, distribution, serving, broker

class RuntimeComponentTests(unittest.TestCase):
    def test_training_only_scaling(self):
        a,b,_=data.scale_training_only([[1.],[3.]],[[100.]])
        np.testing.assert_allclose(a,[[-1.],[1.]])
        self.assertEqual(b[0,0],98)
        np.testing.assert_array_equal(data.discard_warmup([[np.nan,1],[2,3]]),[[2,3]])
        data.configure_logging();self.assertEqual(training.decision_demo(),['hold','cash'])
        with self.assertRaises(ValueError):training.model([np.nan])

    def test_ohlcv_validation(self):
        frame=pd.DataFrame({'Open':[10,11],'High':[12,12],'Low':[9,10],'Close':[11,12],'Volume':[100,200]},index=pd.date_range('2020',periods=2))
        self.assertEqual(str(data.download_ohlcv('X',downloader=lambda *a,**kw:frame).index.tz),'UTC')
        for col,val in [('High',1),('Volume',-1),('Close',np.nan)]:
            bad=frame.copy();bad[col]=val
            with self.subTest(col=col),self.assertRaises(ValueError):data.download_ohlcv('X',downloader=lambda *a,**kw:bad)

    def test_hmm_streaming_agrees_with_batch(self):
        x=np.random.default_rng(42).normal(size=(100,2));m=regime.DailyHMM();self.assertTrue(m.fit(x[:70]))
        full=m.predict_proba(x)
        np.testing.assert_allclose([m.step_forward(row) for row in x[70:]],full[70:],rtol=1e-7,atol=1e-9)
        np.testing.assert_array_equal(regime.rank_states(m.model),m.order)
        self.assertEqual(regime.state_return_means(m.scaler,m.model).shape,(3,))
        np.testing.assert_allclose(regime.walk_forward(x[:65]),regime.walk_forward(np.r_[x[:65],x[65:]*100])[:5])
        s=training.SafeSignal();s.retrain(x[:70]);self.assertEqual(s.predict(x[70:]).shape,(30,3))

    def test_signal_gates(self):
        self.assertEqual(regime.regime_signal([.8,.1,.1],-2),1)
        self.assertEqual(regime.regime_signal([.2,.1,.7],-2),0)
        self.assertAlmostEqual(regime.regime_multiplier([.8,.1,.1]),1.35)
        self.assertEqual(regime.blend_votes(2,1,5,10),0)
        self.assertEqual(regime.blend_votes(2,.5,8,10),1)
        self.assertEqual(regime.pipeline_direction([.8,.1,.1],-2),1)
        self.assertEqual(regime.pipeline_direction([.1,.1,.8],-2),-1)
        self.assertEqual(regime.pipeline_direction([.2,.6,.2],-2),0)

    def test_filters_and_risk(self):
        f=kalman.KalmanFilter1D(Q=0,R=1);self.assertEqual(f.update(2),1)
        self.assertEqual(kalman.kalman_zscores(np.exp(np.arange(80)*.001)).shape,(80,))
        self.assertAlmostEqual(statistics.compute_zscore([1,2,3,4,100],3)[3],2/np.std([1,2,3]))
        self.assertAlmostEqual(statistics.kelly_fraction(.6,0,'equity'),.1)
        self.assertEqual(risk.calculate_kelly_size(.6,1,macro_gate=0),0)
        self.assertEqual(risk.hard_stop(-.03).action,-1);self.assertIsNone(risk.hard_stop(0))
        self.assertAlmostEqual(risk.atr_stop_loss(2,100),-.03)
        self.assertAlmostEqual(risk.drawdowns([100,80,100])[1],.2)
        self.assertAlmostEqual(risk.return_tail([-.4,-.3,.2,.5],.375)[1],-.55/1.5)
        self.assertEqual(risk.return_tail([-.4,.4],1),(.4,0.))
        store=risk.HighWatermarks();self.assertEqual(store.save('X',100),100);self.assertEqual(store.save('X',90),100)
        with self.assertRaises(ValueError):store.save('X',float('nan'))

    def test_pair_selection_and_spread(self):
        x=np.arange(30.)+100;noise=np.random.default_rng(7).normal(size=30)
        self.assertTrue(0<=pairs.cointegration_pvalue(x,x+noise)<=1)
        frame=pd.DataFrame({'a':x,'b':2*x+noise})
        with patch('quantlab.pairs.cointegration_pvalue',return_value=.001),patch('quantlab.pairs.calc_halflife',return_value=2):
            self.assertEqual(len(pairs.find_cointegrated_pairs(frame,{'s':['a','b'],'duplicate':['a','b']},min_correlation=-1)),1)
        np.testing.assert_array_equal(pairs.log_spread([2,3],[1,1],2),[0,1])
        self.assertEqual(pairs.holding_limit(.8,4+6*.5**np.arange(30)),2)
        self.assertIsNone(pairs.holding_limit(.1,4+6*.5**np.arange(30)))

    def test_objectives(self):
        trial=SimpleNamespace(suggest_float=lambda *a:.5)
        self.assertAlmostEqual(training.threshold_objective(trial,[1,-1,0,999],[.1,.2,.3,-999],1,3),-.104)
        self.assertTrue(np.isfinite(training.vote_objective(trial,[.9,.1],[.1,.2])))
        self.assertEqual(training.vote_objective(trial,[.5,.5],[0,0]),-float('inf'))
        self.assertEqual(training.create_study().direction.name,'MAXIMIZE')

    def test_quantile_risk_and_exit(self):
        torch.manual_seed(7);net=distribution.QuantileNet(state_dim=11,n_quantiles=5,hidden=8)
        agent=distribution.QuantileAgent(net);state=torch.zeros((1,11))
        self.assertEqual(agent.act(state).shape,(1,));self.assertEqual(agent.cvar(state,1).shape,(1,))
        cfg=SimpleNamespace(cvar_target=0.,cvar_scale=1.,size_floor=.1)
        result=distribution.apply_cvar_to_signal(risk.Signal(-1,size=2),distribution.CVaRSizer(agent,cfg),state)
        self.assertEqual(result.size,2);self.assertEqual(result.metadata['cvar_scale'],1)
        self.assertEqual(distribution.cvar_position_scaler(-2,cfg),.1)
        with self.assertRaises(ValueError):agent.cvar(state,3)

    def test_broker_preview_and_account_scoping(self):
        async def run():
            b=broker.PaperBrokerBackend();s=broker.BrokerSession(b);self.assertTrue(await s.connect())
            self.assertEqual((await s.session_info('X'))['timezone'],'UTC')
            self.assertEqual((await s.preview_order('X','BUY',2))['mode'],'preview')
            b.open=False;self.assertFalse((await s.preview_order('X','BUY',2))['success'])
            self.assertEqual(len(await s.positions('paper')),1);self.assertEqual(await s.positions('other'),[])
            self.assertEqual((await s.account_summary('paper'))['NetLiquidation']['USD'],10000)
            with self.assertRaises(ValueError):await s.preview_order('X','BUY',float('nan'))
        asyncio.run(run())

    def test_real_sac_training_saving_loading(self):
        returns=np.random.default_rng(7).normal(0,.002,32);agent=serving.train_and_track(returns,steps=40)
        env=training.make_environment(returns);state,_=env.reset()
        self.assertTrue(np.isfinite(training.rollout(env,agent)))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'model.zip';agent.save(path)
            np.testing.assert_allclose(training.load_and_act(path,state),agent.predict(state,deterministic=True)[0])

    def test_serving_reload_preserves_good_model(self):
        good=torch.nn.Sequential(torch.nn.Linear(11,1),torch.nn.Tanh())
        def loader(path):
            if path=='bad':raise ValueError('bad')
            return good
        async def run():
            s=serving.SACServingState('good',loader);self.assertIn(await s.predict([0.]*11),[-1,0,1])
            with self.assertRaises(ValueError):await s.reload('bad')
            self.assertEqual(s.checkpoint_path,'good')
            await serving.download_and_reload('run','model',s,lambda **kw:'new');self.assertEqual(s.checkpoint_path,'new')
            with self.assertRaises(ValueError):await s.predict([0]*6)
        asyncio.run(run())
        compiled=serving.compile_jit(good)
        np.testing.assert_allclose(serving.predict_actor(compiled,[0.]*11).numpy(),good(torch.zeros(1,11)).detach().numpy(),atol=1e-7)

    def test_tracking_failure_is_distinct_from_training_failure(self):
        class Backend:
            def set_experiment(self,*a):pass
            def start_run(self,**kw):return object()
            def log_params(self,*a):pass
            def set_tags(self,*a):pass
            def log_metrics(self,*a,**kw):raise RuntimeError('offline')
            def end_run(self,**kw):self.status=kw['status']
        backend=Backend()
        with self.assertLogs(level='ERROR'):
            with serving.MlflowTracker('x','x',{}, {},backend=backend) as tracker:tracker.log_metrics({},0)
        self.assertEqual(backend.status,'FAILED')
        with self.assertRaisesRegex(RuntimeError,'training'):
            with serving.MlflowTracker('x','x',{}, {},backend=backend):raise RuntimeError('training')

    def test_shared_calculations_from_imported_implementation(self):
        from ch02_features import compute_daily_features
        from ch12_practice import compute_entry_score
        self.assertAlmostEqual(statistics.fractional_kelly(.01,.04),.0625)
        self.assertAlmostEqual(statistics.compute_sortino([-.1,.2],periods_per_year=1),2**-.5)
        np.testing.assert_allclose(compute_daily_features([100.]*25,[10.]*25)[20],[0,0,0,1])
        self.assertAlmostEqual(compute_entry_score(-2.,10.,.5,1.),.1)
        previous=None
        for high in [102.99,103.,109.99,110.,120.]:
            stop=risk.ratchet_long_stop(previous,high,risk._get_tiered_trailing_stop(None,high/100-1))
            if previous is not None:self.assertGreaterEqual(stop,previous)
            previous=stop

    def test_source_must_be_executable_and_symbol_must_exist(self):
        from sync_book import source_text, contract
        e=dict(contract()[0]);e['source']='book_sources/index.json'
        with self.assertRaises(ValueError):source_text(e)
        e=dict(contract()[0]);e['symbol']='missing'
        with self.assertRaises(ValueError):source_text(e)

if __name__=='__main__':unittest.main()
