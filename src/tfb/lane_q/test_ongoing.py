"""Q-04 online causal-feature, actual predictor and expected-IoU contracts."""
import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from .runtime import pin_lightgbm
pin_lightgbm()
from . import test_onset as onset_tests
from .windows import Release,online_view
from .evaluation_cache import EvaluationRelease
from .ongoing_features import feature_frame,static_sites,FEATURES
from .ongoing_model import OngoingModel,train,expected_iou_set

class OngoingTests(unittest.TestCase):
    def fixture(self,root):
        helper,panel,prior,_=onset_tests.OnsetTests().fixture(root)
        sites=static_sites(prior['panels'][panel])
        return helper,panel,prior,sites

    def tiny_model(self,frame,panel,prior,sites):
        fit=pd.concat([frame]*60,ignore_index=True)
        target=((frame.last_queued.gt(0)) & frame.step.le(3)).astype(np.float32).to_numpy()
        fit['target']=np.tile(target,60);fit['weight']=1/len(frame)
        booster,categories=train(fit,FEATURES)
        return OngoingModel({panel:booster},{'features':FEATURES,'panels':{panel:{'categories':categories,'sites':sites}}},prior)

    def test_future_poison_all_splits_features_probabilities_predictions(self):
        for split in ['train','validation','private']:
            for cached in [False,True]:
                with self.subTest(split=split,cached=cached),tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);helper,panel,prior,sites=self.fixture(root)
                    loader=EvaluationRelease if cached else Release
                    release=loader(root)
                    if cached and split=='train':release.train_grid(panel,'2031-01-02')
                    args=(panel,split,'2031-01-02','2031-01-02T12:00:00Z')
                    view=online_view(*args,release=release)
                    before=feature_frame(view,sites)
                    model=self.tiny_model(before,panel,prior,sites)
                    probs=model.probabilities(before,panel);pred=model.forecast(view)
                    for value in [-9e8,np.nan]:
                        helper.poison(root,split,value,args[3])
                        fresh=loader(root)
                        if cached and split=='train':fresh.train_grid(panel,'2031-01-02')
                        after_view=online_view(*args,release=fresh)
                        after=feature_frame(after_view,sites)
                        pd.testing.assert_frame_equal(before,after,check_exact=True)
                        np.testing.assert_array_equal(probs,model.probabilities(after,panel))
                        np.testing.assert_array_equal(pred,model.forecast(after_view))
                    self.assertEqual(pred.dtype,np.dtype(np.int8))
                    self.assertEqual(pred.shape,(6,2))

    def test_keyed_forecast_is_invariant_to_window_ids_and_rows(self):
        from .predict import predict_queue,KEYS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,panel,prior,sites=self.fixture(root)
            view=online_view(panel,'private','2031-01-02','2031-01-02T12:00:00Z',release_root=root)
            model=self.tiny_model(feature_frame(view,sites),panel,prior,sites)
            cfg={'release_root':str(root),'ongoing_model':model}
            original=predict_queue(panel,'private',cfg).sort_values(KEYS).reset_index(drop=True)
            mapping={'onset_join':'renamed_q','ongoing_join':'renamed_o'}
            for name in ['window_index.csv','window_history.parquet','sample_submission_queue.csv']:
                path=root/'task2'/panel/'private'/name
                frame=pd.read_parquet(path) if name.endswith('.parquet') else pd.read_csv(path)
                frame.window_id=frame.window_id.map(mapping)
                frame=frame.sample(frac=1,random_state=91).reset_index(drop=True)
                if name.endswith('.parquet'):frame.to_parquet(path,index=False)
                else:frame.to_csv(path,index=False)
            after=predict_queue(panel,'private',cfg)
            after.window_id=after.window_id.map({v:k for k,v in mapping.items()})
            pd.testing.assert_frame_equal(original,after.sort_values(KEYS).reset_index(drop=True),check_exact=True)
            self.assertEqual(original.queue_pred.dtype,np.dtype(np.int8))

    def test_expected_iou_matches_exhaustive_Bernoulli_truth(self):
        for p in [np.array([.1,.2,.8,.9]),np.array([.03,.07,.1]),np.array([.8,.8,.8]),np.array([0.,0.,0.])]:
            n=len(p);order=np.argsort(-p,kind='stable');values=[]
            for m in range(1,n+1):
                chosen=set(order[:m]);expectation=0.
                for bits in itertools.product([0,1],repeat=n):
                    truth={i for i,b in enumerate(bits) if b}
                    probability=np.prod([p[i] if b else 1-p[i] for i,b in enumerate(bits)])
                    expectation+=probability*len(chosen&truth)/len(chosen|truth)
                values.append(expectation)
            expected=min(range(1,n+1),key=lambda m:(-values[m-1],m))
            prediction=expected_iou_set(p)
            self.assertEqual(int(prediction.sum()),expected)
            self.assertEqual(set(np.flatnonzero(prediction)),set(order[:expected]))

    def test_queue_state_velocity_and_history_causal_sanity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,panel,_,sites=self.fixture(root)
            view=online_view(panel,'train','2031-01-02','2031-01-02T12:00:00Z',release_root=root)
            frame=feature_frame(view,sites)
            self.assertTrue(frame.iloc[::2].last_queued.eq(1).all())
            self.assertTrue(frame.iloc[::2].history_queue_count.gt(0).all())
            self.assertTrue(frame.component_present.eq(1).all())
            self.assertTrue(frame.iloc[::2].tail_distance_km.eq(0).all())
            self.assertTrue(frame.iloc[::2].head_distance_km.eq(0).all())
            self.assertFalse(np.isinf(frame[FEATURES].to_numpy()).any())

if __name__=='__main__':unittest.main(verbosity=2)
