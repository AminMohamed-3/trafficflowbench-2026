"""Prediction/feature future poisoning for the canonical Q-03 model."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from .test_predict import PredictionTests
from .windows import online_view, Release
from .evaluation_cache import EvaluationRelease
from .onset_features import feature_frame, site_table, PriorDayContext, NUMERIC_FEATURES
from .onset_model import OnsetModel, train, choose_cells

class OnsetTests(unittest.TestCase):
    def fixture(self,root):
        helper,panel,cfg=PredictionTests().fixture(root)
        fd=root/'corridors'/panel/'network/fd_parameters.csv'
        frame=pd.read_csv(fd)
        frame['capacity_vph']=[2000.,1800.]
        frame['length_km']=[1.,2.]
        frame.to_csv(fd,index=False)
        prior=cfg['prior']
        prior['panels'][panel]['sites'][0]['n_windows']=3
        sites=site_table(prior['panels'][panel])
        return helper,panel,prior,sites

    def tiny_model(self,frame,panel,prior,sites):
        training=pd.concat([frame]*35,ignore_index=True)
        training['target']=np.tile([1,0],35)
        training['weight']=.5
        booster,categories=train(training,NUMERIC_FEATURES)
        return OnsetModel(booster,{'features':NUMERIC_FEATURES,'categories':categories,'sites':{panel:sites}},prior)

    def test_future_poison_features_and_actual_lgbm_predictions(self):
        for split in ['train','validation','private']:
            for cached in [False,True]:
                with self.subTest(split=split,cached=cached),tempfile.TemporaryDirectory() as temp:
                    root=Path(temp)
                    helper,panel,prior,sites=self.fixture(root)
                    kind=EvaluationRelease if cached else Release
                    release=kind(root)
                    if cached and split=='train':release.train_grid(panel,'2031-01-02')
                    args=(panel,split,'2031-01-02','2031-01-02T11:20:00Z')
                    view=online_view(*args,release=release)
                    before=feature_frame(view,sites)
                    model=self.tiny_model(before,panel,prior,sites)
                    prediction=model.forecast(view)
                    probability=model.probabilities(before)
                    for value in [-9e8,np.nan]:
                        helper.poison(root,split,value,args[3])
                        fresh=kind(root)
                        if cached and split=='train':fresh.train_grid(panel,'2031-01-02')
                        after_view=online_view(*args,release=fresh)
                        after=feature_frame(after_view,sites)
                        pd.testing.assert_frame_equal(before,after,check_exact=True)
                        np.testing.assert_array_equal(probability,model.probabilities(after))
                        # Refresh context so a stale cache cannot conceal leakage.
                        model.context=PriorDayContext()
                        np.testing.assert_array_equal(prediction,model.forecast(after_view))

    def test_classifier_keyed_contract_and_id_order_invariance(self):
        from .predict import predict_queue,KEYS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _,panel,prior,sites=self.fixture(root)
            view=online_view(panel,'private','2031-01-02','2031-01-02T11:20:00Z',release_root=root)
            model=self.tiny_model(feature_frame(view,sites),panel,prior,sites)
            cfg={'release_root':str(root),'prior':prior,'onset_model':model}
            original=predict_queue(panel,'private',cfg).sort_values(KEYS).reset_index(drop=True)
            self.assertEqual(original.queue_pred.dtype,np.dtype(np.int8))
            mapping={'onset_join':'opaque_changed_z','ongoing_join':'opaque_changed_a'}
            dest=root/'task2'/panel/'private'
            for name in ['window_index.csv','window_history.parquet','sample_submission_queue.csv']:
                path=dest/name
                frame=pd.read_parquet(path) if name.endswith('.parquet') else pd.read_csv(path)
                frame.window_id=frame.window_id.map(mapping)
                frame=frame.sample(frac=1,random_state=7).reset_index(drop=True)
                if name.endswith('.parquet'):frame.to_parquet(path,index=False)
                else:frame.to_csv(path,index=False)
            model.context=PriorDayContext()
            after=predict_queue(panel,'private',cfg)
            after.window_id=after.window_id.map({v:k for k,v in mapping.items()})
            after=after.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(original,after,check_exact=True)

    def test_prior_context_first_day_and_backward_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _,panel,_,sites=self.fixture(root)
            context=PriorDayContext()
            late=online_view(panel,'train','2031-01-03','2031-01-03T11:20:00Z',release_root=root)
            context.shares(late,sites)
            first=online_view(panel,'train','2031-01-01','2031-01-01T11:20:00Z',release_root=root)
            self.assertEqual(set(context.shares(first,sites).values()),{0.})
            early=online_view(panel,'train','2031-01-02','2031-01-02T11:20:00Z',release_root=root)
            self.assertEqual(context.shares(early,sites),PriorDayContext().shares(early,sites))

    def test_expected_iou_decision_uses_distribution_and_two_sites(self):
        sites=[{'positions':[0],'links':['A'],'cell_sets':[{'links':['A'],'count':10}]},
               {'positions':[1],'links':['B'],'cell_sets':[{'links':['B'],'count':10}]}]
        self.assertEqual(choose_cells(sites,[.9,.1]),('A',))
        self.assertEqual(choose_cells(sites,[.9,.9]),('A','B'))
        self.assertEqual(choose_cells(sites,[.1,.9]),('B',))

if __name__=='__main__':unittest.main(verbosity=2)
