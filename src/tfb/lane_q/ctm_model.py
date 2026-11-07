"""Q05 CTM features within the canonical strongest Q04 learner."""
from .ongoing_runtime import OngoingPredictor
from .ongoing_model import expected_iou_set
from .ctm_features import feature_frame,BUILDER
from .predict import forecast as baseline_forecast

class CTMModel(OngoingPredictor):
    def forecast(self,view):
        if view.condition()!='queue_ongoing':return baseline_forecast(view,self.prior)
        frame=feature_frame(view,self.metadata['panels'][view.panel]['sites'])
        return expected_iou_set(self.probabilities(frame,view.panel)).reshape(6,len(view.network.links))
