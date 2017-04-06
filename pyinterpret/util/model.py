from functools import partial


from ..model.remote import DeployedModel
from ..model.local import InMemoryModel

def build_static_predictor(model):
    if isinstance(model, DeployedModel):
        uri = model.uri
        input_formatter = model.input_formatter
        output_formatter = model.output_formatter
        formatter=model.formatter
        predict_fn = partial(DeployedModel.static_predict,
                             uri=uri,
                             input_formatter=input_formatter,
                             output_formatter=output_formatter,
                             formatter=formatter)
        return predict_fn
    elif isinstance(model, InMemoryModel):
        formatter = model.formatter
        prediction_fn = model.prediction_fn
        predict_fn = partial(InMemoryModel.static_predict,
                             formatter=formatter,
                             predict_fn=prediction_fn)
        return predict_fn
