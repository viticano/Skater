from functools import partial

from ..model.deploy import DeployedModel
from ..model.local import InMemoryModel


def get_predictor(model):
    """Takes a model object, creates an independent prediction
    function

    Parameters
    ----------
    model: subtype of skater.model.model.Model
        Either skater.model.remote.DeployedModel
        or skater.model.local.InMemoryModel

    Returns:
    ----------
    predict_fn: callable
        Static prediction function that takes (data) as an argument.


    """
    if isinstance(model, DeployedModel):
        uri = model.uri
        input_formatter = model.input_formatter
        output_formatter = model.output_formatter
        formatter = model.formatter
        predict_fn = partial(DeployedModel._predict,
                             uri=uri,
                             input_formatter=input_formatter,
                             output_formatter=output_formatter,
                             formatter=formatter)
        return predict_fn
    elif isinstance(model, InMemoryModel):
        formatter = model.formatter
        prediction_fn = model.prediction_fn
        predict_fn = partial(InMemoryModel._predict,
                             formatter=formatter,
                             predict_fn=prediction_fn)
        return predict_fn
    else:
        return None
