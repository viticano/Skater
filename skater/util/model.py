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
        transformer = model.transformer
        predict_fn = partial(DeployedModel._predict,
                             uri=uri,
                             input_formatter=input_formatter,
                             output_formatter=output_formatter,
                             transformer=transformer)
        return predict_fn
    elif isinstance(model, InMemoryModel):
        transformer = model.transformer
        input_formatter = model.input_formatter
        output_formatter = model.output_formatter
        prediction_fn = model.prediction_fn
        predict_fn = partial(InMemoryModel._predict,
                             transformer=transformer,
                             predict_fn=prediction_fn,
                             input_formatter=input_formatter,
                             output_formatter=output_formatter,
                             )
        return predict_fn
    else:
        return None


def identity_function(x):
    return x
