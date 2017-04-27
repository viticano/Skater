"""Static types class and methods for inferring types."""
class ModelTypes(object):
    """Stores values for model types and keywords"""
    regressor = 'regressor'
    classifier = 'classifier'
    unknown = 'unknown'


class OutputTypes(object):
    """Stores values for output types, and keywords"""
    float = 'float'
    int = 'int'
    string = 'string'
    iterable = 'iterable'
    unknown = 'unknown'


class StaticTypes(object):
    """Stores values for model types, output types, and keywords"""
    model_types = ModelTypes
    output_types = OutputTypes
    unknown = 'unknown'
    not_applicable = 'not applicable'

def return_data_type(thing):
    """Returns an output type given a variable"""
    if isinstance(thing, (str, unicode)):
        return StaticTypes.output_types.string
    elif isinstance(thing, int):
        return StaticTypes.output_types.int
    elif isinstance(thing, float):
        return StaticTypes.output_types.float
    elif hasattr(thing, "__iter__"):
        return StaticTypes.output_types.iterable
    else:
        return StaticTypes.unknown
