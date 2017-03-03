class StaticTypes(object):
    model_types = ModelTypes
    output_types = OutputTypes
    unknown = 'unknown'
    not_applicable = 'not applicable'


class ModelTypes(object):
    regressor = 'regressor'
    classifier = 'classifier'
    unknown = 'unknown'


class OutputTypes(object):
    float = 'float'
    int = 'int'
    string = 'string'
    iterable = 'iterable'
    unknown = 'unknown'


def return_data_type(thing):
    if isinstance(thing, (str, unicode)):
        return StaticTypes.output_types.string
    elif isinstance(thing, int):
        return StaticTypes.output_types.int
    elif isinstance(thing, float):
        return StaticTypes.output_types.float
    elif hasattr(thing, "__iter__"):
        return StaticTypes.output_types.iterable





