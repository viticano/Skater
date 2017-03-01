import abc

class Model(object):
    """
    What is a model? A model needs to make predictions, so a means of
    passing data into the model, and receiving results.

    Goals:
        We want to abstract away how we access the model.
        We want to make inferences about the format of the output.
        We want to able to map model outputs to some smaller, universal set of output types.
        We want to infer whether the model is real valued, or classification (n classes?)

    """


    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def predict(self,*args, **kwargs):
        return 

    @staticmethod
    def infer_data_types(data):
        pass
        
    @classmethod
    def get_output_signature(self, *args, **kwargs):
        inputs = self.get_inputs_for_output_signature()
        outputs = self.predict(inputs)
        output_type  = self.infer_data_types(outputs)
        return output_type



class LocalModel(Model):
    def __init__(self, prediction_fn = None, model_path = None):
        super(Model, self).__init__() 
        prediction_fn = prediction_fn or self.load_from_object(model_path)


    def load_from_object(self,object):
        self.predict = object
        super(Model, self).__init__()        

    def predict(self, input):
        output = self.prediction_fn(input)
        return output          

        
        
class WebModel(Model):
    def __init__(self, uri, parse_function = None):
        super(Model, self).__init__()
        self.uri = uri
        self.parse_function = parse_function or lambda x: x
    
    def predict(self, request_body):
        output = parse_function(requests.get(uri, data=request_body).content)  
        return output              
        
    def __confirm_server_is_healthy(self, candidate):
        pass        
