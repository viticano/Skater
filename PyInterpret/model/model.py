class Model(object):
    def __init__(self):
        self.predict = None
        self.training_data = None
        self.unit_tests = []
        self.run_tests()
                
    def unit_test(self,func):
        self.unit_tests.append(func)
        
    def run_tests(self):
        for test in self.unit_tests:
            test()

        

class LocalModel(Model):
    def __init__(self, path):
        self.predict = pickle.load(open(path))        
        super(Model, self).__init__()

        
        
class WebModel(Model):
    def __init__(self, uri, parse_function):
        self.predict = lambda body: parse_function(requests.get(uri, data=body).content)        
        super(Model, self).__init__()
        
    def __confirm_server_is_healthy(self, candidate):
        pass        