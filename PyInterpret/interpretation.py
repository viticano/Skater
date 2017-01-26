from .oracle.oracle import DataOracle
class Interpretation(object):
    def __init__(self, model, data = None):
        self.model = model
        self.oracle = DataOracle(data)
        self.util = Util(self.oracle)
        
        
    def regress_model(self, add_constant = False):
        X = self.oracle.sample_data()
        if add_constant:
            X = add_constant(X)
        y = self.model.predict(X)
        regression = GLS(y, X)
        result = regression.fit()
        return result.summary()