import numpy as np
from .oracle.oracle import DataOracle


class Interpretation(object):
    def __init__(self, model, data = None):
        self.model = model
        self.oracle = DataOracle(data)
        
        
    def regress_model(self, add_constant = False):

        X = self.oracle.get_population()
        if add_constant:
            X = add_constant(X)
        y = self.model.predict(X)
        regression = GLS(y, X)
        result = regression.fit()
        return result.summary()



    def partial_dependence(self, feature_id):

        assert feature_id in self.oracle.feature_ids, "Pass in a valid ID"        

        
        sample_size = self.oracle.get_reasonable_sample_size()        

        X = self.oracle.sample_from_population(size = sample_size)
        target_feature_values = self.oracle.get_useful_values(feature_id)

        means, stds = [], []

        #print X.shape
        for value in target_feature_values:
            X[:, feature_id] = value
            predictions = self.model.predict(X)
            means.append( np.mean(predictions) ) 
            stds.append( np.std(predictions))

        
        return {'vals': target_feature_values, 'means':means, 'stds':stds}



        