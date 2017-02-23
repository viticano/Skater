import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from data import DataSet
import datetime
import abc

class ModelInterpreter(object):

    #__metaclass__ = abc.ABCMeta

    def __init__(self, interpretation_type):
        self.type = interpretation_type
        self.data_set = None

    @staticmethod
    def _types():
        return ['pdp','lime']

    def consider(self, training_data, index = None, feature_names = None):
    	
        self.data_set = DataSet(training_data, index = None, feature_names = None)




# Create based on class name:
def interpretation(interpretation_type):
    
    class PartialDependency(ModelInterpreter):
            
        def partial_dependence(self, feature_ids, predict_fn, grid = None, grid_resolution = 100, 
                        grid_range = (0.03, 0.97), sample = False, n_samples = 5000,
                        sampling_strategy = 'uniform-over-similarity-ranks'):
            assert all(feature_id in self.data_set.feature_ids for feature_id in feature_ids), "Pass in a valid ID"  

            #type checking on inputs
        
            #make sure data_set module is giving us correct data structure
            if not grid :
                grid = self.data_set.generate_grid(feature_ids, grid_resolution = grid_resolution, grid_range = grid_range)            

            #make sure data_set module is giving us correct data structure
            X = self.data_set.generate_sample(strategy = sampling_strategy, sample = sample, n_samples_from_dataset = n_samples)                 
            
            pdp_vals = {}
            
            for feature_id, grid_column in zip(feature_ids, grid):
                X_mutable = X.copy()
                pdp_vals[feature_id] = {}
                for value in grid_column:
                    X_mutable[feature_id] = value
                    predictions = predict_fn(X_mutable.values)
                    pdp_vals[feature_id][value] = {'mean':np.mean(predictions), 'std':np.std(predictions)}

            return pdp_vals

        def partial_dependency_sklearn(self):
            pass

    class LocalInterpretation(ModelInterpreter):
     
        def lime_ds(self, data_row, predict_fn, sample = False, 
                            n_samples = 5000, sampling_strategy = 'uniform-over-similarity-ranks',
                            distance_metric='euclidean', kernel_width = None, 
                            explainer_model = None):

            if kernel_width is None:
                kernel_width = np.sqrt(self.data_set.dim) * .75  

            if explainer_model == None:
                explainer_model = LinearRegression

            kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))        

            neighborhood = self.data_set.generate_sample(strategy = sampling_strategy, sample = sample, n_samples_from_dataset = n_samples)
            
            distances = sklearn.metrics.pairwise_distances(
                neighborhood,
                data_row.reshape(1, -1),
                metric=distance_metric) \
                .ravel()

            weights = kernel_fn(distances)
            predictions = predict_fn(neighborhood)
            explainer_model = explainer_model()
            explainer_model.fit(neighborhood, predictions, sample_weight = weights)
            return explainer_model.coef_        
            
        def lime(self):
            pass


    if interpretation_type == "pdp": 
        return PartialDependency(interpretation_type)
    if interpretation_type == "lime": 
        return LocalInterpretation(interpretation_type)
    else:
        raise KeyError("interpretation_type needs to be element of {}".format(ModelInterpreter._types()))
    assert 0, "Bad shape creation: " + type


def main():
    pdp_obj = interpretation_type("pdp")
    pdp_obj.partial_dependency()

    local_obj = interpretation_type("local")
    local_obj.lime()

    var_imp_obj = interpretation_type("var_imp")
    var_imp_obj.default()

if __name__ == '__main__':
    main()