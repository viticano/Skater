import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from data import DataSet

class Explainer(object):
	def __init__(self, training_data, n_percentiles = 100, training_labels=None, feature_names=None,
				 categorical_features=None, categorical_names=None,
				 kernel_width=None, verbose=False, class_names=None):
		
		if kernel_width is None:
			kernel_width = np.sqrt(training_data.shape[1]) * .75	
		
		self.DataSet = DataSet(training_data, n_percentiles = n_percentiles)
		self.kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
		self.explainer_model = LinearRegression

	def explain_instance(self, data_row, classifier_fn, labels=(1,),
							 top_labels=None, num_features=10, n=5000, samples_per_bin = 10, 
							 distance_metric='euclidean', model_regressor=None):



			neighborhood = self.DataSet.generate_sample(samples_per_bin = samples_per_bin)

			distances = sklearn.metrics.pairwise_distances(
			neighborhood,
			data_row.reshape(1, -1),
			metric=distance_metric).ravel()
			weights = self.kernel_fn(distances)
			predictions = classifier_fn(neighborhood)
			explainer_model = self.explainer_model()
			explainer_model.fit(neighborhood, predictions, sample_weights = weights)
			return explainer_model.coef_




    def partial_dependence(self, feature_id, samples_per_bin = 10):

        assert feature_id in self.DataSet.feature_ids, "Pass in a valid ID"        

        
        

        X = self.DataSet.generate_sample(samples_per_bin = samples_per_bin)
        target_feature_values = self.DataSet.percentiles(:, feature_id)

        means, stds = [], []

        #print X.shape
        for value in target_feature_values:
            X[:, feature_id] = value
            predictions = self.model.predict(X)
            means.append( np.mean(predictions) ) 
            stds.append( np.std(predictions))

        
        return {'vals': target_feature_values, 'means':means, 'stds':stds}



        			