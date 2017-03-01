from ..data.dataset import DataSet

class ModelInterpreter(object):
	'''
	Base Interpreter class. Common methods include loading a dataset and type setting.
	'''

	def __init__(self, interpreter):
		self.interpreter = interpreter

	@property
	def data_set(self):
		return self.interpreter.data_set

	@staticmethod
	def _types():
		return ['partial_dependence', 'lime']

	def consider(self, training_data, index=None, feature_names=None):
		self.interpreter.consider(training_data, index=index, feature_names=feature_names)
