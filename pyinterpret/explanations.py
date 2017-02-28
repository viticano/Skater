import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import datetime
import abc

print "__name__: {}".format(__name__ )
from .base import ModelInterpreter
from .local_interpretation.lime import Lime
from .global_interpretation.partial_dependence import PartialDependence


# Create based on class name:
def Interpretation(interpretation_type):
	'''
	Returns an interpretation class.

	Parameters:
	-----------
		interpretation_type(string): pdp, lime

	Returns:
	----------
		interpretation subclass
	'''
	
	if interpretation_type == "partial_dependence": 
		return PartialDependence()
	if interpretation_type == "lime": 
		return Lime()
	else:
		raise KeyError("interpretation_type needs to be element of {}".format(ModelInterpreter._types()))

