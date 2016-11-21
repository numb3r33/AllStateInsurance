import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def transform_target_variable(y):
	"""
	Performs a transformation on the target variable

	Arguments:
	---------

	y: Pandas Series representing the target variable

	Returns:
	--------

	Transformed target variable which is also a Pandas Series
	"""

	return y.map(np.log)


def retransform_target_variable(y):
	"""
	Brings back the transformed target variable to the original domain

	Arguments:
	----------

	y: Pandas Series representing the original transformed variable

	Returns:

	Pandas Series representing the target variable in the original domain
	"""

	return y.map(np.exp)

def bin_variable(y, n_labels=3):
	"""
	Bin the variable based on the range of the values it can take and
	the number of labels provided in the input.

	Arguments:
	----------
	y: Pandas Series representing a feature
	n_labels: Number of labels to be assigned the variable.

	Returns:
	--------

	labels: Pandas Series representing categorical variable generated
			after binning.
	"""

	bins   = np.linspace(y.min(), y.max(), n_labels)
	labels = pd.Series(np.digitize(y, bins))

	return labels

def freq_bin_variable(y, var_name, n_labels=2):
	"""
	Bin the variable by first transforming the original variable
	to encapsulate the frequency counts and then assigning the bins based on
	the frequency.

	Arguments:
	----------

	y: Pandas Series representing a feature.
	var_name: Feature Name, will be used to calculate frequency.
	n_labels: Number of labels to be assigned to the variable.

	Returns:
	--------

	labels: Pandas Series representing the categorical variable generated
			after binning.

	"""
	y_freq = y.reset_index().groupby([var_name])[var_name].transform(lambda x: len(x))

	bins   = np.linspace(y_freq.min(), y_freq.max(), n_labels)
	labels = pd.Series(np.digitize(y_freq, bins))

	return labels



def label_encoding(data, categorical_features):
	for feat in categorical_features:
		lbl = LabelEncoder()
		lbl.fit(data[feat])

		data[feat] = lbl.transform(data[feat])

	return data

def get_categorical_features(columns):
	return [col for col in columns if 'cat' in col]

def get_numerical_features(columns):
	return [col for col in columns if 'cont' in col]
