"""
This `mod` contains functions that would help process the data.
"""

import pandas as pd

def get_multi_valued_features(train, test):
	columns  = train.select_dtypes(include=['object']).columns
	return [col for col in columns if train[col].nunique() > 2 or test[col].nunique() > 2]


def one_hot_encode_features(train, test, features):
	"""
	One Hot Encoding of categorical features

	Arguments
	=========

	train    : Pandas Dataframe
	test     : Pandas Datafrme
	features : List of names of categorical columns
	"""

	data          = pd.get_dummies(pd.concat((train[features], test[features]), axis=0))
	ntrain        = train.shape[0] # number of rows

	train_encoded = data.iloc[:ntrain, :]
	test_encoded  = data.iloc[ntrain:, :]

	return train_encoded, test_encoded