"""
This `mod` contains functions that would help process the data.
"""
from sklearn.preprocessing import LabelEncoder

import os, sys
import pandas as pd

basepath = os.path.expanduser('~/Desktop/src/AllState_Claims_Severity/')
sys.path.append(os.path.join(basepath, 'src'))

def load_data():
	train      = pd.read_csv(os.path.join(basepath, 'data/raw/train.csv'))
	test       = pd.read_csv(os.path.join(basepath, 'data/raw/test.csv'))
	sample_sub = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv'))

	return (train, test, sample_sub)

def get_multi_valued_features(train, test):
	columns  = train.select_dtypes(include=['object']).columns
	return [col for col in columns if train[col].nunique() > 2 or test[col].nunique() > 2]

def get_binary_valued_features(train, test):
	columns = train.select_dtypes(include=['object']).columns
	return [col for col in columns if train[col].nunique() == 2 or test[col].nunique() == 2]

def encode_categorical_features(train, test):
	columns = train.select_dtypes(include=['object']).columns

	for col in columns:
		lbl = LabelEncoder()
		lbl.fit(list(train[col]) + list(test[col]))

		train[col] = lbl.transform(train[col])
		test[col]  = lbl.transform(test[col])

	return train, test


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
