import numpy as np
import xgboost as xgb

from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error

def get_stratified_sample(loss_indicator):
	skf = StratifiedKFold(loss_indicator, n_folds=2, shuffle=True, random_state=111)
	itrain, itest = next(iter(skf))

	return itrain, itest


def get_cv_scores(loss_indicator, X_train, y_train, pipeline, n_folds=3):

	skf = StratifiedKFold(loss_indicator, n_folds=n_folds, shuffle=True, random_state=112)
	scores = []

	for itr, ite in skf:
		X_tr = X_train.iloc[itr]
		X_te = X_train.iloc[ite]

		y_tr = y_train.iloc[itr]
		y_te = y_train.iloc[ite]

		pipeline.fit(X_tr, y_tr)

		y_pred1  = (pipeline.predict(X_te))
		y_pred = 1.0 * y_pred1

		scores.append(mean_absolute_error(y_te, y_pred))

	return scores

def mae(y, y0):

	y0=y0.get_label()
	return 'error',mean_absolute_error(np.exp(y), np.exp(y0))


def cv_xgboost(train, target):
	kf = KFold(len(train), n_folds=3, shuffle=True, random_state=12313)
	scores = []

	for i, (itr, ite) in enumerate(kf):
		print('Fold: {}'.format(i))

		Xtr = train.iloc[itr]
		Xte = train.iloc[ite]

		ytr = target.iloc[itr]
		yte = target.iloc[ite]

		# number of trees
		n_rounds = 300

		# set up configurations
		params = {}

		params['max_depth']        = 6
		params['objective']        = 'reg:linear'
		params['eta']              = 20 / n_rounds
		params['nthread']          = 4
		params['gamma']            = 1
		params['min_child_weight'] = 2
		params['subsample']        = 1.0
		params['colsample_bytree'] = 0.8


		ytr_transformed = np.log(ytr)
		yte_transformed = np.log(yte)

		Dtrain = xgb.DMatrix(Xtr, ytr_transformed)
		Dval   = xgb.DMatrix(Xte, yte_transformed)
		plst   = list(params.items())


		# define a watch list to observe the change in error for training and holdout data
		watchlist  = [ (Dtrain, 'train'),(Dval, 'eval')]

		model = xgb.train(plst,
						  Dtrain,
						  n_rounds,
						  watchlist,
						  feval=mae,  # custom evaluation function
						  early_stopping_rounds=50) # stops 50 iterations after marginal improvements or drop in performance on your hold out set

		print('best ite:',model.best_iteration)
		print('best score:',model.best_score)

		yhat = model.predict(Dval)
		yhat = np.exp(yhat)

		score = mean_absolute_error(np.exp(yte_transformed), yhat)
		print('MAE: {}'.format(score))

		scores.append(score)

	return scores
