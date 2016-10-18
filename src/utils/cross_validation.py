from sklearn.cross_validation import StratifiedKFold
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
