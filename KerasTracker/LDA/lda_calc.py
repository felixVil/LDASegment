from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from Tracker_Params import Tracker_params


ALPHA_STD = Tracker_params['std_moving_average']


class LDANormalized(LinearDiscriminantAnalysis):

    def __init__(self, solver='svd', shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4):

        super(LDANormalized, self).__init__(solver=solver, shrinkage=shrinkage, priors=priors,
                                            n_components=n_components, store_covariance=store_covariance, tol=1e-4)
        self.prob_std = None

    def update_lda_params(self, cov_matrix, mean_vecs, priors):
        self.covariance_ = cov_matrix
        self.means_ = mean_vecs
        self.priors_ = priors
        self.coef_ = np.linalg.lstsq(self.covariance_, self.means_.T, rcond= None)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T)) +
                           np.log(self.priors_))
        self.coef_ = np.array(self.coef_[1, :] - self.coef_[0, :], ndmin=2)
        self.intercept_ = np.array(self.intercept_[1] - self.intercept_[0],
                                       ndmin=1)

    def predict_proba_limited(self, X):
        prob = self.decision_function(X)
        if self.prob_std is None:
            self.prob_std = np.std(prob)
        else:
            self.prob_std = ALPHA_STD * np.std(prob) + (1-ALPHA_STD) * self.prob_std
        prob = prob/self.prob_std # normalizing decision values by std.
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        prob = prob/np.amax(prob)
        return np.column_stack([1 - prob, prob])

