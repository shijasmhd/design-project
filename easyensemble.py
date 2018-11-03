import numbers

import numpy as np

from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.utils.deprecation import deprecated

from .base import BaseEnsembleSampler
from ..under_sampling import RandomUnderSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring
from ..pipeline import Pipeline

MAX_INT = np.iinfo(np.int32).max


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
@deprecated('EasyEnsemble is deprecated in 0.4 and will be removed in 0.6. '
            'Use EasyEnsembleClassifier instead.')
class EasyEnsemble(BaseEnsembleSampler):
    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False,
                 n_subsets=10,
                 ratio=None):
        super(EasyEnsemble, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.replacement = replacement
        self.n_subsets = n_subsets

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)

        X_resampled = []
        y_resampled = []
        if self.return_indices:
            idx_under = []

        for _ in range(self.n_subsets):
            rus = RandomUnderSampler(
                sampling_strategy=self.sampling_strategy_,
                random_state=random_state.randint(MAX_INT),
                replacement=self.replacement)
            sel_x, sel_y = rus.fit_resample(X, y)
            X_resampled.append(sel_x)
            y_resampled.append(sel_y)
            if self.return_indices:
                idx_under.append(rus.sample_indices_)

        if self.return_indices:
            return (np.array(X_resampled), np.array(y_resampled),
                    np.array(idx_under))
        else:
            return np.array(X_resampled), np.array(y_resampled)


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class EasyEnsembleClassifier(BaggingClassifier):
    
    def __init__(self, n_estimators=10, base_estimator=None, warm_start=False,
                 sampling_strategy='auto', replacement=False, n_jobs=1,
                 random_state=None, verbose=0):
        super(EasyEnsembleClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            oob_score=False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    def _validate_estimator(self, default=AdaBoostClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = clone(default)

        self.base_estimator_ = Pipeline(
            [('sampler', RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                replacement=self.replacement)),
             ('classifier', base_estimator)])

    def fit(self, X, y):
        # RandomUnderSampler is not supporting sample_weight. We need to pass
        # None.
        return self._fit(X, y, self.max_samples, sample_weight=None)
