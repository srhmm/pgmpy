import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery.GDS import GDS
from pgmpy.structure_score import BaseStructureScore


@pytest.fixture
def nonlinear_data():
    rng = np.random.default_rng(0)
    n = 2000
    A = rng.uniform(-2.0, 2.0, n)
    B = np.sin(2.0 * A) + rng.normal(0.0, 0.25, n)
    C = 0.5 * A**2 + rng.normal(0.0, 0.25, n)
    D = B * C + rng.normal(0.0, 0.25, n)

    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D})


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [GDS(return_type="dag")],
    expected_failed_checks=expected_failed_checks,
)
def test_gds_compatibility(estimator, check):
    check(estimator)


class TestScore(BaseStructureScore):
    def __init__(self, data, degree=3):
        super().__init__(data)
        self.data = data
        self.degree = degree

    def local_score(self, variable, parents):
        y = self.data[variable].to_numpy()
        n = len(y)

        if len(parents) == 0:
            residuals = y - y.mean()
            rss = np.sum(residuals**2)
            k = 1
        else:
            X = self.data[list(parents)].to_numpy()
            X_poly = PolynomialFeatures(
                degree=self.degree,
                include_bias=False,
            ).fit_transform(X)

            reg = LinearRegression().fit(X_poly, y)
            residuals = y - reg.predict(X_poly)
            rss = np.sum(residuals**2)
            k = X_poly.shape[1] + 1

        rss = max(rss, 1e-12)

        return -0.5 * n * math.log(rss / n) - 0.5 * k * math.log(n)


def test_gds_recovers_nonlinear(nonlinear_data):
    score = TestScore(nonlinear_data, degree=3)
    est = GDS(scoring_method=score).fit(nonlinear_data)

    # Attributes set by fit.
    assert est.n_features_in_ == nonlinear_data.shape[1]
    assert list(est.feature_names_in_) == list(nonlinear_data.columns)
    assert est.causal_graph_ is not None
    assert est.adjacency_matrix_.shape == (4, 4)

    # All true edges should be recovered.
    learned = set(est.causal_graph_.edges())
    true_edges = {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    assert true_edges.issubset(learned)
