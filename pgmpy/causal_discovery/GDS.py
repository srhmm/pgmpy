__authors__ = ["osman-mian", "srhmm"]  # etc


import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.structure_score import BaseStructureScore


class GDS(_BaseCausalDiscovery):
    """The GDS (Greedy DAG Search) algorithm for causal discovery / structure learning.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/ecoli70")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use GDS to learn the causal structure from data:

    >>> from pgmpy.causal_discovery.GDS import GDS
    >>> gds = GDS()
    >>> _ = gds.fit(df)
    >>> edges = sorted(gds.causal_graph_.edges())
    >>> len(edges) > 0
    True

    References
    ----------
    .. [1] Mian, O., Marx, A. and Vreeken, J. Discovering Fully Oriented Causal Networks.
           Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2021.
    """

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        return_type: str = "dag",
        show_progress: bool = True,
    ):
        self.return_type = return_type
        self.scoring_method = scoring_method
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """The fitting procedure for GDS.

        Parameters
        ----------
        X: pd.DataFrame
            The input dataset.
        """
        # Step 0: Initialize scoring method and data structures
        # score = get_scoring_method(scoring_method=self.scoring_method, data=X)

        dag_current = DAG()
        dag_current.add_nodes_from(list(X.columns))
        # nodes = list(dag_current.nodes)
        raise NotImplementedError()

        if self.return_type == "dag":
            self.causal_graph_ = dag_current
        elif self.return_type == "pdag":
            self.causal_graph_ = dag_current.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag, got {self.return_type}")

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_)

        return self
