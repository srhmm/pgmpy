__authors__ = ["osman-mian", "srhmm"]

import itertools

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.structure_score import BaseStructureScore, get_scoring_method


class GDS(_BaseCausalDiscovery):
    """The GDS (Greedy DAG Search) algorithm for causal discovery / structure learning.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore instance, default=None
        The local score used to evaluate edge additions and prunings. Supported structure scores: k2, bdeu, bds, bic-d,
        aic-d, ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should be an instance of
        ``BaseStructureScore``. If ``None``, an appropriate default is selected automatically based on whether the
        data is continuous or discrete.

    return_type : str, default="dag"
        The type of structure to return. One of ``"dag"`` (a fully directed DAG) or ``"pdag"`` (the DAG converted to a
        PDAG instance). GDS always orients every edge, so the PDAG is just a different wrapper around the same
        directed structure.

    min_improvement : float, default=1e-6
        Threshold used symmetrically for edge addition and pruning. An edge ``source -> node`` is added if its local
        score improvement exceeds ``min_improvement``; an incoming edge to ``source`` is removed if doing so decreases
        the local score by at most ``min_improvement``.

    n_jobs : int, default=-1
        Number of jobs to use for parallel score computations. Parallelization is used in the initialization and
        backward phases to compute relative score improvements when adding and removing edges.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : :class:`~pgmpy.base.DAG` or :class:`~pgmpy.base.PDAG`
        The learned causal graph (a DAG if ``return_type="dag"``, a PDAG if ``return_type="pdag"``).

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    priority_queue_ : Dict
        Candidate directed edges to be added to the causal graph with priority values.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

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
        min_improvement: float = 1e-6,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        self.return_type = return_type
        self.scoring_method = scoring_method
        self.min_improvement = min_improvement
        self.show_progress = show_progress
        self.n_jobs = n_jobs

    def _fit(self, X: pd.DataFrame):
        """The fitting procedure for GDS.

        Parameters
        ----------
        X: pd.DataFrame
            The input dataset.
        """
        # Step 0: Initialize scoring method and data structures
        score = get_scoring_method(scoring_method=self.scoring_method, data=X)
        # the current learned graph
        dag_current = DAG()
        dag_current.add_nodes_from(list(X.columns))
        variables = list(dag_current.nodes)
        # ordered candidate pairs with a priority score
        self.priority_queue_ = {}
        # unordered candidate pairs that can still be considered
        remaining_pairs = {frozenset((x1, x2)) for x1, x2 in itertools.combinations(variables, 2)}

        def compare_local_score(parent, child, parent_parents=None, child_parents=None):
            """Computes the relative score improvements of the edge (parent,child) over (child,parent)"""
            parent_parents = [] if parent_parents is None else parent_parents
            child_parents = [] if child_parents is None else child_parents

            score_fw = score.local_score(child, tuple(child_parents + [parent])) - score.local_score(
                child, tuple(child_parents)
            )
            score_bw = score.local_score(parent, tuple(parent_parents + [child])) - score.local_score(
                parent, tuple(parent_parents)
            )
            score_improvement = score_fw - score_bw
            return parent, child, score_improvement

        # Step 1: Initialization phase, score each directed edge.
        # edges_initial = list(itertools.combinations_with_replacement(variables, 2))
        edges_initial = list(itertools.combinations(variables, 2))

        pbar_initial = tqdm(
            edges_initial,
            desc="Initial scoring of candidate edges",
            unit="edge(s)",
            disable=not (self.show_progress and config.SHOW_PROGRESS),
        )
        local_scores_initial = (
            [compare_local_score(x1, x2) for x1, x2 in pbar_initial]
            if self.n_jobs == 1
            else Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(compare_local_score)(x1, x2) for x1, x2 in pbar_initial
            )
        )

        for x1, x2, phi in local_scores_initial:
            if abs(phi) >= self.min_improvement:  # and  x1 != x2
                self.priority_queue_.update({(x1, x2): phi, (x2, x1): -phi})

        # Step 2: Greedy forward/backward search.
        converged = False

        while not converged:
            # Step 2.1: Forward phase, add directed edges by priority,
            #  updating other edge scores whenever necessary
            forward_converged = True
            if len(self.priority_queue_) == 0:
                converged = True
                continue

            pbar_forward = tqdm(
                desc="Forward phase",
                unit="edge(s)",
                disable=not (self.show_progress and config.SHOW_PROGRESS),
            )

            while len(self.priority_queue_) > 0:
                parent, child = max(self.priority_queue_, key=self.priority_queue_.get)
                curr_priority = self.priority_queue_[(parent, child)]

                # remove forward and backward directions from consideration
                del self.priority_queue_[(parent, child)]

                if parent != child and (child, parent) in self.priority_queue_:
                    del self.priority_queue_[(child, parent)]

                remaining_pairs.discard(frozenset((parent, child)))

                has_cycle = nx.has_path(dag_current, child, parent)
                no_improvement = curr_priority < self.min_improvement

                if has_cycle or no_improvement:
                    pbar_forward.update(1)
                    pbar_forward.set_postfix_str(f"edges={dag_current.number_of_edges()}")
                    continue

                # add edge and update priority scores
                dag_current.add_edge(parent, child)
                forward_converged = False  # whenever at least one edge was added, go into backward phase as well

                curr_parents = list(dag_current.predecessors(child))
                cand_parents = [
                    cand_parent
                    for cand_parent in variables
                    if cand_parent != child and frozenset((cand_parent, child)) in remaining_pairs
                ]
                local_scores_current = [
                    compare_local_score(cand_parent, child, list(dag_current.predecessors(cand_parent)), curr_parents)
                    for cand_parent in cand_parents
                ]

                for cand_parent, cand_child, cand_phi in local_scores_current:
                    self.priority_queue_.update(
                        {(cand_parent, cand_child): cand_phi, (cand_child, cand_parent): -cand_phi}
                    )

                pbar_forward.update(1)
                pbar_forward.set_postfix_str(f"edges={dag_current.number_of_edges()}")

            pbar_forward.close()
            if forward_converged:
                converged = True
                continue

            # Step 2.2: Backward phase, remove directed whenever necessary.
            backward_converged = True

            pbar_backward = tqdm(
                variables,
                desc="Backward phase",
                unit="node(s)",
                disable=not (self.show_progress and config.SHOW_PROGRESS),
            )
            for node in pbar_backward:
                edge_removed = True
                while edge_removed:
                    edge_removed = False
                    best_s = None
                    curr_parents = list(dag_current.predecessors(node))
                    if len(curr_parents) <= 1:
                        continue

                    s_full = score.local_score(node, tuple(curr_parents))
                    set_size = len(curr_parents) - 1
                    sets = itertools.combinations(curr_parents, set_size)

                    edges_backward = [(node, list(set_)) for set_ in sets]
                    if self.n_jobs == 1:
                        local_scores_current = [
                            (score.local_score(child, tuple(parent_set)), parent_set)
                            for child, parent_set in edges_backward
                        ]
                    else:
                        scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                            delayed(score.local_score)(child, tuple(parent_set)) for child, parent_set in edges_backward
                        )
                        local_scores_current = list(zip(scores, [parent_set for _, parent_set in edges_backward]))

                    for s_trunc, parent_set in local_scores_current:
                        if s_trunc >= s_full - self.min_improvement:
                            edge_removed = True
                            s_full = s_trunc
                            best_s = parent_set

                    if edge_removed and best_s is not None:
                        removed_parent = list(set(curr_parents) - set(best_s))[0]
                        dag_current.remove_edge(removed_parent, node)
                        backward_converged = False

                    pbar_backward.set_postfix_str(f"edges={dag_current.number_of_edges()}")

            if backward_converged:
                converged = True

        # Step 3: Store the learned causal graph and related attributes.
        if self.return_type == "dag":
            self.causal_graph_ = dag_current
        elif self.return_type == "pdag":
            self.causal_graph_ = dag_current.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag, got {self.return_type}")

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_)

        return self
