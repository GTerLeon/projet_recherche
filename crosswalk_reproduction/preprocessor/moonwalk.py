import dgl
import torch
import numpy as np
from crosswalk_reproduction.graphs.graph import get_uniform_weights

import logging
logger = logging.getLogger(__name__)

def estimate_node_colorfulness(g, p, node_idx, walk_length, walks_per_node, group_key, prob=None):
    """Estimate proximity of a node to those of another group according to equation (3) of https://arxiv.org/abs/2105.02725.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph.
        node_idx (int): Index of the node for which to compute colorfulness.
        walk_length (int): The number of hops per walk.
        walks_per_node (_type_): The number of walks to sample when estimating colorfulness.
        group_key (str): Key of group labels to be used stored in g.ndata. 
        prob (str, optional): Key of weights to be used for obtaining probabilities when traversing graphs stored in g.edata. 
            Each weight has to be non-negative. If None, all weights are initialized uniformly as in the original paper. 
            Defaults to None.

    Returns:
        float: The estimated colorfulness of this node.
    """
    # Obtain walks starting from source node
    start_nodes = (torch.ones(walks_per_node) * node_idx).type(torch.int64)
    walks, _ = dgl.sampling.random_walk(g, start_nodes, length=walk_length, prob=prob)

    # Obtain groups of nodes visited
    visited_nodes = walks.flatten()
    # Drop all entries from hops where the source node had no outgoing edge
    visited_nodes = visited_nodes[visited_nodes != -1]
    visited_groups = g.ndata[group_key][visited_nodes]

    # Compute colorfulness
    #Original colorfulness
    #colorfulness = torch.sum(visited_groups != g.ndata[group_key][node_idx]) / len(visited_nodes)

    #moonwalk 1er essai
    _, counts = torch.unique(visited_groups, return_counts=True)
    colorfulness = torch.sum(counts**2) / (counts**p).sum()**(1/p)

    # r = len(g.nodes())  # Nb nodes in graphs
    # rd = len(visited_nodes)  # Number of visited nodes
    # x = torch.sum(visited_groups != g.ndata[group_key][node_idx]) / len(visited_nodes)

    # Modified colorfulness calculation
    # colorfulness = 1 / (rd * r) * torch.sum(x**p)
    return colorfulness.item()

# def estimate_node_colorfulness(G,v,l,p=2):
#   if len(G[v]) == 0:
#     return 0
#   v_color = G.attr[v]
#   class_list = np.unique(list(G.attr.values()))

#   visited_vector={c:0 for c in class_list}
#   cur = v
#   for i in range(l):
#     cur = np.random.choice(G[cur])
#     if G.attr[cur] != v_color:
#       visited_vector[G.attr[cur]]+=1
#   vec = list(visited_vector.values())
#   clf = np.sum(vec)**2/(np.linalg.norm(vec, ord = p)+1e-7)
#   return clf / l

def get_moonwalk_weights(g, alpha, p, walk_length, walks_per_node, group_key, prior_weights_key=None):
    """Computes new weights for each edge according to MoonWalk strategy.

    Args:
        g (dgl.heterograph.DGLHeteroGraph): The graph without parallel edges.
        alpha (float): Parameter in (0,1) controlling the degree of inter-group and intra-group connections.
        p (float): Parameter to control the bias towards nodes at group boundaries.
        walk_length (int): Length of random walks for estimating colorfulness.
        walks_per_node (int): Number of random walks for estimating colorfulness.
        group_key (str): Key of group labels in g.ndata to be used by MoonWalk.
        prior_weights_key (str, optional): Key of prior weights in g.edata to be modified by MoonWalk.
            Each weight has to be non-negative, and each node needs at least one edge with a positive weight.
            If None, each node's outgoing weights are initialized uniformly with their sum being normalized to 1.
            Defaults to None.

    Returns:
        torch.Tensor: Tensor with one weight per edge, ordered according to id's of g.edges.
    """
    assert not g.is_multigraph, "MoonWalk reweighting does not support parallel edges."
    assert 0 < alpha < 1, f"alpha needs to be in (0,1). Received {alpha=}"
    assert 0 < p, f"p needs to be greater than 0. Received {p=}"
    assert prior_weights_key is None or (g.edata[prior_weights_key] > 0).all(), "If provided, prior weights must be larger than 0."

    # Initialize weights if not provided
    if prior_weights_key is not None:
        prior_weights = g.edata[prior_weights_key]
    else:
        prior_weights = get_uniform_weights(g)

    # Pre-compute colorfulness and normalization factors for each node
    colorfulnesses = torch.tensor([
        estimate_node_colorfulness(g, p, node, walk_length, walks_per_node, group_key, prob=None) for node in g.nodes()
    ])

    # Compute new weights
    new_weights = torch.empty_like(prior_weights)
    for source in g.nodes():
        source_group = g.ndata[group_key][source]
        all_neighbors = g.successors(source)
        neighbor_groups = g.ndata[group_key][all_neighbors]
        same_group_neighbors = all_neighbors[neighbor_groups == source_group]
        n_different_groups_in_neighborhood = len(neighbor_groups[neighbor_groups != source_group].unique())

        # Handle edges towards nodes of the same group
        for group in neighbor_groups.unique():
            group_neighbors = all_neighbors[neighbor_groups == group]
            z = sum([(colorfulnesses[nb.item()] ** p) * prior_weights[g.edge_ids(source, nb)]
                    for nb in group_neighbors])

            # Compute weights for all neighboring nodes of same group
            for nb in group_neighbors:
                prior_weight = prior_weights[g.edge_ids(source, nb)]

                # Use MoonWalk colorfulness in calculation
                n = prior_weight * (colorfulnesses[nb.item()] ** p) / z

                # Compute n with either equation 2a or 2b
                if z > 0:
                    n = prior_weight * (colorfulnesses[nb.item()] ** p) / z
                else:
                    total_prior_weights_towards_group = sum(
                        [prior_weights[g.edge_ids(source, nb)] for nb in group_neighbors]
                    )
                    n = prior_weight / total_prior_weights_towards_group

                # Compute weights towards neighbors of the same color with either equation 3a or 4a
                if group == source_group:
                    # Equation 3a, if neighbors exist neighbors of different color
                    if len(group_neighbors) < len(all_neighbors):
                        new_weights[g.edge_ids(source, nb)] = (1 - alpha) * n
                    # Equation 4a, otherwise
                    else:
                        new_weights[g.edge_ids(source, nb)] = n
                # Compute weights towards neighbors of different colors with either equation 3b or 4b
                else:
                    # Equation 3b, if v has at least one neighbor of the same color
                    if len(same_group_neighbors) > 0:
                        new_weights[g.edge_ids(source, nb)] = alpha * n / n_different_groups_in_neighborhood
                    # Equation 4b, otherwise
                    else:
                        new_weights[g.edge_ids(source, nb)] = n / n_different_groups_in_neighborhood

    return new_weights

