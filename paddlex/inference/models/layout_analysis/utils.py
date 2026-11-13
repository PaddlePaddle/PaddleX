# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import numpy as np

STATIC_SHAPE_MODEL_LIST = [
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocLayoutV4",
]


def _find_cycle(n, edges):
    """
    Detect the first cycle in a directed graph using DFS.

    Args:
        n (int): Number of nodes.
        edges (dict): Edge dict {(i, j): confidence, ...}.

    Returns:
        list|None: List of nodes forming the first found cycle, or None.
    """
    graph = defaultdict(list)
    for i, j in edges.keys():
        graph[i].append(j)
    visited = [False] * n
    rec_stack = [False] * n

    def dfs(vertex, path):
        visited[vertex] = True
        rec_stack[vertex] = True
        path.append(vertex)
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                ret = dfs(neighbor, path)
                if ret is not None:
                    return ret
            elif rec_stack[neighbor]:
                cycle_start = path.index(neighbor)
                return path[cycle_start:]
        path.pop()
        rec_stack[vertex] = False
        return None

    for node in range(n):
        if not visited[node]:
            cycle = dfs(node, [])
            if cycle:
                return cycle
    return None


def _remove_cycles(n, edges):
    """
    Greedily remove cycles from a directed graph to obtain a DAG.

    Args:
        n (int): Number of nodes.
        edges (dict): Edge dict {(i, j): confidence, ...}.

    Returns:
        list: List of remaining DAG edges as (i, j) tuples.
    """
    edges = dict(edges)
    while True:
        cycle = _find_cycle(n, edges)
        if cycle is None:
            break
        min_edge, min_conf = None, 1e8
        for i, j in zip(cycle, cycle[1:] + cycle[:1]):
            if (i, j) in edges and edges[(i, j)] < min_conf:
                min_edge = (i, j)
                min_conf = edges[(i, j)]
        if min_edge is not None:
            del edges[min_edge]
    return list(edges.keys())


def _topological_sort(n, edges, rel_matrix):
    """
    Kahn's algorithm that picks, among the nodes with no remaining predecessor,
    the one with the highest Borda count from the relative order matrix.

    Args:
        n (int): Number of nodes.
        edges (list[tuple]): DAG edges as (i, j) pairs.
        rel_matrix (np.ndarray): Relative order scores [n, n].

    Returns:
        list[int]: Topologically sorted node indices.
    """
    in_degree = [0] * n
    graph = defaultdict(list)
    for i, j in edges:
        graph[i].append(j)
        in_degree[j] += 1

    candidates = [i for i in range(n) if in_degree[i] == 0]
    result = []
    while candidates:
        if len(candidates) == 1:
            best = candidates[0]
        else:
            best, best_score = None, -1.0
            for c in candidates:
                score = sum(rel_matrix[c][o] for o in candidates if o != c)
                if score > best_score or (
                    score == best_score and (best is None or c < best)
                ):
                    best_score = score
                    best = c
        candidates.remove(best)
        result.append(best)
        for neighbor in graph[best]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                candidates.append(neighbor)
    return result


def _find_connected_components(n, edges):
    """
    Find connected components of an undirected view of a directed graph.

    Args:
        n (int): Number of nodes.
        edges (list[tuple]): DAG edges as (i, j) pairs.

    Returns:
        list[list[int]]: List of connected components.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        union(i, j)

    comp_map = defaultdict(list)
    for i in range(n):
        comp_map[find(i)].append(i)

    return list(comp_map.values())


def _get_order_single_image(roor_b, rel_b):
    """
    Decode reading order for a single image.

    Args:
        roor_b (np.ndarray): ROOR logits [N, N].
        rel_b (np.ndarray): Relative order logits [N, N].

    Returns:
        np.ndarray: Position indices [N]. pos[i] = rank of element i.
    """
    N = roor_b.shape[0]

    # Step 1: Threshold ROOR logits to get candidate edges
    candidates = roor_b > 0
    np.fill_diagonal(candidates, False)
    edges = {
        (int(i), int(j)): float(roor_b[i, j]) for i, j in zip(*np.nonzero(candidates))
    }

    # Step 2: Remove cycles greedily
    dag_edges = _remove_cycles(N, edges)

    # Step 3: Find connected components
    components = _find_connected_components(N, dag_edges)

    # Step 4: Compute node votes from relative scores
    # Numerically stable sigmoid: the raw logits reach +-1e4 (masked entries),
    # which overflows `exp(-x)` for the naive form.
    rel_scores = np.where(
        rel_b >= 0,
        1.0 / (1.0 + np.exp(-np.abs(rel_b))),
        np.exp(-np.abs(rel_b)) / (1.0 + np.exp(-np.abs(rel_b))),
    )
    np.fill_diagonal(rel_scores, 0.0)
    node_votes = np.sum(rel_scores, axis=0)  # [N]

    # Step 5: Topological sort within each component
    comp_sequences = {}
    for ci, comp in enumerate(components):
        if len(comp) == 1:
            comp_sequences[ci] = list(comp)
            continue
        comp_set = set(comp)
        sorted_comp = sorted(comp)
        old2new = {old: new for new, old in enumerate(sorted_comp)}
        new2old = {new: old for old, new in old2new.items()}
        mapped_edges = [
            (old2new[u], old2new[v])
            for u, v in dag_edges
            if u in comp_set and v in comp_set
        ]
        m = len(comp)
        local_rel = rel_scores[np.ix_(sorted_comp, sorted_comp)]
        local_order = _topological_sort(m, mapped_edges, local_rel)
        comp_sequences[ci] = [new2old[idx] for idx in local_order]

    # Step 6: Sort components by mean node_votes
    comp_avg = []
    for ci, comp in enumerate(components):
        avg_rank = np.mean([node_votes[node] for node in comp])
        comp_avg.append((avg_rank, ci))
    comp_avg.sort(key=lambda x: x[0])

    order_list = []
    for _, ci in comp_avg:
        order_list.extend(comp_sequences[ci])

    # Step 7: Convert to position-index array
    pos = np.zeros(N, dtype=np.int64)
    for rank, node in enumerate(order_list):
        pos[node] = rank

    return pos


def compute_reading_order(qi, rel_logits, roor_logits):
    """
    Compute the reading order of the boxes kept after score filtering.

    Args:
        qi (np.ndarray): Query index [K] of each box. A query detected under
            several labels appears more than once and gets a single rank.
        rel_logits (np.ndarray): Relative order logits [N, N] over all queries.
        roor_logits (np.ndarray): ROOR order logits [N, N] over all queries.

    Returns:
        np.ndarray: Reading order ranks [K], 0-based.
    """
    K = len(qi)
    if K <= 1:
        return np.zeros(K, dtype=np.int64)

    unique_qi, inv_map = np.unique(qi, return_inverse=True)
    sub_roor = roor_logits[unique_qi][:, unique_qi]
    sub_rel = rel_logits[unique_qi][:, unique_qi]
    sub_pos = _get_order_single_image(sub_roor, sub_rel)
    return sub_pos[inv_map]
