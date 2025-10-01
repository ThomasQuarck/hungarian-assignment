import numpy as np

def reduce(matrix):
    M = matrix.copy()
    M -= M.min(axis=1, keepdims=True)
    M -= M.min(axis=0, keepdims=True)
    return M

def maximal_coupling_with_min_cover(reduced_matrix):
    size = reduced_matrix.shape[0]
    zero_positions = np.argwhere(reduced_matrix == 0)

    X1 = set(zero_positions[:, 0])
    X2 = set(zero_positions[:, 1])
    adj = {i: [] for i in X1}
    for i, j in zero_positions:
        adj[i].append(j)

    coupling_1 = {}
    coupling_2 = {}

    def find_path(u, visit):
        for v in adj.get(u, []):
            if v not in visit:
                visit.add(v)
                if v not in coupling_2 or find_path(coupling_2[v], visit):
                    coupling_2[v] = u
                    coupling_1[u] = v
                    return True
        return False

    for u in X1:
        if u not in coupling_1:
            find_path(u, set())

    marked_zeros = np.array([[u, v] for u, v in coupling_1.items()])

    # König's algorithm for minimal vertex cover from maximal matching
    unmatched_rows = set(range(size)) - set(coupling_1.keys())
    visited_rows = set()
    visited_cols = set()

    def dfs_konig(u):
        visited_rows.add(u)
        for v in adj.get(u, []):
            if v not in visited_cols:
                visited_cols.add(v)
                if v in coupling_2:
                    next_u = coupling_2[v]
                    if next_u not in visited_rows:
                        dfs_konig(next_u)

    for r in unmatched_rows:
        dfs_konig(r)

    covered_rows = set(range(size)) - visited_rows
    covered_cols = visited_cols

    return marked_zeros, covered_rows, covered_cols

def hungarian_algorithm(matrix):
    size = len(matrix)
    reduced_matrix = reduce(matrix)
    marked_zeros, covered_rows, covered_cols = maximal_coupling_with_min_cover(reduced_matrix)

    while len(marked_zeros) < size:
        uncovered_rows = list(set(range(size)) - covered_rows)
        uncovered_cols = list(set(range(size)) - covered_cols)

        min_non_covered = np.min(reduced_matrix[np.ix_(uncovered_rows, uncovered_cols)])

        covered_rows_mask = np.zeros(size, dtype=bool)
        covered_rows_mask[list(covered_rows)] = True
        covered_cols_mask = np.zeros(size, dtype=bool)
        covered_cols_mask[list(covered_cols)] = True

        rows_mask_2d = covered_rows_mask[:, np.newaxis]
        cols_mask_2d = covered_cols_mask[np.newaxis, :]
        doubly_covered_mask = rows_mask_2d & cols_mask_2d
        uncovered_mask = (~rows_mask_2d) & (~cols_mask_2d)

        reduced_matrix = np.where(doubly_covered_mask, reduced_matrix + min_non_covered, reduced_matrix)
        reduced_matrix = np.where(uncovered_mask, reduced_matrix - min_non_covered, reduced_matrix)

        marked_zeros, covered_rows, covered_cols = maximal_coupling_with_min_cover(reduced_matrix)

    total_cost = sum(matrix[i, j] for i, j in marked_zeros)
    return marked_zeros, total_cost

def pad_matrix_to_square(matrix):
    n_rows, n_cols = matrix.shape
    if n_rows == n_cols:
        return matrix.copy(), n_rows, n_cols
    size = max(n_rows, n_cols)
    square_matrix = np.zeros((size, size))
    square_matrix[:n_rows, :n_cols] = matrix
    return square_matrix, n_rows, n_cols

def rectangular_hungarian(cost_matrix):
    C, n, m = pad_matrix_to_square(cost_matrix)
    marked_zeros, total_cost = hungarian_algorithm(C)
    if n <= m:
        return marked_zeros[marked_zeros[:, 0] < n], total_cost
    else:
        return marked_zeros[marked_zeros[:, 1] < m], total_cost
