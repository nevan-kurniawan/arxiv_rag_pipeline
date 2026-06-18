from evaluation.search_evaluation import hit_rate, mrr, ndcg_at_k #, map_at_k
import numpy as np
import pytest

matrix1 = np.array([[True, False, False, True, False]])
matrix2 = np.array(
    [
        [True, False, False, True, False],
        [False, False, False, False, False],
        [False, False, False, False, True],
        [True, True, True, True, True],
    ]
)
matrix3 = np.array([[False, False, False], [False, False, False]])


def test_hit_rate():
    assert hit_rate(matrix1) == 1.0
    assert hit_rate(matrix2) == 0.75
    assert hit_rate(matrix3) == 0.0


def test_mrr():
    assert mrr(matrix1) == 1.0
    assert mrr(matrix2) == 0.55
    assert mrr(matrix3) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(matrix1) == 1.0
    expected_ndcg_matrix2 = (1.0 + 0.0 + (1 / np.log2(6)) + 1.0) / 4.0
    assert ndcg_at_k(matrix2) == pytest.approx(expected_ndcg_matrix2)
    assert ndcg_at_k(matrix3) == 0.0


# def test_map_at_k():
#     assert map_at_k(matrix1) == 1.0
#     assert map_at_k(matrix2) == 0.55
#     assert map_at_k(matrix3) == 0.0
