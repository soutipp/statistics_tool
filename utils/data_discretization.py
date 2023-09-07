import pandas as pd
import numpy as np


def _discretize_series(x: np.ndarray, n: int = 30, method="qcut"):
    """对数据序列采用等频分箱"""
    q = int(len(x) // n)
    if method == "qcut":
        x_enc = pd.qcut(x, q, labels=False, duplicates="drop").flatten()  # 等频分箱
    elif method == "cut":
        x_enc = pd.cut(x, q, labels=False, duplicates="drop").flatten()  # 等宽分箱
    return x_enc


def discretize_arr(X: np.ndarray, n: int = None, method: str = "qcut"):
    """逐列离散化"""
    if n is None:
        n = X.shape[0] // 20
    X = X.copy()
    for i in range(X.shape[1]):
        X[:, i] = _discretize_series(X[:, i], n, method)
    return X.astype(int)


def test():
    X = pd.read_csv(r'D:\git_repo\statistics\test\encoded_sample1_10000.csv').to_numpy()
    res = discretize_arr(X, n=83)
    print(X)
    print(res)
    print(res.shape)
    print(res.T[0].max())
    print(res.T[0].min())
    print(res.T[1].max())
    print(res.T[1].min())
