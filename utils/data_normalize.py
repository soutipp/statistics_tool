import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize(X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.copy())
    return X


def _convert_1d_series2int(x):
    """将一维数据按照标签进行编码为连续整数"""
    x = x.flatten()
    x_unique = np.unique(x)

    # if len(x_unique) > 100:
    #     raise RuntimeWarning(
    #         f"too many labels: {len(x_unique)} for the discrete data")

    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    return x


def _convert_arr2int(arr):
    """将一维数据按照标签进行编码为连续整数"""
    _, D = arr.shape
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    return arr.astype(int)


def stdize_values(x: np.ndarray, dtype: str, eps: float = 1e-10, discrete2integer: bool = True):
    """数据预处理: 标签值整数化、连续值归一化, 将连续和离散变量样本处理为对应的标准格式用于后续分析"""
    x = x.copy()
    x = x.reshape(x.shape[0], -1)
    if dtype == "continuous":
        # 连续值加入噪音并归一化
        x += eps * np.random.random_sample(x.shape)
        return normalize(x)
    elif dtype == "discrete":
        if discrete2integer:
            # 将标签值转为连续的整数值
            x = _convert_arr2int(x)
        return x
