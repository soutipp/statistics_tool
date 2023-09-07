import io
import time
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.stats import skew, kurtosis, norm, kstest, skewtest, kurtosistest
# from sklearn.neighbors import KernelDensity, BallTree, KDTree
# from scipy.special import gamma, psi
import streamlit as st

from core.draw_picture.plot import dist_plot_seaborn, dist_plot_plotly, dist_plot_pyecharts, dist_plot_wo_kde_pyecharts, \
    dist_plot_wo_kde_plotly, dist_plot_wo_kde_pyecharts_2, dist_plot_plotly_2
from core.exception import EarlyExitException, SqlError
from db.db_handler import DatabaseLoader
from utils.data_normalize import stdize_values
from utils.string_utils import split_string


@st.cache_data
def single_descriptive_table(table, field):
    table_data = [['item', 'value'],
                  ['count', round(table[field].shape[0], 0)],
                  ['mean', round(table[field].mean(), 4)],
                  ['std', round(table[field].std(), 4)],
                  ['cv', round(table[field].std() / table[field].mean(), 4)],
                  ['mode', round(table[field].mode()[0], 4)],
                  ['min', round(table[field].min(), 4)],
                  ['0.01%', round(table[field].quantile(0.0001), 4)],
                  ['25%', round(table[field].quantile(0.25), 4)],
                  ['50%', round(table[field].quantile(0.5), 4)],
                  ['75%', round(table[field].quantile(0.75), 4)],
                  ['99.99%', round(table[field].quantile(0.9999), 4)],
                  ['max', round(table[field].max(), 4)],
                  ['skewness', round(skew(table[field]), 4)],
                  # ['skew_test', round(skewtest(table[field]).pvalue, 4)],
                  ['kurtosis', round(kurtosis(table[field]), 4)],
                  # ['kurt_test', round(kurtosistest(table[field]).pvalue, 4)],
                  # ['norm_test',
                  #  round(kstest(table[field], cdf="norm", args=(table[field].mean(), table[field].std())).pvalue, 4)],
                  ]

    return table_data


@st.cache_data
def double_descriptive_table(table_A, table_B, field):
    table_data = [
        ['item', 'table_A', 'table_B'],
        ['count', round(table_A.shape[0], 0), round(table_B.shape[0], 0)],
        ['mean', round(table_A[field].mean(), 4), round(table_B[field].mean(), 4)],
        ['std', round(table_A[field].std(), 4), round(table_B[field].std(), 4)],
        ['cv', round(table_A[field].std() / table_A[field].mean(), 4),
         round(table_B[field].std() / table_B[field].mean(), 4)],
        ['mode', round(table_A[field].mode()[0], 4), round(table_B[field].mode()[0], 4)],
        ['min', round(table_A[field].min(), 4), round(table_B[field].min(), 4)],
        ['0.01%', round(table_A[field].quantile(0.0001), 4), round(table_B[field].quantile(0.0001), 4)],
        ['25%', round(table_A[field].quantile(0.25), 4), round(table_B[field].quantile(0.25), 4)],
        ['50%', round(table_A[field].quantile(0.5), 4), round(table_B[field].quantile(0.5), 4)],
        ['75%', round(table_A[field].quantile(0.75), 4), round(table_B[field].quantile(0.75), 4)],
        ['99.99%', round(table_A[field].quantile(0.9999), 4), round(table_B[field].quantile(0.9999), 4)],
        ['max', round(table_A[field].max(), 4), round(table_B[field].max(), 4)],
        ['skewness', round(skew(table_A[field]), 4), round(skew(table_B[field]), 4)],
        # ['skew_test', round(skewtest(table_A[field]).pvalue, 4), round(skewtest(table_B[field]).pvalue, 4)],
        ['kurtosis', round(kurtosis(table_A[field]), 4), round(kurtosis(table_B[field]), 4)],
        # ['kurt_test', round(kurtosistest(table_A[field]).pvalue, 4), round(kurtosistest(table_B[field]).pvalue, 4)],
        # ['norm_test', round(kstest(table_A[field], cdf="norm", args=(table_A[field].mean(), table_A[field].std())).pvalue, 4),
        #  round(kstest(table_B[field], cdf="norm", args=(table_B[field].mean(), table_B[field].std())).pvalue, 4)]
    ]

    return table_data


def describe_one_table(name, field, sql, file=None):
    if file is None:
        loader = DatabaseLoader()
        try:
            local_path = loader.load_data(name=name, field=field, sql=sql)
        except EarlyExitException as e1:
            raise e1
        except SqlError as e2:
            raise e2
    else:
        bytes_data = file.getvalue().decode("utf-8")
        local_path = file

    if field == '*':
        cols = pd.read_csv(local_path, nrows=1).select_dtypes(include=[np.number]).columns
    else:
        cols = [x.strip() for x in field.split(',')]

    for i, col in enumerate(cols):
        df = pd.read_csv(local_path if file is None else io.StringIO(bytes_data), usecols=[col])

        descriptive_result = single_descriptive_table(df, col)
        des = pd.DataFrame(descriptive_result[1:],
                           columns=descriptive_result[0]).set_index('item')

        st.write(f"## {col}")

        st.dataframe(des.T)

        count = des.loc['count', 'value']
        length = abs(des.loc['max', 'value'] - des.loc['min', 'value'])

        start_time = time.time()

        if count <= 5 * 100 * 1000:  # 数据量小于500w用dist_plot_plotly渲染，否则用dist_plot_wo_kde_pyecharts
            try:
                if length <= 2:
                    dist_plot_plotly(df, col, bin_size=0.1)
                else:
                    dist_plot_plotly(df, col)
            except LinAlgError as e:
                st.caption(f'{e}, {col}列的数值异常，导致协方差矩阵奇异，请检查数据!')
        else:
            dist_plot_wo_kde_pyecharts(data=df, field=col)

        # dist_plot_wo_kde_plotly(df, col)
        # dist_plot_pyecharts(df, col)
        # dist_plot_seaborn(df, col)

        elapsed_time = time.time() - start_time
        status_text = st.empty()
        status_text.text(f"绘图用时 {elapsed_time:.2f} 秒")


def describe_two_table(name_A, name_B, field, sql_A, sql_B, files=None):
    if files is None:
        loader = DatabaseLoader()
        try:
            local_path_A = loader.load_data(name=name_A, field=field, sql=sql_A)
            local_path_B = loader.load_data(name=name_B, field=field, sql=sql_B)
        except EarlyExitException:
            raise EarlyExitException

        table_A = pd.read_csv(local_path_A, usecols=[field])
        table_B = pd.read_csv(local_path_B, usecols=[field])
    else:
        table_A = pd.read_csv(files[0])
        table_B = pd.read_csv(files[1])

    table_data = double_descriptive_table(table_A, table_B, field)

    stats = pd.DataFrame(table_data[1:], columns=table_data[0]).set_index('item')

    st.header("描述性统计")
    st.dataframe(stats.T)

    count_A = stats.loc['count', 'table_A']
    count_B = stats.loc['count', 'table_B']
    length_A = abs(stats.loc['max', 'table_A'] - stats.loc['min', 'table_A'])
    length_B = abs(stats.loc['max', 'table_B'] - stats.loc['min', 'table_B'])

    start_time = time.time()

    if max(count_A, count_B) <= 5000000:
        if min(length_A, length_B) <= 2:
            dist_plot_plotly_2(table_A, table_B, field, bin_size=0.1)
        else:
            dist_plot_plotly_2(table_A, table_B, field)
    else:
        dist_plot_wo_kde_pyecharts_2(table_A, table_B, field)

    elapsed_time = time.time() - start_time
    status_text = st.empty()
    status_text.text(f"绘图用时 {elapsed_time:.2f} 秒")

    return stats, table_A, table_B


def calculate_entropy_discrete(x):
    _, counts = np.unique(x, return_counts=True, axis=0)
    proba = counts / len(x)
    proba = proba[proba > 0.0]
    entropy_discrete = np.sum(proba * np.log(1. / proba))
    return np.round(entropy_discrete, 2)


# 用k近邻计算连续数据的熵, 但暂时有问题(样本数据算出的熵是负的), 所以暂时注释掉
# def calculate_entropy_continuous(x, k, metric="chebyshev"):
#     assert k <= len(x) - 1
#     N, D = x.shape
#
#     # 构建距离树
#     tree = BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(x, metric=metric)
#
#     # 计算结果
#     nn_distc = tree.query(x, k=k + 1)[0][:, -1]  # 获得了各样本第k近邻的距离
#
#     if metric == "euclidean":
#         v = (np.pi ** (D / 2)) / gamma(1 + D / 2)
#     elif metric == "chebyshev":
#         v = 1
#     else:
#         raise ValueError(f"unsupported metric {metric}")
#
#     entropy_continuous = -psi(k) + psi(N) + np.log(v) + D * np.log(nn_distc).mean()
#
#     return entropy_continuous


class MargEntropy:
    """计算任意连续和离散变量的信息熵"""

    def __init__(self, x: np.ndarray, xtype: str):
        assert xtype in ['discrete', 'continuous']
        self.x_norm = stdize_values(x, xtype, discrete2integer=False)
        self.xtype = xtype

    def __call__(self, k: int = 3, metric: str = "euclidean"):
        if self.xtype == "discrete":
            return calculate_entropy_discrete(self.x_norm) / np.log(np.e)
        # elif self.xtype == "continuous":
        #     return calculate_entropy_continuous(self.x_norm, k, metric) / np.log(np.e)
