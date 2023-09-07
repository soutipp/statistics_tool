import streamlit as st
import scipy.stats as ss
from scipy.stats import mannwhitneyu
import numpy as np


@st.cache_data
def effect_cal(A_mean, A_std, A_count, B_mean, B_std, B_count, sig):
    if sig == '显著':
        sp = ((A_count - 1) * A_std ** 2 + (B_count - 1) * B_std ** 2) / (A_count + B_count - 2)
        eff = round(abs((A_mean - B_mean) / sp), 4)
    else:
        eff = '--'

    return eff


@st.cache_data
def z_test(a_mean, a_std, a_count, b_mean, b_std, b_count, direction='没有预期'):
    z = (a_mean - b_mean) / np.sqrt(a_std ** 2 / a_count + b_std ** 2 / b_count)
    if direction == 'A表大':
        pvalue = 1 - ss.norm.cdf(z)
    elif direction == 'B表大':
        pvalue = ss.norm.cdf(z)
    else:
        pvalue = 2 * (1 - ss.norm.cdf(abs(z)))
    return np.round(pvalue, 4)


@st.cache_data
def Mann_Whiteney(a_value, b_value, direction='没有预期'):
    if direction == 'A表大':
        _, pvalue = mannwhitneyu(b_value, a_value, method="auto", use_continuity=True, alternative="less")
        # if p > 0.05:
        #     _, p = mannwhitneyu(b_value, a_value, method="auto", use_continuity=False, alternative="less")
    elif direction == 'B表大':
        _, pvalue = mannwhitneyu(a_value, b_value, method="auto", use_continuity=True, alternative="less")
        # if p > 0.05:
        #     _, p = mannwhitneyu(a_value, b_value, method="auto", use_continuity=False, alternative="less")
    else:
        _, pvalue = mannwhitneyu(a_value, b_value, method="auto", use_continuity=True, alternative="two-sided")
        # if p > 0.05:
        #     _, p = mannwhitneyu(a_value, b_value, method="auto", use_continuity=False, alternative="two-sided")
    return np.round(pvalue, 4)


@st.cache_data
def hypothesis_testing(stats, value_A, value_B, statistic, direction, disp_by_df=False):
    global improve, improve_ratio, p, significance, effect
    A_mean = stats.loc['mean', 'table_A']
    A_std = stats.loc['std', 'table_A']
    A_count = stats.loc['count', 'table_A']
    B_mean = stats.loc['mean', 'table_B']
    B_std = stats.loc['std', 'table_B']
    B_count = stats.loc['count', 'table_B']

    if statistic == '均值':
        improve = "%.2f" % (B_mean - A_mean)
        improve_ratio = "{:.2%}".format(((B_mean - A_mean) / A_mean))

        p = z_test(A_mean, A_std, A_count, B_mean, B_std, B_count, direction=direction) \
            if direction != '--' else z_test(A_mean, A_std, A_count, B_mean, B_std, B_count)
        significance = "显著" if p < 0.05 else "不显著"
        effect = effect_cal(A_mean, A_std, A_count, B_mean, B_std, B_count, significance)

    elif statistic == '中位数':
        improve = "%.2f" % (stats.loc['50%', 'table_B'] - stats.loc['50%', 'table_A'])
        improve_ratio = "{:.2%}".format(
            ((stats.loc['50%', 'table_B'] - stats.loc['50%', 'table_A']) / stats.loc['50%', 'table_A']))

        p = Mann_Whiteney(value_A, value_B, direction) \
            if direction != '--' else Mann_Whiteney(value_A, value_B)
        significance = "显著" if p < 0.05 else "不显著"
        effect = effect_cal(A_mean, A_std, A_count, B_mean, B_std, B_count, significance)

    return {'improve_value': improve, 'improve_ratio': improve_ratio, 'pvalue': p, 'sig': significance,
            'effect': effect, 'disp_by_df': disp_by_df}
