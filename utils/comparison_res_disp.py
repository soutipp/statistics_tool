import streamlit as st
import pandas as pd


def res_disp(improve_value, improve_ratio, pvalue, sig, effect, disp_by_df=False, df_index=None):
    global ht_res_df
    if not disp_by_df:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(label="提升值", value=improve_value)
        col2.metric(label="提升百分比", value=improve_ratio)
        col3.metric(label="P值", value=pvalue)
        col4.metric(label="结果显著性", value=sig)
        col5.metric(label="效应量", value=effect)

    else:
        if df_index is not None:
            ht_res_df = pd.DataFrame(
                {'差值': improve_value, '差值百分比': improve_ratio, 'P值': pvalue, '差异显著性': sig,
                 '效应量': effect}, index=df_index)

        return ht_res_df


def effect_hint(effect='--'):
    if effect != '--':
        st.caption("注：显著性用来说明是否有统计上的差异性，效应量用来说明差异的大小")

        effect_reference_value = pd.DataFrame(
            {'0.01': ['微小'], '0.2': ['小'], '0.5': ['中等'], '0.8': ['大'], '1.2': ['很大'], '2.0': ['巨大']},
            index=['差异大小'])

        st.dataframe(effect_reference_value)

    else:
        pass
