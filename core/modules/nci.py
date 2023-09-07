import datetime
from zipfile import ZipFile
import pandas as pd
import streamlit as st

from core.draw_picture.plot import bar_plot_matplotlib, dist_plot_plotly_2, group_box_plot_plotly
from core.stats.descriptive_statistics import double_descriptive_table
from core.stats.hypothesis_testing import hypothesis_testing
from utils.comparison_res_disp import effect_hint, res_disp


# 12345
class NciActiveAnalyzer:
    def __init__(self):
        self.uploaded_files = st.file_uploader('请依次上传.zip压缩文件：', type='zip', accept_multiple_files=True)
        # st.caption('注：1、支持上传商业库分子、合成分子、专利分子中的任意两者或者三者的zip压缩文件\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
        # 2、压缩文件夹命名必须是commercial、synthesis、patent')
        st.caption('注：1、支持上传一个或多个zip压缩文件，每个zip里面是一个或多个csv文件，每个csv代表一个分子\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
                   2、程序获取的是csv文件中”time“列到”Repl“列之间的数据\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
                   3、支持拖拽上传')

        col1, col2, col3, col4 = st.columns([0.09, 0.2, 0.17, 0.54], gap='small')
        self.button = col1.button('开始')
        self.descriptive_statistics = col2.checkbox('查看描述性统计')
        self.median = col3.checkbox('检验中位数')
        col4.caption('(默认检验均值)')

    def run(self):
        if self.button and (len(self.uploaded_files) >= 2 if self.uploaded_files else None):
            dct = self.dataProcess()  # 各个预处理后的分子集合（专利、商业库、合成分子）, 是字典, value是df, df的每一行代表一种分子样本

            dct_comparison_group = self.getMaxAbsDiffTop10(dct=dct)

            self.describeAndHypothesis(dct_comparison_group)

    @st.cache_data(ttl=datetime.timedelta(seconds=20))
    def dataProcess(_self):
        # 读取、剔除、求均值、拼接、填充缺失值
        dct = {}
        for file in _self.uploaded_files:
            sample_lst = []
            # 解压缩上传的压缩包
            with ZipFile(file, "r") as zip_ref:

                # 遍历压缩包内的csv文件
                for file_info in zip_ref.infolist():
                    # 检查文件名是否以 ".csv" 结尾
                    if file_info.filename.endswith(".csv"):
                        # 尝试不同的编码来读取 CSV 文件
                        encodings = ["utf-8", "gbk", "iso-8859-1"]
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(zip_ref.open(file_info.filename), encoding=encoding)
                                break
                            except UnicodeDecodeError:
                                continue
                        time_index = df.columns.get_loc('time')
                        repl_index = df.columns.get_loc('Repl')
                        result = df.iloc[:, time_index + 1:repl_index].filter(regex='^(?!HY_)')

                        mean = result.mean().reset_index().rename(columns={'index': 'residue', 0: 'com_counts'}).T
                        mean = mean.rename(columns=mean.iloc[0])

                        # 列是各个基团（受体供体 or something）。值只有一行，是平均值。一行代表一个分子。因为每个分子的原文件是一个时间序列数据，因此取平均值作为它的值
                        mean = mean[1:]

                        sample_lst.append(mean)

            dct[file.name.split('.')[0]] = pd.concat(sample_lst, ignore_index=True).fillna(0)

            # st.dataframe(dct[file.name.split('.')[0]])

        return dct

    @st.cache_data(ttl=datetime.timedelta(seconds=20))
    def getMaxAbsDiffTop10(_self, dct: dict) -> dict:
        dct_concentrated_trend_indicator = {}  # 集中趋势指标，即均值or中位数。一个zip文件对应一个key
        if _self.median:
            for key in dct:
                concentrated = pd.DataFrame.from_records({f'{key}_concentrated': dct[key].median()})
                dct_concentrated_trend_indicator[key] = concentrated
        else:
            for key in dct:
                concentrated = pd.DataFrame.from_records({f'{key}_concentrated': dct[key].mean()})
                dct_concentrated_trend_indicator[key] = concentrated

        # 创建一个空字典来存储对比组。例如：有三个表a,b,c, 那么一共就有三种两两对比, 即ab,ac,bc, 那么此字典的三个键就是a_b, a_c, b_c, 键的值是经过处理后合并起来的两个表
        dct_comparison_group = {}
        # 使用嵌套循环来构建两两对比组
        keys = list(dct_concentrated_trend_indicator.keys())  # 获取字典的所有键

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                # 找绝对差值排名前十的指标（不知道学名是什么，暂且叫指标），存入concat_diff_top10_index
                concat = pd.concat(
                    [dct_concentrated_trend_indicator[keys[i]], dct_concentrated_trend_indicator[keys[j]]],
                    axis=1).dropna()
                concat[f'{keys[i]}_{keys[j]}_diff_top10_wo_HY'] = abs(
                    concat[f'{keys[i]}_concentrated'] - concat[f'{keys[j]}_concentrated'])
                if concat.shape[0] >= 10:
                    concat_diff_top10_index = concat.sort_values(by=f'{keys[i]}_{keys[j]}_diff_top10_wo_HY',
                                                                 ascending=False).head(10).index.tolist()
                else:
                    concat_diff_top10_index = concat.sort_values(by=f'{keys[i]}_{keys[j]}_diff_top10_wo_HY',
                                                                 ascending=False).index.tolist()  # 有可能实际这时候不满10个

                # 分别提取A表B表concat_diff_top10_index中的列
                table_A = dct[keys[i]].loc[:, concat_diff_top10_index]
                table_A['group'] = keys[i]
                table_B = dct[keys[j]].loc[:, concat_diff_top10_index]
                table_B['group'] = keys[j]

                # 合并AB表，并宽表转窄表（窄表画箱线图用，因为plotly需要的输入是窄表）
                # table_AB = pd.concat([table_A, table_B], ignore_index=True)
                # melted_table_AB = pd.melt(table_AB, id_vars='group', var_name='item', value_name='value')  # 宽表变窄表

                table_A.drop('group', axis=1, inplace=True)
                table_B.drop('group', axis=1, inplace=True)

                # 把table_A, table_B, melted_table_AB放进一个list作为dct_comparison_group的value
                # table_A, table_B用来做描述性统计和假设检验，melted_table_AB用来画箱线图
                # lst = [table_A, table_B, melted_table_AB]

                lst = [table_A, table_B]
                dct_comparison_group[f'{keys[i]}_{keys[j]}'] = lst

        return dct_comparison_group

    @st.cache_data(ttl=datetime.timedelta(seconds=20))
    def describeAndHypothesis(_self, dct_comparison_group: dict):
        for key in dct_comparison_group:
            st.subheader(key)
            df_A = dct_comparison_group[key][0]
            df_B = dct_comparison_group[key][1]

            AB_compare = pd.concat([df_A.mean(), df_B.mean()], axis=1)
            AB_compare = AB_compare.rename(columns={0: key.split('_')[0], 1: key.split('_')[1]})
            AB_compare.columns.name = f'{key}_diff_top10_wo_HY'  # 为列名集合设置一个名称

            # melted_table_AB = dct_comparison_group[key][2]

            if _self.descriptive_statistics:
                tab1, tab2, tab3 = st.tabs([":bar_chart:箱线图", ":paperclip:假设检验", ":pencil:描述性统计"])
            else:
                tab1, tab2 = st.tabs([":bar_chart:箱线图", ":paperclip:假设检验"])

            with tab1:  # 画图区域
                # st.write(f'{key}_diff_top10_wo_HY')
                # fig = group_box_plot_plotly(melted_table_AB)
                # st.plotly_chart(fig, use_container_width=True)

                fig = bar_plot_matplotlib(AB_compare)  # bar plot 1111
                st.pyplot(fig)

            with tab2:  # 假设检验区域
                em = st.empty()
                effect_hint('效应值参考表')

            fields = df_A.columns.tolist()

            merge = pd.DataFrame()
            for field in fields:
                desc_2d_lst = double_descriptive_table(df_A, df_B, field)
                desc_df = pd.DataFrame(desc_2d_lst[1:], columns=desc_2d_lst[0]).set_index('item')

                if _self.median:
                    ht_res = hypothesis_testing(desc_df, df_A.loc[:, field], df_B.loc[:, field], '中位数',
                                                direction='没有预期', disp_by_df=True)
                else:
                    ht_res = hypothesis_testing(desc_df, df_A.loc[:, field], df_B.loc[:, field], '均值',
                                                direction='没有预期', disp_by_df=True)

                ht_res.update({'df_index': [field]})
                ht_df = res_disp(**ht_res)

                merge = pd.concat([merge, ht_df])
                em.dataframe(merge)

                if _self.descriptive_statistics:
                    with tab3:  # 描述性统计区域
                        st.write(field)
                        st.dataframe(desc_df.T)
                        dist_plot_plotly_2(df_A, df_B, field, bin_size=0.015)
