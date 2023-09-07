import streamlit as st

from core.exception import EarlyExitException
from core.stats.descriptive_statistics import describe_two_table
from core.stats.hypothesis_testing import hypothesis_testing
from utils.comparison_res_disp import res_disp, effect_hint


class DoubleTableAnalyzer:
    def __init__(self, mode='--'):
        self.table_name_A = None
        self.table_name_B = None
        self.cache_name_A = None
        self.cache_name_B = None
        self.field = None
        self.statistic = None
        self.direction = None
        self.sql_A = None
        self.sql_B = None
        self.files = None
        self.mode = mode

    def run(self):
        global stats, value_A, value_B
        if self.mode == '简单查询(数仓已存在该表)':
            self.table_name_A = st.sidebar.text_input("请输入A表名称：")
            self.table_name_B = st.sidebar.text_input("请输入B表名称：")
            self.field = st.sidebar.text_input("请输入字段名称：")

        elif self.mode == '复杂查询(数仓不存在该表or更灵活的查询)':
            st.sidebar.caption('请先在右侧主页面填入查询信息！')

            self.sql_A = st.text_area('请输入SQL语句：', height=200, help='注：大于5500w的数据量尚未测试本工具可用性！',
                                      placeholder='A表查询SQL')
            self.cache_name_A = st.text_input("可为A表查询命名（选填）：",
                                              placeholder="注：已命名的查询结果可被缓存，下次可用名称直接读取缓存而不再运行SQL语句（存在缓存时SQL可为空）, 目前只在同一台机器有效")
            self.sql_B = st.text_area('请输入SQL语句：', height=200, help='注：大于5500w的数据量尚未测试本工具可用性！',
                                      placeholder='B表查询SQL')
            self.cache_name_B = st.text_input("可为B表查询命名（选填）：",
                                              placeholder="注：已命名的查询结果可被缓存，下次可用名称直接读取缓存而不再运行SQL语句（存在缓存时SQL可为空）, 目前只在同一台机器有效")
            self.field = st.text_input("请输入查询的字段名（必填!）：",
                                       placeholder="只能填入一个字段，且A、B的字段须一致，如不一致请在SQL中起别名令其一致！")
            st.caption('请在侧边栏填入更多信息并点击开始！')

        elif self.mode == '上传本地csv':
            self.files = st.file_uploader("请依次上传两个csv文件", type=['csv'], accept_multiple_files=True,
                                          help='注：目前仅支持大小为2G的csv文件！')
            self.field = st.text_input("请输入要查看的列名（必填）：",
                                       placeholder="只能填入一个字段，且A、B表的字段名称须一致")
            st.caption('请在侧边栏填入更多信息并点击开始！')

        self.statistic = st.sidebar.selectbox(
            '请选择要检验的统计量',
            ('--', '均值', '中位数'),
            index=0
        )
        st.sidebar.caption("注：如果数据分布近似正态，对均值或中位数进行假设检验均可；反之建议对中位数进行假设检验！")

        self.direction = st.sidebar.selectbox(
            '针对您选择的字段，您预期A表的值大，还是B表的值大？',
            ('--', 'A表大', 'B表大', '没有预期'),
            index=0
        )

        button = st.sidebar.button('开始')

        if (self.table_name_A and self.table_name_B and self.field) or (
                self.sql_A and self.sql_B and self.field) or (
                len(self.files) == 2 if self.files else None and self.field):
            if button:
                try:
                    data_name_A = self.cache_name_A if self.table_name_A is None else self.table_name_A
                    data_name_B = self.cache_name_B if self.table_name_B is None else self.table_name_B
                    stats, value_A, value_B = describe_two_table(data_name_A, data_name_B, self.field,
                                                                 self.sql_A, self.sql_B, self.files)
                except EarlyExitException:
                    st.text('数据量过大，暂不支持！')

                st.header("假设检验")

                ht_res = hypothesis_testing(stats, value_A, value_B, self.statistic, self.direction)

                res_disp(**ht_res)
                effect_hint(ht_res['effect'])
