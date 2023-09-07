import streamlit as st

from core.exception import EarlyExitException, SqlError
from core.stats.descriptive_statistics import describe_one_table


class SingleTableAnalyzer:
    def __init__(self, mode='--'):
        self.table_name = None
        self.cache_name = None
        self.field = None
        self.button = None
        self.sql = None
        self.uploaded_file = None
        self.mode = mode

    def run(self):
        if self.mode == '简单查询(数仓已存在该表)':
            self.table_name = st.sidebar.text_input("请输入表名称：")
            self.field = st.sidebar.text_input("请输入字段名称：")
            self.button = st.sidebar.button('开始')

        elif self.mode == '复杂查询(数仓不存在该表or更灵活的查询)':
            self.sql = st.text_area('请输入SQL语句：', height=200, help='',
                                    placeholder='')
            self.cache_name = st.text_input("可为本次查询结果命名（选填）：",
                                            placeholder="注：已命名的查询结果可被缓存，下次可用名称直接读取缓存而不再运行SQL语句（存在缓存时SQL可为空）, 目前只在同一台机器有效")
            self.field = st.text_input("请输入查询的列名（必填）：",
                                       placeholder="若有别名请填别名；多个列用‘,’隔开；若全部字段填‘*’")
            self.button = st.button('开始')

        elif self.mode == '上传本地csv':
            self.uploaded_file = st.file_uploader("请选择csv文件上传", type=['csv'],
                                                  help='目前单机最大支持文件行数5500w')
            self.field = st.text_input("请输入要查看的列名（必填）：", placeholder="若多个列用‘,’隔开；若全部字段填‘*’")
            self.button = st.button('开始')

        if ((self.table_name and self.field)
                or (self.field and self.sql)
                or (self.field
                    and (self.uploaded_file is not None
                         and self.uploaded_file))):
            if self.button:
                try:
                    data_name = self.cache_name if self.table_name is None else self.table_name
                    describe_one_table(data_name, self.field, self.sql, self.uploaded_file)
                except EarlyExitException:
                    st.text('数据量过大，暂不支持！')
                except SqlError:
                    pass
