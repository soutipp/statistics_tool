import json

import numpy as np
import pandas as pd
import streamlit as st

from core.draw_picture.plot import density_heatmap_integer_lattice_point_matplotlib, \
    density_heatmap_continuous_lattice_point_matplotlib
from core.stats.descriptive_statistics import MargEntropy
from db.db_handler import DatabaseLoader
from utils.data_discretization import discretize_arr


class ChemicalSpaceAnalyzer:
    def __init__(self, mode: str = '--') -> None:
        self.mode = mode
        self.table_name = None
        self.table_name_A = None
        self.table_name_B = None
        self.uploaded_files = None
        self.button = None

        if mode != '--':
            self.one_or_two = st.sidebar.radio("请选择：",
                                               ["单表", "双表"],
                                               captions=["单独查看某个化学空间", "两个化学空间的对比"])

            if self.mode == '简单查询(数仓已存在该表)':
                if self.one_or_two == '单表':
                    self.table_name = st.sidebar.text_input("请输入表名称：")
                else:
                    self.table_name_A = st.sidebar.text_input("请输入A表名称：")
                    self.table_name_B = st.sidebar.text_input("请输入B表名称：")

                self.bins = st.sidebar.slider('可选择渲染分辨率', 120, 2000, 1200)
                self.button = st.sidebar.button('开始')

            elif self.mode == '复杂查询(数仓不存在该表or更灵活的查询)':
                st.write('功能尚未开发，暂不支持，敬请期待！')

            elif self.mode == '上传本地csv':
                self.uploaded_files = st.file_uploader('请上传结果数据：', type=['json', 'csv'],
                                                       accept_multiple_files=True)
                self.bins = st.sidebar.slider('可选择渲染分辨率', 120, 2000, 1200)
                self.button = st.button('开始')

    def run(self):
        bins_num = 800 * 10000 // 300  # 写死了，如有需要，后面可供用户配置.bins_num是用来计算熵时用来离散化的。这里的含义是，800w数据，每个箱子300个数据

        if (self.table_name or (self.table_name_A and self.table_name_B)) and self.button:  # 输入是数仓的表，走这个
            loader = DatabaseLoader()
            if self.one_or_two == '单表':
                local_path = loader.load_data(name=self.table_name, field='*')
                table = pd.read_csv(local_path).dropna()

                coordinates = table[['x', 'y']].to_numpy()

                st.write(self.table_name)
                fig = density_heatmap_continuous_lattice_point_matplotlib(x=coordinates.T[0], y=coordinates.T[1],
                                                                          bins=self.bins)
                st.pyplot(fig)

                if coordinates.shape[0] >= bins_num:
                    coordinates = discretize_arr(coordinates, n=coordinates.shape[0] // bins_num)  # 离散化为bins_num个区间
                marg_entropy = MargEntropy(coordinates, 'discrete')
                entropy = marg_entropy()
                st.metric(label='熵', value=entropy)

            else:  # 双表
                table_names = [self.table_name_A, self.table_name_B]
                local_path_A = loader.load_data(name=self.table_name_A, field='*')
                local_path_B = loader.load_data(name=self.table_name_B, field='*')
                table_A = pd.read_csv(local_path_A).dropna()
                table_B = pd.read_csv(local_path_B).dropna()
                tables = [table_A, table_B]

                cols = st.columns(2)
                index = 0
                for table in tables:
                    coordinates = table[['x', 'y']].to_numpy()
                    fig = density_heatmap_continuous_lattice_point_matplotlib(x=coordinates.T[0], y=coordinates.T[1],
                                                                              bins=self.bins)

                    if coordinates.shape[0] >= bins_num:
                        coordinates = discretize_arr(coordinates, n=coordinates.shape[0] // bins_num)  # 离散化为bins_num个区间
                    marg_entropy = MargEntropy(coordinates, 'discrete')

                    with cols[index]:
                        st.write(table_names[index])
                        st.pyplot(fig)
                        entropy = marg_entropy()
                        st.metric(label='熵', value=entropy)

                    index += 1

        if (len(self.uploaded_files) != 0 if self.uploaded_files else False) and self.button:  # 输入是本地文件，走这个
            cols = st.columns(len(self.uploaded_files))
            index = 0
            for uploaded_file in self.uploaded_files:
                is_json = True
                if uploaded_file.name.endswith('.json'):  # 输入是json文件, key是坐标, value是该坐标下的点的个数
                    file_contents = uploaded_file.read()
                    parsed_data = json.loads(file_contents)

                    # 提取坐标和个数统计
                    keys = [eval(coord) for coord in parsed_data.keys()]
                    coordinates = np.array([[coord[0] for coord in keys], [coord[1] for coord in keys]]).T
                    counts = list(parsed_data.values())

                elif uploaded_file.name.endswith('.csv'):  # 输入是csv文件, 两列坐标x、y
                    is_json = False

                    coordinates = pd.read_csv(uploaded_file).to_numpy()
                    # _, counts = np.unique(coordinates, return_counts=True, axis=0)

                if is_json:
                    fig = density_heatmap_integer_lattice_point_matplotlib(coordinates=coordinates, values=counts)
                    marg_entropy = MargEntropy(coordinates, 'discrete')
                else:
                    fig = density_heatmap_continuous_lattice_point_matplotlib(x=coordinates.T[0], y=coordinates.T[1],
                                                                              bins=self.bins)
                    # 离散化
                    if coordinates.shape[0] >= bins_num:
                        coordinates = discretize_arr(coordinates, n=coordinates.shape[0] // bins_num)  # 离散化为bins_num个区间
                    marg_entropy = MargEntropy(coordinates, 'discrete')

                with cols[index]:
                    st.write(uploaded_file.name.split('.')[0])
                    st.pyplot(fig)
                    entropy = marg_entropy()
                    st.metric(label='熵', value=entropy)

                index += 1
