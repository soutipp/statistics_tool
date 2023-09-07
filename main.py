import streamlit as st

from core.modules.chemical_space import ChemicalSpaceAnalyzer
from core.modules.nci import NciActiveAnalyzer
from core.modules.single import SingleTableAnalyzer
from core.modules.double import DoubleTableAnalyzer

# main.py是程序的运行入口

st.set_page_config(
    page_title="统计分析工具",  # 页面标题
    page_icon=":computer:",  # icon
    layout="wide",  # 页面布局
    initial_sidebar_state="auto"  # 侧边栏
)

if __name__ == '__main__':
    option = st.sidebar.selectbox(
        '请选择功能模块',
        ('--', '单表描述性统计', '双表假设检验', 'NCI统计分析', '化学空间分析'))

    st.title("统计分析工具")

    if option == 'NCI统计分析':  # 这个业务方只需要本地文件上传这一种输入模式
        NciActiveAnalyzer().run()

    elif option != '--':
        mode = st.sidebar.selectbox(
            '请选择输入方式',
            ('--', '简单查询(数仓已存在该表)', '复杂查询(数仓不存在该表or更灵活的查询)', '上传本地csv'))

        if option == '单表描述性统计' and mode != '--':
            SingleTableAnalyzer(mode=mode).run()

        elif option == '双表假设检验' and mode != '--':
            DoubleTableAnalyzer(mode=mode).run()

        elif option == '化学空间分析':
            ChemicalSpaceAnalyzer(mode=mode).run()

