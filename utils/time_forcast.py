import pandas as pd
from swdp import SWDP
import os
import streamlit as st

from core.exception import EarlyExitException
from utils.string_utils import split_string


def time_forcast(name: str, field: str, client: SWDP, path: str):
    rows_count_path = client.execute_sql(query_sql=f'select count(*) from {name}', download_dir=path)
    rows = pd.read_csv(rows_count_path).loc[0, '_col0']
    os.remove(rows_count_path)
    if rows > 80000000:
        raise EarlyExitException()

    cols = len(split_string(field))

    if field == '*':
        cols_count_path = client.execute_sql(query_sql=f'select * from {name} limit 1', download_dir=path)
        cols = pd.read_csv(cols_count_path).shape[1]
        os.remove(cols_count_path)

    if cols * rows >= 54000000 * 12 * 1.5:
        raise EarlyExitException()

    loading_time = rows * cols / 25920000

    if loading_time > 1:
        st.text(f"经计算，数据加载时间约为 {loading_time} 分钟")

    if 15 <= loading_time <= 30:
        option = st.selectbox('是否继续运行？', ('--', '继续运行', '抽样并继续运行', '退出'))
        confirm = st.button('确认')
        if option == '继续运行' and confirm:
            return 'continue'
        elif option == '抽样并继续运行' and confirm:
            number = st.number_input('请输入要抽样的行数：')
            return number
        elif option == '退出' and confirm:
            raise EarlyExitException()






