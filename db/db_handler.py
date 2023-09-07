import json
import os
import streamlit as st
from swdp import SWDP

from core.exception import EarlyExitException, SqlError
from utils.cache_delete import delete_old_files
from utils.time_forcast import time_forcast


class DatabaseLoader:
    def __init__(self):
        self.name = None
        self.field = None

    @staticmethod
    @st.cache_resource
    def connector():
        """
        连接swdp
        :return: client
        """
        return SWDP('xuguangcai', 'db//sMWc19NcdGd3aaTAqSJx6/DGqwtUKKXR7nspOS4=')

    @staticmethod
    @st.cache_data(persist=True, max_entries=10000000)
    def load_data(name=None, field=None, sql=None) -> str:
        if os.name == 'nt':
            path = r'D:\stats_input'
        elif os.name == 'posix':
            path = r'./stats_input'

        if not os.path.exists(path):
            os.makedirs(path)

        # delete_old_files(path)

        if name is not None and name != '':
            all_file_path = os.path.join(path, f'{name}_all.csv')
            part_field_file_path = os.path.join(path, f'{name}_{field}.csv')

            if os.path.exists(all_file_path):
                return all_file_path

            if os.path.exists(part_field_file_path):
                return part_field_file_path

        client = DatabaseLoader.connector()

        if sql is None or sql == '':
            sql = f'select {field} from {name}'
            try:
                res = time_forcast(name=name, field=field, client=client, path=path)
            except EarlyExitException:
                raise EarlyExitException
            if res != 'continue' and res is not None:
                sql = f'select {field} from {name} order by rand() limit {res}'

        result_file_local_path_temp = client.execute_sql(query_sql=sql, download_dir=path)

        if not result_file_local_path_temp.endswith('.csv'):
            try:
                raise SqlError(json.loads(result_file_local_path_temp)["data"])
            except SqlError as e:
                st.text(str(e))
                raise SqlError()

        if name is None or name == '':
            print(rf"执行sql:({sql}) 本地存储路径:{result_file_local_path_temp}")
            return result_file_local_path_temp

        if field == '*':
            os.rename(rf'{result_file_local_path_temp}', all_file_path)
            print(rf"执行sql:({sql}) 本地存储路径:{all_file_path}")
            return all_file_path
        else:
            os.rename(rf'{result_file_local_path_temp}', part_field_file_path)
            print(rf"执行sql:({sql}) 本地存储路径:{part_field_file_path}")
            return part_field_file_path
