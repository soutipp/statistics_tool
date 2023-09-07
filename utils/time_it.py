import time
import streamlit as st

def time_it(func):
    def wrapper(*args, **kwargs):

        start_time = time.time()

        res = func(*args, **kwargs)

        end_time = time.time()

        em = st.empty()
        em.caption(f'{func.__name__} 运行时间：{end_time - start_time:.2f}秒')

        return res

    return wrapper()
