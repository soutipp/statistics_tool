import numpy as np
import pandas as pd

import streamlit as st
import streamlit_echarts as st_echarts

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

from pyecharts.charts import Bar, Line
from pyecharts import options as opts
from scipy.stats import gaussian_kde, norm


# 12345
@st.cache_data(experimental_allow_widgets=True)
def dist_plot_plotly(data, field, bin_size=0.5):
    lst = [data[field].values.tolist()]

    group_labels = [field]

    fig = ff.create_distplot(lst, group_labels, histnorm='probability density', curve_type='kde', bin_size=bin_size,
                             show_rug=False)
    fig2 = ff.create_distplot(lst, group_labels, histnorm='probability density', curve_type='normal', bin_size=bin_size)

    fig.update_layout(title_text=field)

    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']

    fig.add_traces(go.Scatter(x=normal_x, y=normal_y, mode='lines',
                              line=dict(color='red',
                                        dash='dash',
                                        width=1),
                              name='normal'
                              ))

    # fig.update_layout(xaxis_range=[data[field].quantile(0.01), data[field].quantile(0.99)])

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(experimental_allow_widgets=True)
def dist_plot_plotly_2(data_A, data_B, field, bin_size=0.5):
    data_A = data_A[field].values.tolist()
    data_B = data_B[field].values.tolist()
    data = [data_A, data_B]

    group_labels = [f'{field}_A', f'{field}_B']

    fig = ff.create_distplot(data, group_labels, curve_type='kde', bin_size=bin_size, show_rug=False)
    fig2 = ff.create_distplot(data, group_labels, curve_type='normal', bin_size=bin_size, show_rug=False)

    fig.update_layout(title_text=f'{field}')

    normal_x_A = fig2.data[2]['x']
    normal_y_A = fig2.data[2]['y']

    normal_x_B = fig2.data[3]['x']
    normal_y_B = fig2.data[3]['y']

    fig.add_traces(go.Scatter(x=normal_x_A, y=normal_y_A, mode='lines',
                              line=dict(color='red',
                                        dash='dash',
                                        width=2),
                              name='normal_A'
                              ))

    fig.add_traces(go.Scatter(x=normal_x_B, y=normal_y_B, mode='lines',
                              line=dict(color='green',
                                        dash='dash',
                                        width=2),
                              name='normal_B'
                              ))

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(experimental_allow_widgets=True)
def dist_plot_plotly_n_one_df(n=3, ):
    pass


@st.cache_data
def dist_plot_wo_kde_plotly(data, field):
    list = [data[field].values.tolist()]

    group_labels = [field]

    fig = ff.create_distplot(list, group_labels, curve_type='normal', show_rug=False)

    fig.update_layout(title_text=field)

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def dist_plot_seaborn(data, field):
    # 设置matplotlib的backend为Agg
    # 这会将图形渲染到内存中，而不是直接渲染到浏览器中
    # 从而避免WebSocket消息大小超出限制的问题
    import matplotlib
    matplotlib.use('Agg')

    # quantile_9999 = data[field].quantile(0.9999)
    # data = data[data[field] <= quantile_9999]

    fig, ax = plt.subplots()
    # fig = sns.histplot(data=data, kde=True, stat="density", kde_kws=dict(cut=3, kernel="epanechnikov"), alpha=0.4)
    sns.histplot(data=data[field], stat="density", alpha=0.6)
    sns.kdeplot(data=data[field], cut=3, alpha=1, bw_method='silverman')
    # fig.set_xlim(data[field].min(), quantile_9999)
    # ax.set_xlim([data[field].min(), quantile_9999])

    mu, std = norm.fit(data[field])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=1, linestyle='--')

    st.pyplot(fig)


@st.cache_data(experimental_allow_widgets=True)
def dist_plot_wo_kde_pyecharts(data, field):
    # 计算频率分布
    hist, bin_edges = np.histogram(data[field], bins='auto', density=True)
    x = [round((bin_edges[i] + bin_edges[i + 1]) / 2, 2) for i in range(len(bin_edges) - 1)]

    # 创建频率分布直方图
    bar = (
        Bar()
        .add_xaxis(x)
        .add_yaxis("频率分布", hist.tolist(), label_opts=opts.LabelOpts(is_show=False), z=2,
                   itemstyle_opts=opts.ItemStyleOpts(opacity=0.6))
    )

    # 创建正态分布曲线
    line = (
        Line()
        .set_global_opts(title_opts=opts.TitleOpts(title="正态分布曲线"))
        .add_xaxis(list(map(str, x)))
        .add_yaxis("正态分布", norm.pdf(bin_edges, loc=data[field].mean(), scale=data[field].std()).tolist(),
                   is_smooth=True, label_opts=opts.LabelOpts(is_show=False), z=1)
    )

    # 合并图表
    chart = bar.overlap(line)

    # 显示图例
    chart.set_series_opts(
        legend_opts=opts.LegendOpts(
            is_show=True, pos_top="5%"
        )
    )

    chart.set_global_opts(
        yaxis_opts=opts.AxisOpts(name=f"{field}"),
        title_opts=opts.TitleOpts(title="分布图"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {
                    "yAxisIndex": "none",
                    "title": {"zoom": "区域缩放", "back": "区域缩放还原"},
                },
                "dataView": {}
            },
        )
    )

    st_echarts.st_pyecharts(chart)


@st.cache_data(experimental_allow_widgets=True)
def dist_plot_wo_kde_pyecharts_2(data_A, data_B, field):
    hist1, bin_edges1 = np.histogram(data_A[field], bins=80, density=True)
    hist2, bin_edges2 = np.histogram(data_B[field], bins=80, density=True)

    x1 = [round((bin_edges1[i] + bin_edges1[i + 1]) / 2, 2) for i in range(len(bin_edges1) - 1)]
    x2 = [round((bin_edges2[i] + bin_edges2[i + 1]) / 2, 2) for i in range(len(bin_edges2) - 1)]
    y1 = hist1.tolist()
    y2 = hist2.tolist()

    bar = (
        Bar()
        .add_xaxis(x1)
        .add_yaxis(f'{field}_A', y1, label_opts=opts.LabelOpts(is_show=False), z=2,
                   itemstyle_opts=opts.ItemStyleOpts(opacity=0.6))
        .add_xaxis(x2)
        .add_yaxis(f'{field}_B', y2, label_opts=opts.LabelOpts(is_show=False), z=2,
                   itemstyle_opts=opts.ItemStyleOpts(opacity=0.6))
    )

    line = (
        Line()
        .add_xaxis(list(map(str, x1)))
        .add_yaxis('正态分布_A', (norm.pdf(bin_edges1, loc=data_A[field].mean(), scale=data_A[field].std())).tolist(),
                   is_smooth=True, label_opts=opts.LabelOpts(is_show=False), z=1)
        .add_xaxis(list(map(str, x2)))
        .add_yaxis('正态分布_B', (norm.pdf(bin_edges2, loc=data_B[field].mean(), scale=data_B[field].std())).tolist(),
                   is_smooth=True, label_opts=opts.LabelOpts(is_show=False), z=1)
    )

    chart = bar.overlap(line)

    chart.set_series_opts(
        legend_opts=opts.LegendOpts(
            is_show=True, pos_top="5%"
        )
    )

    chart.set_global_opts(
        xaxis_opts=opts.AxisOpts(name=f'{field}'),
        title_opts=opts.TitleOpts(title='频率分布直方图'),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {
                    "yAxisIndex": "none",
                    "title": {"zoom": "区域缩放", "back": "区域缩放还原"},
                },
                "dataView": {}
            },
        )
    )

    st_echarts.st_pyecharts(chart)


@st.cache_data(experimental_allow_widgets=True)
def dist_plot_pyecharts(data, field):
    # 计算频率分布
    hist, bin_edges = np.histogram(data[field], bins='auto', density=True)
    mid = [round((bin_edges[i] + bin_edges[i + 1]) / 2, 2) for i in range(len(bin_edges) - 1)]

    # 计算核密度估计
    density = gaussian_kde(data[field].values)
    x_range = bin_edges.tolist()
    # x_range = [_data[field].min() + x * (_data[field].max() - _data[field].min()) / 500 for x in range(500)]
    density_list = density(x_range)

    # 创建频率分布直方图
    bar = (
        Bar()
        .add_xaxis(mid)
        .add_yaxis("频率分布", hist.tolist(), label_opts=opts.LabelOpts(is_show=False),
                   itemstyle_opts=opts.ItemStyleOpts(opacity=0.65), z=10)
    )

    # 创建正态分布曲线
    line1 = (
        Line()
        .add_xaxis(list(map(str, mid)))
        .add_yaxis("正态分布", (norm.pdf(bin_edges, loc=data.mean(), scale=data.std())).tolist(),
                   is_smooth=True, label_opts=opts.LabelOpts(is_show=False), z=8)
    )

    # 创建核密度曲线
    line2 = (
        Line()
        .add_xaxis(list(map(str, mid)))
        .add_yaxis("核密度", density_list, is_smooth=True, label_opts=opts.LabelOpts(is_show=False), z=9)
    )

    # 合并图表
    chart = bar.overlap(line1).overlap(line2)

    # 显示图例
    chart.set_series_opts(
        legend_opts=opts.LegendOpts(
            is_show=True, pos_top="5%"
        )
    )

    chart.set_global_opts(
        yaxis_opts=opts.AxisOpts(name=f"{field}"),
        title_opts=opts.TitleOpts(title="分布图"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {
                    "yAxisIndex": "none",
                    "title": {"zoom": "区域缩放", "back": "区域缩放还原"},
                },
                "dataView": {}
            },
        )
    )

    st_echarts.st_pyecharts(chart)


@st.cache_data(experimental_allow_widgets=True)
def bar_plot_matplotlib(df: pd.DataFrame):  # bar plot 1111
    plt.figure(figsize=(20, 7))

    x = range(len(df))

    plt.axhline(y=0.91, xmin=0.86, xmax=0.89, color='red', linewidth=1)
    plt.annotate(df.columns[0], xy=(8, 0.9), xytext=(8.5, 0.9))
    plt.axhline(y=0.86, xmin=0.86, xmax=0.89, color='blue', linewidth=1)
    plt.annotate(df.columns[1], xy=(8, 0.85), xytext=(8.5, 0.85))

    for i in range(0, 2):
        y = df.iloc[:, i]
        if i == 1:
            plt.vlines([xi + 0.05 for xi in x], 0, y, color='blue', linewidth=1)
        else:
            plt.vlines(x, 0, y, color='red', linewidth=1)

    x_show = df.index.tolist()
    plt.xticks(x, x_show, fontsize=10)

    plt.title(df.columns.name, fontsize=20)

    # st.pyplot(plt)

    return plt.gcf()


@st.cache_data(experimental_allow_widgets=True)
def density_heatmap_continuous_lattice_point_matplotlib(x, y, bins=120, cmap='turbo', vmin=0.001):
    # 计算直方图
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # 翻转坐标轴
    counts = counts.T

    # 创建网格
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)

    # 设置颜色映射和最小值
    cmap_modified = mpl.colormaps.get_cmap(cmap)
    cmap_modified.set_under('white')

    plt.figure()

    # 绘制密度热力图
    plt.pcolormesh(xx, yy, counts, cmap=cmap_modified, vmin=vmin)

    # 添加颜色条
    plt.colorbar(extend='both')

    # 添加轴标签
    plt.xlabel('X')
    plt.ylabel('Y')

    # # 保存图片
    # plt.savefig('density_heatmap.png')

    # # 显示图形
    # plt.show()

    # st.pyplot(plt)

    return plt.gcf()


@st.cache_data(experimental_allow_widgets=True)
def density_heatmap_integer_lattice_point_matplotlib(coordinates, values, cmap='turbo', vmin=1):
    # 获取坐标范围
    x_coords, y_coords = coordinates.T[0], coordinates.T[1]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # 创建热力图数据
    heatmap_data = np.zeros((int(np.ceil(x_max - x_min)) + 1, int(np.ceil(y_max - y_min)) + 1))
    for coord, value in zip(coordinates, values):
        x, y = coord
        heatmap_data[x - x_min, y - y_min] = value

    # 设置颜色映射和最小值
    cmap_modified = mpl.colormaps.get_cmap(cmap=cmap)
    cmap_modified.set_under('white')

    # 创建新的Figure对象
    plt.figure()

    # 绘制热力图
    plt.imshow(heatmap_data.T, cmap=cmap_modified, vmin=vmin, origin='lower')
    plt.colorbar(extend='both')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap')
    # plt.show()
    # st.pyplot(plt)

    return plt.gcf()


@st.cache_data(experimental_allow_widgets=True)
def group_box_plot_plotly(melted_table_AB: pd.DataFrame):
    group = melted_table_AB['group'].unique()
    color_map = {group[0]: 'red', group[1]: 'blue'}
    fig = px.box(data_frame=melted_table_AB, x="item", y="value", color="group",
                 color_discrete_map=color_map)
    fig.update_traces(quartilemethod="linear")
    # fig.show()
    return fig
