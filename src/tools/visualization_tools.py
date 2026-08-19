"""
数据可视化工具
支持柱状图、箱线图、直方图、散点图、相关性热力图
"""
import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from src.tools.base import BaseTool
from src.config import Config


class PlotTool(BaseTool):
    """可视化基类，封装画图公共逻辑"""

    def _save_fig(self):
        """保存当前图表到文件，返回路径"""
        plt.tight_layout()
        filename = f"{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(Config.OUTPUT_DIR, filename)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()
        return filepath


class HistogramTool(PlotTool):
    """直方图"""

    @property
    def name(self):
        return "plot_histogram"

    @property
    def description(self):
        return "绘制数值列的直方图，展示数据分布"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "要绘制的数值列名"},
                "bins": {"type": "integer", "description": "直方图的柱数，默认30"},
            },
            "required": ["column"],
        }

    def execute(self, df, **kwargs):
        column = kwargs["column"]
        bins = kwargs.get("bins", 30)
        if column not in df.columns:
            return {"result": None, "message": f"列 {column} 不存在", "image": None}
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column].dropna(), bins=bins, kde=True, color="steelblue")
        plt.title(f"{column} 分布直方图")
        plt.xlabel(column)
        plt.ylabel("频次")
        path = self._save_fig()
        return {"result": None, "message": f"已生成 {column} 的直方图", "image": path}


class BoxplotTool(PlotTool):
    """箱线图"""

    @property
    def name(self):
        return "plot_boxplot"

    @property
    def description(self):
        return "绘制箱线图，展示数据分布和异常值"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "数值列名"},
                "by": {"type": "string", "description": "分组列名（可选）"},
            },
            "required": ["column"],
        }

    def execute(self, df, **kwargs):
        column = kwargs["column"]
        by = kwargs.get("by", None)
        if column not in df.columns:
            return {"result": None, "message": f"列 {column} 不存在", "image": None}
        plt.figure(figsize=(8, 5))
        if by and by in df.columns:
            sns.boxplot(data=df, x=by, y=column, palette="Set2")
            plt.title(f"{by} 分组下 {column} 的箱线图")
        else:
            sns.boxplot(data=df, y=column, color="steelblue")
            plt.title(f"{column} 箱线图")
        path = self._save_fig()
        return {"result": None, "message": f"已生成 {column} 的箱线图", "image": path}


class BarplotTool(PlotTool):
    """柱状图"""

    @property
    def name(self):
        return "plot_barplot"

    @property
    def description(self):
        return "绘制柱状图，比较不同类别的数值"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "X轴类别列名"},
                "y": {"type": "string", "description": "Y轴数值列名"},
            },
            "required": ["x", "y"],
        }

    def execute(self, df, **kwargs):
        x = kwargs["x"]
        y = kwargs["y"]
        if x not in df.columns or y not in df.columns:
            return {"result": None, "message": "列不存在", "image": None}
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x=x, y=y, palette="Set2", ci="sd")
        plt.title(f"{x} vs {y} 柱状图")
        plt.xticks(rotation=45)
        path = self._save_fig()
        return {"result": None, "message": "已生成柱状图", "image": path}


class ScatterTool(PlotTool):
    """散点图"""

    @property
    def name(self):
        return "plot_scatter"

    @property
    def description(self):
        return "绘制散点图，展示两个数值变量的关系"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "X轴列名"},
                "y": {"type": "string", "description": "Y轴列名"},
            },
            "required": ["x", "y"],
        }

    def execute(self, df, **kwargs):
        x = kwargs["x"]
        y = kwargs["y"]
        if x not in df.columns or y not in df.columns:
            return {"result": None, "message": "列不存在", "image": None}
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=x, y=y, alpha=0.6, color="steelblue")
        plt.title(f"{x} vs {y} 散点图")
        path = self._save_fig()
        return {"result": None, "message": "已生成散点图", "image": path}


class HeatmapTool(PlotTool):
    """相关性热力图"""

    @property
    def name(self):
        return "plot_heatmap"

    @property
    def description(self):
        return "绘制相关性热力图，展示多个数值变量之间的相关关系"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要分析的列名列表，为空则使用所有数值列",
                }
            },
            "required": [],
        }

    def execute(self, df, **kwargs):
        columns = kwargs.get("columns", [])
        if columns:
            target_df = df[columns]
        else:
            target_df = df.select_dtypes(include=[np.number])
        if target_df.empty or len(target_df.columns) < 2:
            return {"result": None, "message": "需要至少 2 个数值列", "image": None}
        plt.figure(figsize=(10, 8))
        corr = target_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                    square=True, linewidths=0.5)
        plt.title("相关性热力图")
        path = self._save_fig()
        return {"result": corr, "message": "已生成相关性热力图", "image": path}
