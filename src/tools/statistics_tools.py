"""
描述统计工具
提供数据描述性统计功能
"""
import pandas as pd
import numpy as np
from src.tools.base import BaseTool


class DescribeDataTool(BaseTool):
    """数据描述统计"""

    @property
    def name(self):
        return "describe_data"

    @property
    def description(self):
        return "对数据进行描述性统计，包括均值、标准差、最小值、最大值、分位数等"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要统计的列名列表，为空则统计所有数值列",
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
        if target_df.empty:
            return {"result": None, "message": "没有数值列可以进行描述统计"}
        stats = target_df.describe().round(2)
        return {"result": stats, "message": f"描述统计完成，共 {len(stats.columns)} 个数值列"}


class ValueCountsTool(BaseTool):
    """频次统计"""

    @property
    def name(self):
        return "value_counts"

    @property
    def description(self):
        return "统计某一列中每个值出现的次数（频次分布）"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "要统计的列名"},
                "top_n": {"type": "integer", "description": "只显示前 N 个，默认全部显示"},
            },
            "required": ["column"],
        }

    def execute(self, df, **kwargs):
        column = kwargs["column"]
        top_n = kwargs.get("top_n", None)
        if column not in df.columns:
            return {"result": None, "message": f"列 {column} 不存在"}
        counts = df[column].value_counts()
        if top_n:
            counts = counts.head(top_n)
        return {"result": counts, "message": f"{column} 列共有 {df[column].nunique()} 个不同值"}


class CorrelationTool(BaseTool):
    """相关性分析"""

    @property
    def name(self):
        return "correlation"

    @property
    def description(self):
        return "计算数值列之间的相关系数矩阵"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要分析的列名列表，为空则分析所有数值列",
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
            return {"result": None, "message": "需要至少 2 个数值列才能计算相关性"}
        corr = target_df.corr().round(3)
        return {"result": corr, "message": f"相关性矩阵计算完成，共 {len(corr.columns)} 个变量"}