"""
统计推断工具
t检验、方差齐性检验、正态性检验、卡方检验
"""
import pandas as pd
import numpy as np
from scipy import stats
from src.tools.base import BaseTool


class TTestTool(BaseTool):
    """t检验：比较两组均值是否有显著差异"""

    @property
    def name(self):
        return "t_test"

    @property
    def description(self):
        return "对两组数据做t检验，判断均值是否存在显著差异。适用于两组独立样本。"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "数值列名"},
                "group_column": {"type": "string", "description": "分组列名（只能有2个组）"},
            },
            "required": ["column", "group_column"],
        }

    def execute(self, df, **kwargs):
        column = kwargs["column"]
        group_column = kwargs["group_column"]
        groups = df[group_column].dropna().unique()
        if len(groups) != 2:
            return {"result": None, "message": f"t检验需要恰好2个组，当前有{len(groups)}个组"}
        g1 = df[df[group_column] == groups[0]][column].dropna()
        g2 = df[df[group_column] == groups[1]][column].dropna()
        stat, p_value = stats.ttest_ind(g1, g2)
        result = {"statistic": round(stat, 4), "p_value": round(p_value, 6),
                  "significant": p_value < 0.05}
        msg = f"t={stat:.4f}, p={p_value:.6f}，"
        msg += "差异显著（p<0.05）" if p_value < 0.05 else "差异不显著（p>=0.05）"
        return {"result": result, "message": msg}


class NormalityTestTool(BaseTool):
    """正态性检验"""

    @property
    def name(self):
        return "normality_test"

    @property
    def description(self):
        return "Shapiro-Wilk正态性检验，判断数据是否服从正态分布"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "数值列名"},
            },
            "required": ["column"],
        }

    def execute(self, df, **kwargs):
        column = kwargs["column"]
        data = df[column].dropna()
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
        stat, p_value = stats.shapiro(data)
        result = {"statistic": round(stat, 4), "p_value": round(p_value, 6),
                  "normal": p_value > 0.05}
        msg = f"W={stat:.4f}, p={p_value:.6f}，"
        msg += "服从正态分布（p>0.05）" if p_value > 0.05 else "不服从正态分布（p<=0.05）"
        return {"result": result, "message": msg}


class ANOVATestTool(BaseTool):
    """方差分析"""

    @property
    def name(self):
        return "anova_test"

    @property
    def description(self):
        return "单因素方差分析（ANOVA），比较三组及以上均值是否有显著差异"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "数值列名"},
                "group_column": {"type": "string", "description": "分组列名"},
            },
            "required": ["column", "group_column"],
        }

    def execute(self, df, **kwargs):
        column = kwargs["column"]
        group_column = kwargs["group_column"]
        groups = [g[column].dropna().values for _, g in df.groupby(group_column)]
        stat, p_value = stats.f_oneway(*groups)
        result = {"statistic": round(stat, 4), "p_value": round(p_value, 6),
                  "significant": p_value < 0.05}
        msg = f"F={stat:.4f}, p={p_value:.6f}，"
        msg += "至少两组差异显著（p<0.05）" if p_value < 0.05 else "各组差异不显著（p>=0.05）"
        return {"result": result, "message": msg}
