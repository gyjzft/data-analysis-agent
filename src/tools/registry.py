"""
工具注册中心
统一管理所有工具的注册、查找和调用
"""
from src.tools.base import BaseTool
from src.tools.statistics_tools import DescribeDataTool, ValueCountsTool, CorrelationTool
from src.tools.visualization_tools import (HistogramTool, BoxplotTool, BarplotTool,
                                           ScatterTool, HeatmapTool)
from src.tools.inference_tools import TTestTool, NormalityTestTool, ANOVATestTool
from src.tools.ml_tools import RegressionTool, ClassificationTool, ClusteringTool, PredictTool


class ToolRegistry:
    """工具注册中心 - 单例模式"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._register_all()
        return cls._instance

    def _register_all(self):
        """注册所有工具"""
        tools = [
            # 描述统计
            DescribeDataTool(),
            ValueCountsTool(),
            CorrelationTool(),
            # 可视化
            HistogramTool(),
            BoxplotTool(),
            BarplotTool(),
            ScatterTool(),
            HeatmapTool(),
            # 统计推断
            TTestTool(),
            NormalityTestTool(),
            ANOVATestTool(),
            # ML建模
            RegressionTool(),
            ClassificationTool(),
            ClusteringTool(),
            # 预测
            PredictTool(),
        ]
        for tool in tools:
            self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """根据名称获取工具"""
        return self._tools.get(name)

    def get_all_schemas(self) -> list:
        """获取所有工具的 Function Calling schema"""
        return [tool.to_function_schema for tool in self._tools.values()]

    def execute(self, name: str, df, **kwargs) -> dict:
        """执行指定工具"""
        tool = self.get_tool(name)
        if tool is None:
            return {"result": None, "message": f"工具 {name} 不存在", "image": None}
        try:
            return tool.execute(df, **kwargs)
        except Exception as e:
            return {"result": None, "message": f"执行出错: {str(e)}", "image": None}

    @property
    def tool_names(self) -> list:
        return list(self._tools.keys())
