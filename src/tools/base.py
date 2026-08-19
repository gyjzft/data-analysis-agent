"""
工具基类
所有分析工具继承此类，统一接口
"""
from abc import ABC, abstractmethod
import pandas as pd


class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（给 LLM 看的）"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数定义（Function Calling schema）"""
        pass

    @abstractmethod
    def execute(self, df: pd.DataFrame, **kwargs) -> dict:
        """
        执行工具

        参数:
            df: 当前数据集
            **kwargs: 其他参数

        返回:
            {"result": 结果数据, "message": 文字描述, "image": 图片路径(可选)}
        """
        pass

    @property
    def to_function_schema(self) -> dict:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
