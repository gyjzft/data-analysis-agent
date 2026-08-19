"""
LLM 客户端模块
封装 OpenAI API（兼容 DeepSeek、智谱等），提供统一的对话接口
"""
import re
import json
from openai import OpenAI
from src.config import Config


class LLMClient:
    """LLM 客户端，负责与大模型通信"""

    def __init__(self):
        self.client = OpenAI(
            api_key=Config.API_KEY,
            base_url=Config.BASE_URL,
        )
        self.model = Config.MODEL

    def _parse_deepseek_xml_tool_calls(self, content: str) -> list:
        """解析 DeepSeek XML 格式的工具调用"""
        tool_calls = []

        # 匹配 <｜｜DSML｜｜tool_calls> ... </｜｜DSML｜｜tool_calls>
        pattern = r'<\|\|DSML\|\|tool_calls>(.*?)</\|\|DSML\|\|tool_calls>'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            return tool_calls

        tool_calls_xml = match.group(1)

        # 匹配每个 invoke
        invoke_pattern = r'<\|\|DSML\|\|invoke name="([^"]+)">(.*?)</\|\|DSML\|\|invoke>'
        for invoke_match in re.finditer(invoke_pattern, tool_calls_xml, re.DOTALL):
            func_name = invoke_match.group(1)
            params_xml = invoke_match.group(2)

            # 提取参数
            args = {}
            param_pattern = r'<\|\|DSML\|\|parameter name="([^"]+)"[^>]*>(.*?)</\|\|DSML\|\|parameter>'
            for param_match in re.finditer(param_pattern, params_xml, re.DOTALL):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                # 尝试解析为 JSON
                try:
                    args[param_name] = json.loads(param_value)
                except json.JSONDecodeError:
                    args[param_name] = param_value

            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "function": func_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            })

        return tool_calls

    def chat(self, messages: list, tools: list = None, temperature: float = 0) -> dict:
        """
        发送对话请求

        参数:
            messages: 消息列表
            tools: 工具定义列表（Function Calling 用）
            temperature: 温度

        返回:
            {"content": "文字回复", "tool_calls": [工具调用列表]}
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        # 解析回复
        message = response.choices[0].message
        result = {
            "content": message.content,
            "tool_calls": [],
        }

        # 处理标准 OpenAI 格式的工具调用
        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        # 处理 DeepSeek XML 格式的工具调用
        elif message.content and "<｜｜DSML｜｜tool_calls>" in message.content:
            result["tool_calls"] = self._parse_deepseek_xml_tool_calls(message.content)
            # 清除 XML 部分，保留纯文本
            result["content"] = re.sub(
                r'<\|\|DSML\|\|tool_calls>.*?</\|\|DSML\|\|tool_calls>',
                '',
                message.content,
                flags=re.DOTALL
            ).strip()

        return result
