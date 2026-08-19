"""
Agent 核心（代码生成模式）
负责：理解用户意图 → 生成代码 → 沙箱执行 → 解读结果 → 生成回复

采用"代码生成"架构（类似 PandasAI）：
LLM 生成 Python 代码，我们拿到真实的 DataFrame 在本地执行，
再把真实结果交给 LLM 解读。彻底避免模型编造数据。
"""
import io
import os
import re
import uuid
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.config import Config
from src.llm.client import LLMClient
from src.tools import modeling


SYSTEM_PROMPT = """
你是一个专业的数据分析助手。数据已经加载到变量 df（pandas DataFrame）中。

你的工作方式：
1. 理解用户的分析需求
2. 编写 Python 代码来完成分析（代码用 ```python 代码块包裹）
3. 代码中可以使用：df、pd、np、plt、sns
4. 代码执行后会返回真实的输出结果，你根据结果给出清晰的中文解读

编写代码的规则：
- 所有数值必须通过实际计算得到，绝对不要编造数据
- 用 print() 输出关键结果
- 如果要画图：plt.figure(figsize=(8,5))，画完后调用 save_plot() 保存图片
- 画图时绝对不要直接调用 plt.savefig()，统一使用 save_plot() 助手函数
- save_plot() 会返回图片路径，你可以 print() 打印出来
- 注意处理缺失值，例如 df[col].dropna()
- 代码要简洁，一次完成一个分析任务

纯文字对话的规则：
- 如果用户只是打招呼、闲聊、或提出不涉及数据分析的问题（如"你好""谢谢""这个工具怎么用"）
  ，直接用文字回答，不要生成代码
- 只有用户明确要求分析数据时，才编写代码
- 代码执行如果出错，系统会自动把错误反馈给你，你只需修复代码后重新输出完整代码即可

构建预测模型的规则（重要）：
- 当用户要求"建模""训练模型""预测某列""构建回归/分类模型"时，不要编写建模代码
- 而是输出一行特殊标记，格式必须严格是：
  MODELING_REQ: {"target": "目标列名", "task_type": "regression 或 classification"}
- 判断规则：目标列是连续数值（如薪资、年龄）→ regression；目标列是类别（如是否离职）→ classification
- 输出标记后，系统会自动构建并保存模型，然后把结果交给你解读
"""


def _build_data_context(df: pd.DataFrame) -> str:
    """构建数据上下文信息，给 LLM 了解数据结构"""
    context = f"""当前数据集信息：
- 行数: {len(df)}
- 列数: {len(df.columns)}
- 列名和类型:"""
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        null_count = df[col].isnull().sum()
        context += f"\n  - {col} ({dtype}, 唯一值{nunique}个, 缺失{null_count}个)"
        if dtype in ["float64", "int64"]:
            context += f", 范围: [{df[col].min():.2f}, {df[col].max():.2f}]"
    return context


class DataAnalysisAgent:
    """数据分析 Agent（代码生成模式）"""

    def __init__(self):
        self.llm = LLMClient()
        self.df = None
        self.history = []
        self.image_path = None

    def load_data(self, file_path: str) -> str:
        """加载数据文件"""
        try:
            if file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith((".xls", ".xlsx")):
                self.df = pd.read_excel(file_path)
            else:
                return "不支持的文件格式，请上传 CSV 或 Excel 文件"

            # 初始化对话历史
            self.history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": _build_data_context(self.df)},
            ]
            self.image_path = None

            return f"✅ 数据加载成功！{len(self.df)} 行 x {len(self.df.columns)} 列"
        except Exception as e:
            return f"❌ 加载失败: {str(e)}"

    def _extract_code(self, text: str) -> str:
        """从 LLM 回复中提取 Python 代码"""
        # 匹配 ```python ... ``` 代码块
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 兼容只有 ``` 的代码块
        pattern2 = r"```\s*\n(.*?)```"
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            return match2.group(1).strip()

        return None

    def _execute_code(self, code: str) -> tuple:
        """在沙箱中执行代码，返回 (输出文本, 图片路径)"""
        # 记录执行前已有的图片（outputs 目录 + cwd 下的 outputs 目录），用于找出本次新生成的图片
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        cwd_outputs = os.path.join(os.getcwd(), "outputs")
        os.makedirs(cwd_outputs, exist_ok=True)
        before = set(os.listdir(Config.OUTPUT_DIR)) | set(os.listdir(cwd_outputs))

        # 生成一个本次会话唯一的图片前缀，供模型使用
        session_tag = uuid.uuid4().hex[:6]

        def save_plot(filename=None):
            """把当前 matplotlib 图形保存为图片，返回路径"""
            if filename is None:
                filename = f"{session_tag}_{uuid.uuid4().hex[:6]}.png"
            filepath = os.path.join(Config.OUTPUT_DIR, filename)
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            return filepath

        # 准备命名空间
        namespace = {
            "df": self.df,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "Config": Config,
            "save_plot": save_plot,
        }

        # 捕获 print 输出
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                exec(code, namespace)
            output = buffer.getvalue()

            # 找出本次新生成的图片（在 outputs/ 下新增的 .png）
            after = set(os.listdir(Config.OUTPUT_DIR)) | set(os.listdir(cwd_outputs))
            new_files = after - before
            new_pngs = []
            for f in new_files:
                if f.endswith(".png"):
                    for d in (Config.OUTPUT_DIR, cwd_outputs):
                        p = os.path.join(d, f)
                        if os.path.exists(p):
                            new_pngs.append(p)
                            break
            image = None
            if new_pngs:
                # 取最新的一个
                new_pngs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                image = new_pngs[0]

            return output, image
        except Exception as e:
            return f"代码执行出错: {type(e).__name__}: {str(e)}", None

    def chat(self, user_message: str) -> dict:
        """处理用户消息并返回回复"""
        if self.df is None:
            return {"text": "请先上传数据文件", "image": None}

        self.image_path = None
        self.history.append({"role": "user", "content": user_message})

        # 第一轮：让 LLM 生成代码
        response = self.llm.chat(
            messages=self.history,
            temperature=0,
        )
        code_text = response["content"]

        # 先检查是否是"构建模型"请求（沙箱主导，不走代码生成）
        modeling_result = self._handle_modeling_request(code_text)
        if modeling_result is not None:
            return modeling_result

        code = self._extract_code(code_text)

        # 没有代码 → 纯文字对话（打招呼、闲聊、非分析问题），直接返回
        if code is None:
            self.history.append({"role": "assistant", "content": code_text})
            return {"text": code_text, "image": None}

        # 执行代码；出错时自动让 LLM 修复，最多重试 2 次
        exec_output, image = self._execute_code(code)
        max_retries = 2
        attempt = 0

        while exec_output.startswith("代码执行出错") and attempt < max_retries:
            # 把代码和错误喂给 LLM，让它修复
            self.history.append({
                "role": "assistant",
                "content": f"我执行的代码：\n```python\n{code}\n```",
            })
            self.history.append({
                "role": "user",
                "content": f"上面的代码执行出错了：\n{exec_output}\n请修复代码，重新输出完整可运行的 Python 代码。",
            })
            fix_response = self.llm.chat(messages=self.history, temperature=0)
            code = self._extract_code(fix_response["content"])
            attempt += 1
            if code is None:
                break
            exec_output, image = self._execute_code(code)

        # 重试后仍然失败 → 友好提示，不展示原始代码
        if exec_output.startswith("代码执行出错"):
            friendly = (
                f"抱歉，这次分析没有成功。错误信息：{exec_output}\n\n"
                "你可以换个说法再试一次，或者简化一下需求。"
            )
            self.history.append({"role": "assistant", "content": friendly})
            return {"text": friendly, "image": None}

        # 把代码和结果加入历史，让 LLM 解读
        self.history.append({"role": "assistant", "content": f"我执行的代码：\n```python\n{code}\n```"})
        self.history.append({"role": "user", "content": f"代码执行结果如下，请根据这些真实结果给出清晰的中文分析：\n{exec_output}"})

        # 第二轮：LLM 解读真实结果
        final_response = self.llm.chat(
            messages=self.history,
            temperature=0.3,
        )
        reply_text = final_response["content"]
        self.history.append({"role": "assistant", "content": reply_text})

        return {"text": reply_text, "image": image}

    def _handle_modeling_request(self, llm_text: str):
        """检测并处理"构建模型"请求。返回回复 dict 或 None（不是建模请求）"""
        import json as _json
        pattern = r"MODELING_REQ:\s*(\{.*?\})"
        match = re.search(pattern, llm_text, re.DOTALL)
        if not match:
            return None

        try:
            req = _json.loads(match.group(1))
            target = req.get("target")
            task_type = req.get("task_type", "regression")
        except Exception:
            return None

        if not target or target not in self.df.columns:
            return {
                "text": f"无法识别要预测的目标列：'{target}'。请指定数据中存在的列名。",
                "image": None,
            }

        # 沙箱主导建模（固定、可靠的流程）
        try:
            result = modeling.build_model(self.df, target, task_type)
        except Exception as e:
            return {"text": f"建模失败：{str(e)}", "image": None}

        # 把真实结果交给 LLM 解读
        results_text = _json.dumps(result["results"], ensure_ascii=False)
        self.history.append({
            "role": "assistant",
            "content": f"我已构建模型（{task_type}，目标列：{target}），最佳模型是 {result['best_model']}。",
        })
        self.history.append({
            "role": "user",
            "content": (
                f"模型构建完成，这是真实的评估结果：{results_text}\n"
                f"模型ID：{result['model_id']}\n"
                f"特征列：{result['feature_cols']}\n"
                "请用清晰的中文向用户解读这些结果，并告诉用户模型已保存、"
                "可以在预测面板输入模型ID进行新数据预测。"
            ),
        })
        final_response = self.llm.chat(messages=self.history, temperature=0.3)
        reply_text = final_response["content"]
        self.history.append({"role": "assistant", "content": reply_text})
        return {"text": reply_text, "image": None}
