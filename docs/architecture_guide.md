# 项目架构解读指南

> 师傅带徒弟读懂每一行代码 | 版本 v1.1 | 2026-08-18

> ⚠️ **v1.1 重要变更**：Agent 核心已从"函数调用"改为"代码生成"架构，
> 原因和详细改动见 [CHANGELOG.md](CHANGELOG.md)。本指南以下内容已同步更新。

---

## 一、项目全景图

想象这个 Agent 是一个人：

`
┌─────────────────────────────────────────────────────────┐
│                      🧠 大脑（Agent 核心）                │
│                   src/agent/core.py                      │
│  负责：理解你想做什么 → 决定用什么工具 → 解读结果          │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌──────────────────────────┐
│   📝 程序员（LLM）     │      │   💻 执行器（沙箱）        │
│   src/llm/client.py   │      │   src/agent/core.py       │
│  把需求翻译成 Python  │      │  执行 LLM 写的代码         │
│  代码                 │      │  用真实 df 计算+画图       │
└──────────────────────┘      └──────────────────────────┘
           ▲
           │
┌──────────┴──────────┐
│   👤 用户（界面）      │
│   src/ui/app.py       │
│  Gradio Web 界面       │
└─────────────────────┘
`

---

## 二、一次完整的对话是怎么跑通的？（代码生成模式）

用户说：**"帮我看看收入的分布"**

`
步骤 1：用户输入
──────────────────────────────────────────
用户在 Gradio 界面输入文字，点击发送
    ↓
handle_chat() 被调用
    ↓
调用 agent.chat("帮我看看收入的分布")


步骤 2：LLM 生成代码
──────────────────────────────────────────
agent.chat() 做了什么：
    ① 把用户消息加入对话历史 self.history
    ② 把"系统提示词 + 数据上下文 + 对话历史"一起发给 LLM
    ③ 让 LLM 写一段 Python 代码（用 ```python 代码块包裹）

    发给 LLM 的内容大概是：
    ┌─────────────────────────────────────┐
    │ System: 你是数据分析助手...           │
    │ System: 数据已加载到 df，共200行...   │
    │ User: 帮我看看收入的分布              │
    └─────────────────────────────────────┘


步骤 3：提取代码
──────────────────────────────────────────
LLM 返回：
    ```python
    print(df["收入"].describe())
    plt.figure(figsize=(8,5))
    sns.histplot(df["收入"], kde=True)
    save_plot()
    ```

    agent._extract_code() 把代码从代码块里提取出来


步骤 4：沙箱执行（真实计算）
──────────────────────────────────────────
agent._execute_code() 在本地执行代码：
    命名空间里有：df（真实数据）、pd、np、plt、sns、save_plot
    → 捕获 print 输出（真实统计数字）
    → 捕获新生成的图片路径

    返回 {"output": "count 200, mean 15707.04 ...", "image": "outputs/xxx.png"}


步骤 5：LLM 解读真实结果
──────────────────────────────────────────
Agent 把真实输出再发给 LLM：
    "代码执行结果如下，请根据这些真实结果给出清晰的中文分析：count 200, mean ..."

LLM 回复：
    "收入数据右偏分布，均值 15707 元，中位数 15431 元。
     直方图显示大部分集中在 10000-20000 元区间。[附直方图]"


步骤 6：展示给用户
──────────────────────────────────────────
Agent 返回 {"text": "收入数据右偏...", "image": "outputs/xxx.png"}
    ↓
Gradio 界面显示文字 + 图片
`

---

## 三、每个文件到底干了什么？

### 3.1 src/config.py — 配置管理

`python
class Config:
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
`

**作用**：一个全局配置中心，所有参数从这里读。改配置只需要改 .env 文件。

**设计思想**：配置和代码分离。代码不硬编码任何可变的值。


### 3.2 src/llm/client.py — LLM 翻译官

`python
class LLMClient:
    def chat(self, messages, tools=None, temperature=0):
        response = self.client.chat.completions.create(...)
        # 解析回复：文字内容 + 工具调用列表
        return {"content": ..., "tool_calls": [...]}
`

**作用**：封装 OpenAI API，统一返回格式。Agent 只和 LLMClient 打交道，不用管底层 API 细节。

**设计思想**：适配器模式。如果以后换 DeepSeek、换 Claude，只改这一层就行。


### 3.3 src/tools/base.py — 工具蓝图（休眠）

`python
class BaseTool(ABC):
    @property
    def name(self):          # 工具名称（LLM 调用时用）
    @property
    def description(self):   # 工具描述（LLM 据此决定何时调用）
    @property
    def parameters(self):    # 参数定义（JSON Schema 格式）
    def execute(self, df):   # 实际执行逻辑
`

**作用**：定义了所有工具的"模具"。每个工具必须有名字、描述、参数、执行方法。

**设计思想**：模板方法模式。子类只需要填充具体的执行逻辑，接口统一。

**当前状态**：💤 休眠。聊天主流程不再调用工具，但这些代码是很好的设计模式学习素材。


### 3.4 src/tools/registry.py — 工具仓库（部分在用）

`python
class ToolRegistry:
    def _register_all(self):
        # 注册所有 14 个工具
        self._tools = {
            "describe_data": DescribeDataTool(),
            "plot_histogram": HistogramTool(),
            ...
        }

    def execute(self, name, df, **kwargs):
        # 根据名字找到工具并执行
        tool = self._tools[name]
        return tool.execute(df, **kwargs)
`

**作用**：统一管理所有工具。Agent 不需要知道具体有哪些工具，只需要 
egistry.execute("工具名")。

**设计思想**：注册模式 + 单例模式。全局唯一仓库，任何模块都能访问。

**当前状态**：⚠️ 部分在用。聊天主流程不再使用，但界面的"模型预测"功能（`app.py` 的 `handle_predict`）仍通过它调用 `predict` 工具。


### 3.5 src/agent/core.py — 大脑（代码生成模式）

`python
class DataAnalysisAgent:
    def load_data(self, file_path):
        # 读取 CSV/Excel，构建数据上下文

    def chat(self, user_message):
        # 核心循环：
        # 1. 给 LLM 发消息，让它生成代码
        # 2. 提取代码（_extract_code）
        # 3. 沙箱执行代码，拿真实结果（_execute_code）
        # 4. 把真实结果再发给 LLM 解读
        # 5. 返回最终回复
`

**作用**：Agent 的核心。让 LLM 生成代码 → 本地用真实数据执行 → 再解读结果。

**设计思想**：代码生成（Code Generation），类似 PandasAI。计算一定发生在本地，杜绝模型编造数据。

**当前状态**：✅ 在用。这是当前聊天主流程的核心。


### 3.6 src/ui/app.py — 用户界面

`python
with gr.Blocks() as demo:
    file_input = gr.File(...)
    chat_box = gr.Chatbot(...)
    msg_box = gr.Textbox(...)
`

**作用**：Gradio Web 界面。用户上传文件、输入指令、查看结果的地方。

**设计思想**：界面和逻辑分离。UI 只负责展示，所有分析逻辑在 Agent 里。


### 3.7 main.py — 命令行入口

`python
agent = DataAnalysisAgent()
file_path = input("文件路径: ")
agent.load_data(file_path)
while True:
    user_input = input("指令: ")
    response = agent.chat(user_input)
`

**作用**：命令行版本的入口。先测通逻辑，再上 Web 界面。

---

## 四、14 个工具一览

### 描述统计类（3个）
| 工具 | 干啥 | 输入 | 输出 |
|------|------|------|------|
| describe_data | 均值/标准差/分位数 | 列名列表 | 统计表格 |
| alue_counts | 每个值出现次数 | 列名 | 频次表 |
| correlation | 相关系数矩阵 | 列名列表 | 相关系数表 |

### 可视化类（5个）
| 工具 | 干啥 | 输入 | 输出 |
|------|------|------|------|
| plot_histogram | 直方图 | 列名 | 图片 |
| plot_boxplot | 箱线图 | 列名+分组 | 图片 |
| plot_barplot | 柱状图 | X列+Y列 | 图片 |
| plot_scatter | 散点图 | X列+Y列 | 图片 |
| plot_heatmap | 相关性热力图 | 列名列表 | 图片+相关系数表 |

### 统计推断类（3个）
| 工具 | 干啥 | 输入 | 输出 |
|------|------|------|------|
| 	_test | 两组均值差异检验 | 数值列+分组列 | t值+p值 |
| 
ormality_test | 正态性检验 | 列名 | W值+p值 |
| nova_test | 多组均值差异检验 | 数值列+分组列 | F值+p值 |

### ML建模类（3个）
| 工具 | 干啥 | 输入 | 输出 |
|------|------|------|------|
| 
egression | 4个回归模型对比 | 目标列名 | 评估指标+特征重要性图 |
| classification | 4个分类模型对比 | 目标列名 | 准确率+特征重要性图 |
| clustering | K-Means 聚类 | K值 | 聚类结果+可视化图 |

---

## 五、核心设计模式

### 5.1 代码生成（Code Generation，当前模式）

`
没有代码生成的世界：
  用户: "画直方图"
  LLM: "你可以用 plt.hist(df['收入']) 来画"
  用户: 手动复制代码运行 → 手动传结果回去

有代码生成的世界：
  用户: "画直方图"
  LLM: 生成代码 → 程序自动执行 → 返回图片和真实结果
  程序: 把结果交给 LLM 解读
  用户: 直接看到结果
`

**关键**：计算一定发生在本地沙箱里，用真实数据执行，LLM 无法编造数字。

### 5.2 两次 LLM 调用

Agent 的 chat() 方法里，LLM 被调用了**两次**：

`
第一次：写代码
  → 输入：用户消息 + 数据上下文
  → 输出：Python 代码

第二次：解读
  → 输入：代码的真实执行结果
  → 输出：自然语言回复（给用户看的）

为什么不是1次？
  如果1次完成，LLM 可能边算边编造数据。
  2次分离，第一次只"写代码"，第二次只"解读真实结果"，保证数据真实。
`

### 5.3 工具注册中心（已休眠，留作学习）

`
为什么要 Registry？

没有 Registry：
  agent 里写死：
    if name == "describe_data": tool = DescribeDataTool()
    elif name == "plot_histogram": tool = HistogramTool()
    elif ... （14个 elif）

  每加一个工具，都要改 agent 代码。

有 Registry：
  registry.execute(name, df, **kwargs)
  
  每加一个工具，只需要在 registry 里注册一行。
  agent 代码完全不用改。
`

**关键**：开闭原则——对扩展开放（加工具），对修改关闭（不改 agent）。

---

## 六、数据流全景

`
┌──────────────────────────────────────────────────────────────┐
│                         用户上传 CSV                          │
│                              │                                │
│                              ▼                                │
│                    ┌─────────────────┐                        │
│                    │  pd.read_csv()   │                        │
│                    └────────┬────────┘                        │
│                             │                                 │
│                             ▼                                 │
│                    ┌─────────────────┐                        │
│                    │  self.df (内存)  │ ← 数据常驻内存          │
│                    └────────┬────────┘                        │
│                             │                                 │
│     ┌───────────────────────┼───────────────────────┐        │
│     │                       │                       │        │
│     ▼                       ▼                       ▼        │
│ 描述统计工具           可视化工具               ML建模工具     │
│ (读 df 计算)         (读 df 画图)            (读 df 训练)    │
│     │                       │                       │        │
│     ▼                       ▼                       ▼        │
│  表格结果              图片文件                 指标+图片     │
│     │                       │                       │        │
│     └───────────────────────┼───────────────────────┘        │
│                             │                                 │
│                             ▼                                 │
│                    ┌─────────────────┐                        │
│                    │  LLM 生成回复    │                        │
│                    └────────┬────────┘                        │
│                             │                                 │
│                             ▼                                 │
│                    ┌─────────────────┐                        │
│                    │  展示给用户      │                        │
│                    └─────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
`

**关键点**：
- 数据只加载一次，常驻内存（self.df）
- 所有工具共享同一个 DataFrame
- 工具之间互不感知，只通过 Agent 协调

---

## 七、阅读代码的推荐顺序

`
【当前主流程（代码生成模式）】
第1步：config.py          （2分钟）— 知道配置怎么读的
第2步：client.py          （5分钟）— 知道LLM怎么调的
第3步：core.py            （25分钟）— 核心！代码生成循环 + 沙箱执行
第4步：app.py             （10分钟）— 知道界面怎么工作的

【学习用（工具架构，已休眠）】
第5步：base.py            （5分钟）— 知道工具长什么样
第6步：statistics_tools.py （10分钟）— 看第一个具体工具怎么写的
第7步：visualization_tools.py （10分钟）— 看画图工具怎么写的
第8步：ml_tools.py        （15分钟）— 看ML工具怎么写的
第9步：registry.py        （5分钟）— 知道工具怎么管理的
第10步：test_tools.py     （10分钟）— 知道怎么测试的

建议：先看懂主流程（1-4），再回头学习工具架构（5-10）
`

---

## 八、扩展思路

### 方案 A：在当前代码生成模式下扩展

`
1. 在 SYSTEM_PROMPT 里告诉 LLM 可以用哪些库/函数
2. 在 _execute_code() 的命名空间里加入新的可用对象
3. 完成！LLM 会自动写代码使用它
`

### 方案 B：换支持函数调用的模型后切回工具模式

`
1. 把 src/agent/core.py 换回旧的"函数调用"版本（git 历史里可找回）
2. 工具架构原封不动，直接就能用
`
