# 📊 数据分析 AutoAgent

> 一个本地部署的数据分析 AI Agent，上传 CSV/Excel 数据 + 自然语言指令，自动完成统计分析、机器学习建模，图文展示结果。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 功能特性

### 📈 描述统计
- 均值、标准差、分位数等描述统计
- 频次统计、相关性分析
- t检验、正态性检验、方差分析

### 📊 数据可视化
- 直方图、箱线图、柱状图
- 散点图、相关性热力图

### 🤖 机器学习建模
- **回归**：线性回归、决策树、随机森林、XGBoost
- **分类**：逻辑回归、决策树、随机森林、XGBoost
- **聚类**：K-Means
- 自动对比 4 个模型，选出最佳模型

### 🔮 模型保存与预测
- 训练后自动保存模型（含标准化参数、特征列名、标签编码器）
- 支持加载已保存模型对新数据预测

### 💬 自然语言交互
- 用自然语言描述分析需求
- 支持 DeepSeek、OpenAI 等兼容 API

---

## 🖼️ 效果展示

`
用户: "帮我看看数据的描述统计"
Agent: 描述统计完成，共 3 个数值列
       [显示统计表格]
       三个数值列均无缺失值，数据完整

用户: "画一个收入的直方图"
Agent: 已生成收入的直方图
       [显示直方图]
       收入数据呈现右偏分布，大多数集中在 5000-15000 元

用户: "建立一个回归模型预测消费"
Agent: 回归完成，最佳模型: XGBoost (R2=0.81)
       [显示特征重要性图]
`

---

## 🏗️ 项目结构

`
data_analysis_agent/
├── .env                    # API 配置（需自行创建）
├── .env.example            # 配置模板
├── requirements.txt        # Python 依赖
├── main.py                 # 命令行入口
├── src/
│   ├── config.py           # 配置管理
│   ├── agent/
│   │   └── core.py         # Agent 核心（代码生成→沙箱执行→解读）
│   │   └── modeling.py     # 沙箱主导建模（固定流程+完整保存）
│   ├── llm/
│   │   └── client.py       # LLM API 封装
│   ├── tools/
│   │   ├── base.py         # 工具基类
│   │   ├── registry.py     # 工具注册中心
│   │   ├── statistics_tools.py  # 描述统计
│   │   ├── visualization_tools.py  # 可视化
│   │   ├── inference_tools.py  # 统计推断
│   │   └── ml_tools.py     # ML建模 + 预测
│   └── ui/
│       └── app.py          # Gradio Web 界面
├── tests/
│   └── test_tools.py       # 工具测试脚本
├── docs/
│   ├── PRD.md              # 产品需求文档
│   ├── architecture_guide.md  # 架构解读
│   └── study_notes.md      # 学习笔记
├── examples/               # 示例数据
└── outputs/                # 分析结果输出
    └── models/             # 保存的模型
`

---

## 🚀 快速开始

### 1. 安装依赖

`ash
pip install -r requirements.txt
`

### 2. 配置 API Key

`ash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
`

.env 文件内容：
`nv
# DeepSeek（推荐）
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash

# 或 OpenAI
# OPENAI_API_KEY=sk-your-openai-key
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini
`

### 3. 运行

**方式一：Web 界面（推荐）**
`ash
python src/ui/app.py
`
打开浏览器访问 http://127.0.0.1:7860

**方式二：命令行**
`ash
python main.py
`

---

## 💡 使用示例

### Web 界面操作

1. 上传 CSV/Excel 文件
2. 在对话框输入分析指令
3. 查看文字分析 + 图表结果

### 常用指令

| 指令 | 功能 |
|------|------|
| 帮我看看数据的描述统计 | 描述性统计 |
| 画一个收入的直方图 | 直方图 |
| 分析各列之间的相关性 | 相关性分析 |
| 检验收入是否服从正态分布 | 正态性检验 |
| 比较不同地区的收入差异 | t检验/方差分析 |
| 建立一个回归模型预测消费 | 回归建模 |
| 对数据进行聚类分析 | K-Means 聚类 |

---

## 🔧 技术架构

`
用户输入 → [Gradio UI] → [Agent 核心] → [LLM API]
                              │
                              ├── [沙箱: 执行 LLM 生成的代码]
                              ├── [工具: 描述统计（休眠）]
                              ├── [工具: 数据可视化（休眠）]
                              ├── [工具: 统计推断（休眠）]
                              └── [工具: 模型预测（在用）]
`

- **代码生成**：LLM 生成 Python 代码，本地沙箱用真实数据执行（类似 PandasAI）
- **两次调用**：第一次写代码，第二次解读真实结果，杜绝模型编造数据
- **工具架构保留**：BaseTool + 注册中心仍保留，供"模型预测"功能使用及学习
- **数据安全**：数据不上传 LLM，仅发送数据 schema
- **沙箱主导建模**：构建预测模型时，由固定可靠的流程自动建模+完整保存（ColumnTransformer 打包预处理，不依赖 LLM 写代码，保存可靠性 ~100%）

---

## 🧪 运行测试

`ash
# 测试所有工具（不需要 API Key）
python tests/test_tools.py
`

---

## 📝 扩展开发

### 扩展开发（当前代码生成模式）

1. 在 src/agent/core.py 的 SYSTEM_PROMPT 里告诉 LLM 可以用什么
2. 在 _execute_code() 的命名空间里加入新的可用对象
3. 完成！LLM 会自动写代码使用它

### 模型预测（动态输入框）

1. 建模完成后，复制返回的模型ID
2. 在预测面板输入模型ID → 点击「加载模型」
3. 系统根据模型特征自动生成输入框（数值→数字框，类别→下拉框）
4. 填写特征值 → 点击「预测」→ 显示结果
5. 「关闭预测」收起面板

> 注：工具架构（BaseTool + 注册中心）仍保留在 src/tools/，
> 供"模型预测"功能使用和学习。换用支持函数调用的模型后可切回工具模式。

## 📄 License

MIT License

---

## 🙏 致谢

本项目作为 AI Agent 学习项目，涵盖了以下技术栈：
- Python 数据处理（Pandas、NumPy）
- 数据可视化（Matplotlib、Seaborn）
- 机器学习（Scikit-learn、XGBoost）
- 统计推断（Scipy）
- LLM Agent（Function Calling）
- Web UI（Gradio）
