# 数据分析 AutoAgent — 学习笔记

> 师傅带徒弟：Python 核心语法与项目设计 | 2026-08-17

---

## 目录

1. [配置管理（config.py）](#1-配置管理)
2. [包机制（__init__.py）](#2-包机制)
3. [抽象类与接口（ABC）](#3-抽象类与接口)
4. [参数传递（**kwargs）](#4-参数传递)
5. [字典取值（dict.get）](#5-字典取值)
6. [JSON Schema（parameters 设计）](#6-json-schema)
7. [uuid 与 Agg](#7-uuid-与-agg)
8. [下划线命名约定](#8-下划线命名约定)
9. [文件操作（os + with）](#9-文件操作)
10. [matplotlib 画图参数](#10-matplotlib-画图参数)
11. [seaborn 可视化参数](#11-seaborn-可视化参数)
12. [条件分支（if/else）](#12-条件分支)
13. [错误处理（try/except）](#13-错误处理)
14. [列表操作（remove）](#14-列表操作)
15. [LabelEncoder（文字转数字）](#15-labelencoder)
16. [模型保存（joblib）](#16-模型保存)
17. [单例模式（__new__）](#17-单例模式)
18. [类里 vs 类外](#18-类里-vs-类外)

---

## 1. 配置管理

### load_dotenv()

`python
from dotenv import load_dotenv
load_dotenv()  # 把 .env 文件的内容加载到环境变量
`

- .env 文件：纯文本，存 API Key 等配置
- load_dotenv()：让 Python 能读到 .env 里的键值对
- 好处：代码里不写死敏感信息，换环境只改 .env

### class Config 为什么没参数？

`python
class Config:
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")
`

- 用**类属性**（不是实例属性），全局唯一
- 不需要创建实例，直接 Config.API_KEY 访问
- : str 是类型注解，告诉读者"这是字符串"

### os.getenv("KEY", "默认值")

`python
os.getenv("OPENAI_API_KEY", "")
#        ↑ 键名          ↑ 找不到时返回这个
`

- 第一个参数：要查找的键名
- 第二个参数：默认值（找不到时返回）
- 键名可以改，但 .env 文件里的键名要对应

### os.path.dirname 嵌套

`python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
`

从里到外：
1. __file__ → 当前文件路径
2. os.path.abspath() → 转成绝对路径
3. os.path.dirname() → 取目录（去掉文件名）
4. 再套一层 dirname() → 再往上一层

**效果**：无论在哪运行，都能自动定位到项目根目录。

---

## 2. 包机制

### __init__.py 的作用

`
没有 __init__.py → 普通文件夹，不能 import
有 __init__.py   → Python 包，可以 import
`

每个目录都需要一个，一一对应。空的就行。

---

## 3. 抽象类与接口

### from abc import ABC, abstractmethod

`python
from abc import ABC, abstractmethod

class BaseTool(ABC):
    @abstractmethod
    def name(self) -> str:
        pass  # 子类必须实现
`

- ABC：抽象基类，不能直接创建实例
- @abstractmethod：子类**必须**实现，否则报错

### @property

`python
@property
def name(self) -> str:
    return "plot_histogram"

# 使用时不需要括号
tool.name    # ✅
tool.name()  # ❌
`

把方法变成"属性"，访问时不需要加括号。

### pass

`python
pass  # 什么都不做，占位用
`

语法要求函数/类里必须有内容，pass 占位。

---

## 4. 参数传递

### **kwargs

`python
def execute(self, df, **kwargs):
    column = kwargs.get("column")
    bins = kwargs.get("bins", 30)
`

- **kwargs = 接收任意多个关键字参数，打包成字典
- 调用：	ool.execute(df, column="收入", bins=30)
- 函数内：kwargs = {"column": "收入", "bins": 30}

---

## 5. 字典取值

### dict.get(key, 默认值)

`python
d = {"a": 1, "b": 2}

d["a"]        # → 1
d["c"]        # ❌ KeyError

d.get("c")       # → None
d.get("c", [])   # → []（找不到返回默认值）
`

**安全取值，取不到也不报错。**

---

## 6. JSON Schema

### parameters 怎么设计？

`python
{
    "type": "object",
    "properties": {
        "column": {
            "type": "string",
            "description": "要绘制的数值列名"
        },
        "bins": {
            "type": "integer",
            "description": "柱数，默认30"
        }
    },
    "required": ["column"]  # 必填参数
}
`

**设计步骤**：
1. 看 xecute 需要哪些参数
2. 确定每个参数的名字、类型、说明
3. 哪些必填写在 
equired 里

这是给 LLM 看的"说明书"，LLM 据此知道怎么调用工具。

---

## 7. uuid 与 Agg

### uuid

`python
import uuid
filename = f"{uuid.uuid4().hex[:8]}.png"
# → "a3f2b1c4.png"（每次不同）
`

生成不重复的随机字符串，防止图片被覆盖。

### matplotlib.use("Agg")

`python
matplotlib.use("Agg")  # 不弹窗，只保存文件
`

- 默认：画图会弹窗
- Agg：不弹窗，保存为文件（适合命令行/服务器）

---

## 8. 下划线命名约定

| 写法 | 含义 |
|------|------|
| 
ame() | 公共方法 |
| _name() | 内部方法（别从外面调） |
| __name() | 私有方法（强制内部） |

`python
def _save_fig(self):   # 单下划线 = 内部方法
    ...
`

---

## 9. 文件操作

### os.makedirs + exist_ok

`python
os.makedirs("outputs", exist_ok=True)
# 目录不存在 → 创建
# 目录已存在 → 跳过，不报错
`

### with open() as f

`python
with open("test.txt", "w") as f:
    f.write("hello")
# 退出 with 块 → 自动关闭文件
`

**不用 with 的缺点**：忘记 .close() 会导致数据没写入。

---

## 10. matplotlib 画图参数

### plt.savefig

`python
plt.savefig(filepath, dpi=100, bbox_inches="tight")
`

| 参数 | 作用 |
|------|------|
| dpi=100 | 清晰度（100适中，300打印级） |
| box_inches="tight" | 裁剪多余白边 |

### 中文显示

`python
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
`

---

## 11. seaborn 可视化参数

### histplot

`python
sns.histplot(df[column], bins=30, kde=True, color="steelblue")
`

| 参数 | 作用 |
|------|------|
| ins=30 | 柱数 |
| kde=True | 叠加密度曲线（看分布形态） |
| color="steelblue" | 颜色 |

### heatmap

`python
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, square=True, linewidths=0.5)
`

| 参数 | 作用 |
|------|------|
| nnot=True | 格子里显示数字 |
| mt=".2f" | 数字保留2位小数 |
| cmap="RdYlBu_r" | 红-白-蓝配色 |
| center=0 | 0 对应白色 |
| square=True | 格子是正方形 |
| linewidths=0.5 | 格子间有细线 |

### scatterplot

`python
sns.scatterplot(data=df, x=x, y=y, alpha=0.6, color="steelblue")
`

| 参数 | 作用 |
|------|------|
| data=df | 数据来源 |
| lpha=0.6 | 半透明（看出密度） |

### barplot

`python
sns.barplot(data=df, x=x, y=y, palette="Set2", ci="sd")
`

| 参数 | 作用 |
|------|------|
| palette="Set2" | 不同类别不同颜色 |
| ci="sd" | 误差线显示标准差 |

---

## 12. 条件分支

### if/else 判断 by 参数

`python
if by and by in df.columns:
    # 有分组 → 分组箱线图
else:
    # 没分组 → 单变量箱线图
`

y and by in df.columns：
- y 有值（不是 None/空）
- 且 y 这列在数据里

---

## 13. 错误处理

### try/except

`python
try:
    # 可能报错的代码
except:
    # 出错了执行这里
`

### 捕获特定错误

`python
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # 找不到 xgboost
    HAS_XGBOOST = False
`

**效果**：装了 xgboost 就能用，没装也不报错。

---

## 14. 列表操作

### list.remove(值)

`python
feature_cols = ["年龄", "收入", "消费"]
feature_cols.remove("收入")
# → ["年龄", "消费"]
`

从列表中删掉某个元素。

---

## 15. LabelEncoder

### 文字转数字

`python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(["男", "女", "男"])
# → [0, 1, 0]
`

**为什么需要？** sklearn 模型只认数字，不认文字。

---

## 16. 模型保存

### joblib

`python
import joblib

# 保存
joblib.dump(model, "model.joblib")

# 加载
model = joblib.load("model.joblib")
`

### 保存三件套

`
训练后要保存：
1. model.joblib        → 模型
2. scaler.joblib       → 标准化参数
3. label_encoder.joblib → 文字映射（分类任务）
`

少保存一个 → 预测结果出错。

---

## 17. 单例模式

### __new__ 控制创建

`python
class ToolRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._register_all()
        return cls._instance
`

**效果**：无论创建多少次，都返回同一个对象。

`
第1次：创建 → 初始化 → 返回
第2次：直接返回同一个
`

### __init__ vs __new__

| | __new__ | __init__ |
|---|---|---|
| 时机 | 创建前 | 创建后 |
| 作用 | 控制创建什么 | 设置初始状态 |
| 单例 | 需要 | 不需要 |

### Config vs ToolRegistry

| | Config | ToolRegistry |
|---|---|---|
| 模式 | 类属性 | 单例 |
| 需要初始化？ | 不需要 | 需要 |
| 为什么？ | 数据静态 | 要执行 _register_all |

---

## 18. 类里 vs 类外

### 判断标准

`
"这个方法需要访问 self.xxx 吗？"

需要 → 放类里
不需要 → 放类外
`

### 我们的选择

`python
# 放类外：不需要 self
def _build_data_context(df):
    ...

class DataAnalysisAgent:
    def load_data(self, file_path):
        _build_data_context(self.df)  # 调用外部函数
`

---

## 附：常用速查表

### 下划线含义

| 写法 | 含义 |
|------|------|
| _name | 内部使用 |
| __name | 私有（强制） |
| __name__ | Python 魔术方法 |

### 文件模式

| 模式 | 含义 |
|------|------|
| "r" | 读取 |
| "w" | 写入（覆盖） |
| "a" | 追加 |
| "rb" | 读二进制 |

### JSON Schema 类型

| 写法 | Python 类型 |
|------|------------|
| "string" | str |
| "integer" | int |
| "number" | float |
| "array" | list |
| "object" | dict |
| "boolean" | bool |
