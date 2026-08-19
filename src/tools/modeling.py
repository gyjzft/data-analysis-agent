"""
沙箱主导的建模模块（方案 X）

固定但自适应的建模流程，不依赖 LLM 写建模代码：
1. 自动识别特征列（数值 + 类别，排除高基数列）
2. ColumnTransformer 打包完整预处理（标准化 + 独热编码）
3. 训练多个模型对比，选最优
4. 完整保存：model + preprocessor + label_encoder + meta
5. 支持加载模型后对新输入做预测

核心优势：保存的是"完整预处理流程"，预测时 transform 逻辑和训练完全一致。
"""
import os
import json
import uuid

import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, accuracy_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config import Config


# 排除 ID 列的比例阈值：唯一值数量 >= 行数的 80% 视为 ID 列
ID_COLUMN_RATIO = 0.8


def _select_features(df: pd.DataFrame, target: str) -> tuple:
    """自动选择特征列，返回 (numeric_cols, categorical_cols)"""
    numeric_cols = []
    categorical_cols = []
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        nunique = df[col].nunique()
        # 高基数列（每行都不同，如 name）视为 ID 列，不参与建模
        if nunique >= n * ID_COLUMN_RATIO:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _make_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """构造完整预处理流程：缺失值填充 + 数值列标准化 + 类别列独热编码"""
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))
    return ColumnTransformer(transformers)


def _build_model_choices(task_type: str) -> dict:
    """返回候选模型字典"""
    if task_type == "regression":
        models = {
            "线性回归": LinearRegression(),
            "决策树": DecisionTreeRegressor(random_state=42),
            "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)
        return models
    else:
        models = {
            "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
            "决策树": DecisionTreeClassifier(random_state=42),
            "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=42,
                use_label_encoder=False, eval_metric="logloss",
            )
        return models


def build_model(df: pd.DataFrame, target: str, task_type: str = "regression") -> dict:
    """
    构建并保存模型。

    参数:
        df: 数据
        target: 目标列名
        task_type: "regression" 或 "classification"

    返回:
        {
            "model_id": "regression_salary_abc123",
            "best_model": "随机森林",
            "results": {"线性回归": 0.71, ...},
            "metrics_detail": {...},
            "feature_cols": [...],
        }
    """
    if target not in df.columns:
        raise ValueError(f"目标列 '{target}' 不存在")

    task_type = "classification" if task_type != "regression" else "regression"

    # 特征选择
    numeric_cols, categorical_cols = _select_features(df, target)
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        raise ValueError("没有可用的特征列")

    X = df[feature_cols]
    y = df[target]

    # 丢弃目标列为空的行（X 的缺失由 preprocessor 填充）
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    if len(X) == 0:
        raise ValueError("目标列没有有效数据")

    # 分类目标：文字标签转数字
    le = None
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    # 预处理 + 划分
    preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # 训练对比
    models = _build_model_choices(task_type)
    results = {}
    best_model = None
    best_name = None
    best_score = -1e18

    for name, model in models.items():
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        if task_type == "regression":
            score = r2_score(y_test, y_pred)
            results[name] = {
                "R2": round(score, 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
                "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            }
        else:
            score = accuracy_score(y_test, y_pred)
            results[name] = {"准确率": round(score, 4)}
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    # 保存整套产物
    model_id = f"{task_type}_{target}_{uuid.uuid4().hex[:6]}"
    save_dir = os.path.join(Config.OUTPUT_DIR, "models", model_id)
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(save_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.joblib"))
    if le is not None:
        joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))

    # meta.json：记录前端生成输入框所需的特征信息
    meta = {
        "model_id": model_id,
        "task_type": task_type,
        "target": target,
        "best_model": best_name,
        "feature_cols": feature_cols,
        "features": [
            {
                "name": col,
                "type": "numeric" if col in numeric_cols else "categorical",
                "categories": sorted(df[col].dropna().unique().tolist())
                if col in categorical_cols else None,
            }
            for col in feature_cols
        ],
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "model_id": model_id,
        "best_model": best_name,
        "results": results,
        "feature_cols": feature_cols,
    }


def load_model_artifacts(model_id: str) -> dict:
    """加载已保存的模型产物"""
    model_dir = os.path.join(Config.OUTPUT_DIR, "models", model_id)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型 {model_id} 不存在")

    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    artifacts = {
        "model": joblib.load(os.path.join(model_dir, "model.joblib")),
        "preprocessor": joblib.load(os.path.join(model_dir, "preprocessor.joblib")),
        "meta": meta,
        "label_encoder": None,
    }
    le_path = os.path.join(model_dir, "label_encoder.joblib")
    if os.path.exists(le_path):
        artifacts["label_encoder"] = joblib.load(le_path)
    return artifacts


def predict_from_values(model_id: str, values: dict):
    """
    用已保存的模型对一组输入值做预测。

    参数:
        model_id: 模型 ID
        values: {"age": 30, "department": "销售部", ...}

    返回:
        (预测结果, meta)
    """
    artifacts = load_model_artifacts(model_id)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    le = artifacts["label_encoder"]
    meta = artifacts["meta"]

    feature_cols = meta["feature_cols"]
    row = pd.DataFrame([{col: values.get(col) for col in feature_cols}])

    X_p = preprocessor.transform(row)
    pred = model.predict(X_p)

    if le is not None:
        pred = le.inverse_transform(pred)

    return pred[0], meta
