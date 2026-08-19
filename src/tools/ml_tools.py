"""
机器学习建模工具
支持回归（线性回归、决策树、随机森林、XGBoost）
支持分类（逻辑回归、决策树、随机森林、XGBoost）
支持聚类（K-Means）
支持模型保存与预测
"""
import os
import uuid
import json
import joblib

# Fix threadpoolctl issue on Windows
import unittest.mock as _mock
import sys as _sys
_mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}).start()
try:
    import threadpoolctl
    _sys.modules['threadpoolctl'] = _mock.MagicMock()
except ImportError:
    pass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.tools.base import BaseTool
from src.config import Config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class MLTool(BaseTool):
    """ML建模基类"""

    def _prepare_data(self, df, target_column, task_type):
        """准备训练数据"""
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in feature_cols:
            feature_cols.remove(target_column)

        X = df[feature_cols].dropna()
        y = df[target_column].loc[X.index]

        le = None
        if task_type == "classification":
            if y.dtype == "object":
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)

        return X, y, feature_cols, le

    def _save_fig(self):
        plt.tight_layout()
        filename = f"{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(Config.OUTPUT_DIR, filename)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()
        return filepath

    def _save_model(self, model, scaler, feature_cols, le, task_type, target):
        """保存模型及相关对象"""
        save_dir = os.path.join(Config.OUTPUT_DIR, "models")
        os.makedirs(save_dir, exist_ok=True)

        # 生成模型ID
        model_id = f"{task_type}_{target}_{uuid.uuid4().hex[:6]}"
        model_dir = os.path.join(save_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))
        # 保存标准化器
        joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
        # 保存特征列名
        with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
            json.dump(feature_cols, f)
        # 保存 LabelEncoder（如果有）
        if le is not None:
            joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))

        # 保存元信息
        meta = {
            "task_type": task_type,
            "target": target,
            "feature_cols": feature_cols,
            "model_type": type(model).__name__,
        }
        with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return model_id, model_dir


class RegressionTool(MLTool):
    """自动回归：训练多个模型并对比"""

    @property
    def name(self):
        return "regression"

    @property
    def description(self):
        return "对目标列进行回归预测，自动训练线性回归、决策树、随机森林、XGBoost四个模型并对比效果"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "要预测的目标列名（数值列）"},
            },
            "required": ["target"],
        }

    def execute(self, df, **kwargs):
        target = kwargs["target"]
        if target not in df.columns:
            return {"result": None, "message": f"列 {target} 不存在", "image": None}

        X, y, feature_cols, le = self._prepare_data(df, target, "regression")
        if len(feature_cols) == 0:
            return {"result": None, "message": "没有可用的特征列", "image": None}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "线性回归": LinearRegression(),
            "决策树": DecisionTreeRegressor(random_state=42),
            "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)

        results = {}
        best_model_name = None
        best_r2 = -999

        for name, model in models.items():
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {"R2": round(r2, 4), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name

        # 保存最佳模型
        best_model = models[best_model_name]
        model_id, model_dir = self._save_model(best_model, scaler, feature_cols, le, "regression", target)

        # 用最佳模型的特征重要性出图
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.figure(figsize=(8, 5))
            plt.title(f"特征重要性 ({best_model_name})")
            plt.barh(range(len(indices)), importances[indices][::-1], color="steelblue")
            plt.yticks(range(len(indices)), [feature_cols[i] for i in indices][::-1])
            plt.xlabel("重要性")
            img_path = self._save_fig()
        else:
            img_path = None

        return {
            "result": results,
            "message": f"回归完成，最佳模型: {best_model_name} (R2={best_r2:.4f})，模型已保存 (ID: {model_id})",
            "image": img_path,
        }


class ClassificationTool(MLTool):
    """自动分类：训练多个模型并对比"""

    @property
    def name(self):
        return "classification"

    @property
    def description(self):
        return "对目标列进行分类预测，自动训练逻辑回归、决策树、随机森林、XGBoost四个模型并对比效果"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "要预测的目标列名"},
            },
            "required": ["target"],
        }

    def execute(self, df, **kwargs):
        target = kwargs["target"]
        if target not in df.columns:
            return {"result": None, "message": f"列 {target} 不存在", "image": None}

        X, y, feature_cols, le = self._prepare_data(df, target, "classification")
        if len(feature_cols) == 0:
            return {"result": None, "message": "没有可用的特征列", "image": None}
        if y.nunique() < 2:
            return {"result": None, "message": "目标列类别数不足2个", "image": None}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
            "决策树": DecisionTreeClassifier(random_state=42),
            "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")

        results = {}
        best_model_name = None
        best_acc = -1

        for name, model in models.items():
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {"准确率": round(acc, 4)}
            if acc > best_acc:
                best_acc = acc
                best_model_name = name

        # 保存最佳模型
        best_model = models[best_model_name]
        model_id, model_dir = self._save_model(best_model, scaler, feature_cols, le, "classification", target)

        # 特征重要性图
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.figure(figsize=(8, 5))
            plt.title(f"特征重要性 ({best_model_name})")
            plt.barh(range(len(indices)), importances[indices][::-1], color="coral")
            plt.yticks(range(len(indices)), [feature_cols[i] for i in indices][::-1])
            plt.xlabel("重要性")
            img_path = self._save_fig()
        else:
            img_path = None

        return {
            "result": results,
            "message": f"分类完成，最佳模型: {best_model_name} (准确率={best_acc:.4f})，模型已保存 (ID: {model_id})",
            "image": img_path,
        }


class ClusteringTool(MLTool):
    """K-Means 聚类"""

    @property
    def name(self):
        return "clustering"

    @property
    def description(self):
        return "对数据进行K-Means聚类，自动分为K个组"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "n_clusters": {"type": "integer", "description": "聚类数量K，默认3"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用于聚类的列，为空则使用所有数值列",
                },
            },
            "required": ["n_clusters"],
        }

    def execute(self, df, **kwargs):
        n_clusters = kwargs["n_clusters"]
        columns = kwargs.get("columns", [])
        if columns:
            X = df[columns].dropna()
        else:
            X = df.select_dtypes(include=[np.number]).dropna()

        if X.empty or len(X.columns) < 2:
            return {"result": None, "message": "需要至少2个数值列", "image": None}

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_s)

        # 保存模型
        model_id, model_dir = self._save_model(kmeans, scaler, X.columns.tolist(), None, "clustering", "kmeans")

        # 聚类结果可视化（用前2个特征）
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(X_s[:, 0], X_s[:, 1], c=labels, cmap="Set2", alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    c="red", marker="X", s=200, label="质心")
        plt.title(f"K-Means 聚类 (K={n_clusters})")
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.legend()
        img_path = self._save_fig()

        # 各簇统计
        X_copy = X.copy()
        X_copy["cluster"] = labels
        cluster_stats = X_copy.groupby("cluster").mean().round(2)

        return {
            "result": {"n_clusters": n_clusters, "cluster_stats": cluster_stats},
            "message": f"聚类完成，分为 {n_clusters} 个簇，模型已保存 (ID: {model_id})",
            "image": img_path,
        }


class PredictTool(BaseTool):
    """使用已保存的模型进行预测"""

    @property
    def name(self):
        return "predict"

    @property
    def description(self):
        return "使用已保存的模型对新数据进行预测。需要提供模型ID和数据。"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "要使用的模型ID"},
            },
            "required": ["model_id"],
        }

    def execute(self, df, **kwargs):
        model_id = kwargs["model_id"]
        model_dir = os.path.join(Config.OUTPUT_DIR, "models", model_id)

        if not os.path.exists(model_dir):
            return {"result": None, "message": f"模型 {model_id} 不存在", "image": None}

        # 加载模型和相关对象
        try:
            model = joblib.load(os.path.join(model_dir, "model.joblib"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            with open(os.path.join(model_dir, "feature_cols.json"), "r") as f:
                feature_cols = json.load(f)
            le = None
            le_path = os.path.join(model_dir, "label_encoder.joblib")
            if os.path.exists(le_path):
                le = joblib.load(le_path)
            with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            return {"result": None, "message": f"加载模型失败: {str(e)}", "image": None}

        # 准备数据
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) != len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            return {"result": None, "message": f"数据缺少特征列: {missing}", "image": None}

        X = df[available_cols].fillna(0)
        X_s = scaler.transform(X)

        # 预测
        predictions = model.predict(X_s)

        # 如果是分类任务且有 LabelEncoder，转回原始标签
        if le is not None:
            predictions_labels = le.inverse_transform(predictions)
            result_df = df.copy()
            result_df["预测值"] = predictions_labels
        else:
            result_df = df.copy()
            result_df["预测值"] = predictions

        return {
            "result": {"predictions": result_df, "task_type": meta["task_type"]},
            "message": f"预测完成，共 {len(df)} 条数据，任务类型: {meta['task_type']}",
            "image": None,
        }
