import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
工具测试脚本
不依赖 LLM API，直接测试所有工具是否正常工作
"""
import os
import sys
import pandas as pd
import numpy as np
from src.tools.registry import ToolRegistry
from src.config import Config


def create_sample_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "年龄": np.random.randint(18, 65, n),
        "收入": np.random.normal(15000, 5000, n).astype(int),
        "消费": np.random.normal(8000, 3000, n).astype(int),
        "地区": np.random.choice(["北京", "上海", "广州", "深圳"], n),
        "性别": np.random.choice(["男", "女"], n),
        "满意度": np.random.randint(1, 6, n),
    })
    df.loc[np.random.choice(n, 10, replace=False), "收入"] = np.nan
    return df


def test_all_tools():
    registry = ToolRegistry()
    df = create_sample_data()

    print("=" * 50)
    print("  工具测试")
    print("=" * 50)
    print(f"测试数据: {len(df)} 行 x {len(df.columns)} 列")
    print(f"列: {list(df.columns)}")

    tests_passed = 0
    tests_failed = 0
    saved_model_id = None

    tests = [
        ("描述统计", lambda: registry.execute("describe_data", df)),
        ("频次统计", lambda: registry.execute("value_counts", df, column="地区")),
        ("相关性分析", lambda: registry.execute("correlation", df)),
        ("直方图", lambda: registry.execute("plot_histogram", df, column="收入")),
        ("箱线图", lambda: registry.execute("plot_boxplot", df, column="收入", by="地区")),
        ("热力图", lambda: registry.execute("plot_heatmap", df)),
        ("正态性检验", lambda: registry.execute("normality_test", df, column="消费")),
        ("t检验", lambda: registry.execute("t_test", df, column="收入", group_column="性别")),
    ]

    for name, test_fn in tests:
        print(f"\n--- 测试: {name} ---")
        result = test_fn()
        if result.get("result") is not None or result.get("image"):
            print(f"[PASS] {result['message']}")
            tests_passed += 1
        else:
            print(f"[FAIL] {result['message']}")
            tests_failed += 1

    # 回归建模
    print("\n--- 测试: 回归建模 ---")
    result = registry.execute("regression", df, target="消费")
    if result["result"] is not None:
        print(f"[PASS] {result['message']}")
        for model_name, metrics in result["result"].items():
            print(f"   {model_name}: {metrics}")
        msg = result["message"]
        if "ID:" in msg:
            saved_model_id = msg.split("ID: ")[-1].strip(")")
            print(f"   模型ID: {saved_model_id}")
        tests_passed += 1
    else:
        print(f"[FAIL] {result['message']}")
        tests_failed += 1

    # 分类建模
    print("\n--- 测试: 分类建模 ---")
    result = registry.execute("classification", df, target="性别")
    if result["result"] is not None:
        print(f"[PASS] {result['message']}")
        for model_name, metrics in result["result"].items():
            print(f"   {model_name}: {metrics}")
        tests_passed += 1
    else:
        print(f"[FAIL] {result['message']}")
        tests_failed += 1

    # 聚类
    print("\n--- 测试: 聚类 ---")
    result = registry.execute("clustering", df, n_clusters=3)
    if result["result"] is not None:
        print(f"[PASS] {result['message']}")
        tests_passed += 1
    else:
        print(f"[FAIL] {result['message']}")
        tests_failed += 1

    # 预测
    print("\n--- 测试: 模型预测 ---")
    if saved_model_id:
        result = registry.execute("predict", df, model_id=saved_model_id)
        if result["result"] is not None:
            print(f"[PASS] {result['message']}")
            pred_df = result["result"]["predictions"]
            print(f"   预测列: {list(pred_df.columns)}")
            tests_passed += 1
        else:
            print(f"[FAIL] {result['message']}")
            tests_failed += 1
    else:
        print("[SKIP] 没有保存的模型")
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"  测试结果: {tests_passed} 通过, {tests_failed} 失败")
    print("=" * 50)

    return tests_failed == 0


if __name__ == "__main__":
    success = test_all_tools()
    sys.exit(0 if success else 1)
