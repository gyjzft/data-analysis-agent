"""
主入口 - 命令行版本（用于快速测试）
测试通过后可以运行 main_gradio.py 启动 Web 界面
"""
import os
import sys
from src.config import Config
from src.agent.core import DataAnalysisAgent


def main():
    # 验证配置
    if not Config.validate():
        sys.exit(1)

    print("=" * 50)
    print("  数据分析 AutoAgent - 命令行版")
    print("=" * 50)
    print()
    print("提示: 先输入数据文件路径，然后输入分析指令")
    print("     输入 quit 退出")
    print()

    agent = DataAnalysisAgent()

    # 加载数据
    while True:
        file_path = input("📁 请输入数据文件路径: ").strip()
        if file_path.lower() == "quit":
            break
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
        result = agent.load_data(file_path)
        print(result)
        break

    # 对话循环
    while True:
        user_input = input("\n📝 你的指令: ").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break
        if not user_input:
            continue

        response = agent.chat(user_input)
        print(f"\n🤖 Agent: {response['text']}")
        if response.get("image"):
            print(f"   📊 图片已保存: {response['image']}")


if __name__ == "__main__":
    main()
