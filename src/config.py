"""
配置管理模块
负责加载 .env 中的环境变量，提供全局配置
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件（从项目根目录找）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))


class Config:
    """全局配置类"""

    # LLM 配置
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 项目路径
    PROJECT_ROOT: str = _project_root
    OUTPUT_DIR: str = os.path.join(_project_root, "outputs")

    @classmethod
    def validate(cls) -> bool:
        """验证配置是否有效"""
        if not cls.API_KEY:
            print("❌ 错误：未设置 OPENAI_API_KEY")
            print("   请复制 .env.example 为 .env，并填入你的 API Key")
            return False
        return True


