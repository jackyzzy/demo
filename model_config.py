"""
多模型配置系统
支持多种AI模型提供商的统一配置和管理
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class ModelProvider(Enum):
    """模型提供商枚举"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    DEEPSEEK = "deepseek"
    HTTP = "http"


@dataclass
class ModelConfig:
    """模型配置数据类"""

    name: str  # 模型名称
    provider: ModelProvider  # 提供商
    model_id: str  # 模型ID
    api_key_env: str  # API密钥环境变量名
    base_url: Optional[str] = None  # 基础URL（可选）
    max_tokens: int = 4096  # 最大tokens
    temperature: float = 0.1  # 温度参数
    description: str = ""  # 描述
    supports_streaming: bool = True  # 是否支持流式输出
    supports_function_calling: bool = True  # 是否支持函数调用
    # HTTP模型特定配置
    vendor: Optional[str] = None  # HTTP模型供应商名称（如：huawei, alibaba等）
    headers: Optional[Dict[str, str]] = None  # 额外的HTTP头部
    _api_key: Optional[str] = field(default=None, init=False)  # 缓存的API密钥

    def __post_init__(self):
        """初始化后自动加载API密钥"""
        self._load_api_key()

    def _load_api_key(self):
        """从环境变量加载API密钥"""
        if self.api_key_env:
            self._api_key = os.getenv(self.api_key_env)

    @property
    def api_key(self) -> Optional[str]:
        """获取API密钥"""
        if self._api_key is None and self.api_key_env:
            self._load_api_key()
        return self._api_key

    @property
    def is_available(self) -> bool:
        """检查模型是否可用（有API密钥或不需要API密钥）"""
        # Ollama本地模型不需要API密钥
        if self.provider == ModelProvider.OLLAMA:
            return True

        # 其他模型需要API密钥
        if not self.api_key_env:
            # 如果没有定义API密钥环境变量名，则认为不需要API密钥
            return True

        # 检查API密钥是否存在且格式正确
        if not self.api_key or not self.api_key.strip():
            return False

        # 验证API密钥格式
        return self._validate_api_key_format()

    def _validate_api_key_format(self) -> bool:
        """验证API密钥格式是否正确"""
        if not self.api_key:
            return False

        api_key = self.api_key.strip()

        # 检查是否是占位符
        placeholder_patterns = [
            "sk-your-openai-key-here",
            "sk-your-api-key-here",
            "your-api-key-here",
            "gsk-your-groq-key-here",
            "hf_your-huggingface-key-here",
            "tvly-your-tavily-key-here",
            "sk-your-***********here",
            "xxxxxx",
        ]
        if api_key in placeholder_patterns:
            return False

        # 根据提供商验证API密钥格式
        if self.provider == ModelProvider.OPENAI:
            # OpenAI API密钥通常以sk-开头，长度约为51个字符
            return api_key.startswith("sk-") and len(api_key) > 20

        elif self.provider == ModelProvider.ANTHROPIC:
            # Anthropic API密钥通常以sk-ant-开头
            return api_key.startswith("sk-ant-") and len(api_key) > 20

        elif self.provider == ModelProvider.GROQ:
            # Groq API密钥通常以gsk-开头
            return api_key.startswith("gsk-") and len(api_key) > 20

        elif self.provider == ModelProvider.DEEPSEEK:
            # DeepSeek API密钥通常以sk-开头，但自定义部署可能有不同格式
            return (api_key.startswith("sk-") and len(api_key) > 20) or (
                len(api_key) > 30
                and not any(
                    ph in api_key.lower() for ph in ["your", "key", "here", "xxx"]
                )
            )

        elif self.provider == ModelProvider.HUGGINGFACE:
            # HuggingFace API密钥通常以hf_开头
            return api_key.startswith("hf_") and len(api_key) > 20

        elif self.provider == ModelProvider.HTTP:
            # HTTP模型支持多种API密钥格式，进行宽松验证
            return len(api_key) > 10 and not any(
                ph in api_key.lower() for ph in ["your", "key", "here", "xxx"]
            )

        else:
            # 对于其他提供商，只检查密钥不为空且长度合理
            return len(api_key) > 10

    def reload_api_key(self):
        """重新加载API密钥（用于运行时更新）"""
        self._load_api_key()

    def get_model_info(self) -> Dict[str, str]:
        """获取模型信息摘要"""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "model_id": self.model_id,
            "description": self.description,
            "available": "✅" if self.is_available else "❌",
            "api_key_status": (
                "已配置" if self.api_key else "未配置" if self.api_key_env else "不需要"
            ),
        }


# 预定义的模型配置
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # OpenAI 模型
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        description="OpenAI最新的GPT-4o模型，性能卓越",
    ),
    "gpt-4-turbo": ModelConfig(
        name="GPT-4 Turbo",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4-turbo-preview",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        description="OpenAI GPT-4 Turbo模型，速度更快",
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        provider=ModelProvider.OPENAI,
        model_id="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        description="OpenAI GPT-3.5 Turbo，经济实用",
    ),
    # Anthropic 模型
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4096,
        description="Anthropic最新的Claude 3.5 Sonnet模型",
    ),
    "claude-3-opus": ModelConfig(
        name="Claude 3 Opus",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-opus-20240229",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4096,
        description="Anthropic最强大的Claude 3 Opus模型",
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4096,
        description="Anthropic速度最快的Claude 3 Haiku模型",
    ),
    # Groq 模型
    "llama3-70b": ModelConfig(
        name="Llama 3 70B",
        provider=ModelProvider.GROQ,
        model_id="llama3-70b-8192",
        api_key_env="GROQ_API_KEY",
        max_tokens=8192,
        description="Meta Llama 3 70B，高性能开源模型",
    ),
    "mixtral-8x7b": ModelConfig(
        name="Mixtral 8x7B",
        provider=ModelProvider.GROQ,
        model_id="mixtral-8x7b-32768",
        api_key_env="GROQ_API_KEY",
        max_tokens=32768,
        description="Mistral AI Mixtral 8x7B专家混合模型",
    ),
    "gemma-7b": ModelConfig(
        name="Gemma 7B",
        provider=ModelProvider.GROQ,
        model_id="gemma-7b-it",
        api_key_env="GROQ_API_KEY",
        max_tokens=8192,
        description="Google Gemma 7B指令调优模型",
    ),
    # DeepSeek 模型
    "deepseek-chat": ModelConfig(
        name="DeepSeek Chat",
        provider=ModelProvider.DEEPSEEK,
        model_id="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        max_tokens=4096,
        description="DeepSeek 对话模型，性能优异",
    ),
    "deepseek-coder": ModelConfig(
        name="DeepSeek Coder",
        provider=ModelProvider.DEEPSEEK,
        model_id="deepseek-coder",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        max_tokens=4096,
        description="DeepSeek 代码生成专用模型",
    ),
    "deepseek-r1-huawei": ModelConfig(
        name="DeepSeek-R1 huawei",
        provider=ModelProvider.DEEPSEEK,
        model_id="DeepSeek-R1",
        api_key_env="DEEPSEEK_R1_API_KEY_HUAWEI",
        base_url="https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/8a062fd4-7367-4ab4-a936-5eeb8fb821c4/v1",
        max_tokens=4096,
        temperature=1.0,
        description="自定义部署的DeepSeek-R1模型，推理能力强",
    ),
    # Ollama本地模型（无需API密钥）
    "llama2": ModelConfig(
        name="Llama 2 Local",
        provider=ModelProvider.OLLAMA,
        model_id="llama2",
        api_key_env="",
        base_url="http://localhost:11434",
        max_tokens=4096,
        description="本地运行的Llama 2模型",
    ),
    "codellama": ModelConfig(
        name="Code Llama Local",
        provider=ModelProvider.OLLAMA,
        model_id="codellama",
        api_key_env="",
        base_url="http://localhost:11434",
        max_tokens=4096,
        description="本地运行的Code Llama编程专用模型",
    ),
}


class HttpModelDiscovery:
    """HTTP模型动态发现器"""

    # HTTP模型配置模板
    HTTP_MODEL_TEMPLATES = {
        # 华为云模型配置
        "HTTP_DEEPSEEK_R1_API_KEY_HUAWEI": {
            "name": "DeepSeek-R1 (华为云)",
            "model_id": "DeepSeek-R1",
            "base_url": "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/8a062fd4-7367-4ab4-a936-5eeb8fb821c4/v1/chat/completions",
            "vendor": "huawei",
            "temperature": 1.0,
            "description": "华为云部署的DeepSeek-R1模型",
        },
        "HTTP_QWEN_API_KEY_HUAWEI": {
            "name": "通义千问 (华为云)",
            "model_id": "Qwen2.5-72B-Instruct",
            "base_url": "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/qwen-endpoint/v1/chat/completions",
            "vendor": "huawei",
            "temperature": 0.7,
            "description": "华为云部署的通义千问模型",
        },
        # 阿里云模型配置
        "HTTP_QWEN_API_KEY_ALIBABA": {
            "name": "通义千问 (阿里云)",
            "model_id": "qwen-plus",
            "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "vendor": "alibaba",
            "temperature": 0.7,
            "description": "阿里云通义千问模型",
        },
        # 百度云模型配置
        "HTTP_ERNIE_API_KEY_BAIDU": {
            "name": "文心一言 (百度云)",
            "model_id": "ernie-bot-turbo",
            "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
            "vendor": "baidu",
            "temperature": 0.8,
            "description": "百度云文心一言模型",
        },
        # 通用OpenAI兼容模型
        "HTTP_OPENAI_COMPATIBLE_API_KEY": {
            "name": "OpenAI兼容模型",
            "model_id": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "vendor": "openai-compatible",
            "temperature": 0.7,
            "description": "通用OpenAI兼容HTTP API模型",
        },
    }

    @classmethod
    def discover_http_models(cls) -> Dict[str, ModelConfig]:
        """动态发现环境中配置的HTTP模型"""
        discovered_models = {}

        # 扫描环境变量，寻找HTTP_开头的API密钥
        for env_key, api_key in os.environ.items():
            if (
                env_key.startswith("HTTP_")
                and env_key.endswith("_API_KEY")
                or env_key.startswith("HTTP_")
                and "_API_KEY_" in env_key
            ):
                if env_key in cls.HTTP_MODEL_TEMPLATES:
                    template = cls.HTTP_MODEL_TEMPLATES[env_key]

                    # 生成模型key（小写，用连字符）
                    model_key = (
                        env_key.lower().replace("_api_key", "").replace("_", "-")
                    )

                    # 创建ModelConfig实例
                    model_config = ModelConfig(
                        name=template["name"],
                        provider=ModelProvider.HTTP,
                        model_id=template["model_id"],
                        api_key_env=env_key,
                        base_url=template["base_url"],
                        max_tokens=4096,
                        temperature=template["temperature"],
                        vendor=template["vendor"],
                        headers={"Content-Type": "application/json"},
                        description=template["description"],
                    )

                    discovered_models[model_key] = model_config

        return discovered_models

    @classmethod
    def get_supported_http_models(cls) -> Dict[str, dict]:
        """获取支持的HTTP模型列表（不论是否已配置）"""
        return cls.HTTP_MODEL_TEMPLATES.copy()


class ModelManager:
    """模型管理器"""

    @staticmethod
    def get_available_models() -> Dict[str, ModelConfig]:
        """获取所有可用的模型配置（包括动态发现的HTTP模型）"""
        models = MODEL_CONFIGS.copy()

        # 添加动态发现的HTTP模型
        http_models = HttpModelDiscovery.discover_http_models()
        models.update(http_models)

        return models

    @staticmethod
    def get_model_by_provider(provider: ModelProvider) -> Dict[str, ModelConfig]:
        """根据提供商筛选模型"""
        return {
            key: config
            for key, config in MODEL_CONFIGS.items()
            if config.provider == provider
        }

    @staticmethod
    def get_model_config(model_key: str) -> Optional[ModelConfig]:
        """获取指定模型的配置（包括动态发现的HTTP模型）"""
        # 首先检查预定义模型
        config = MODEL_CONFIGS.get(model_key)
        if config:
            return config

        # 检查动态发现的HTTP模型
        http_models = HttpModelDiscovery.discover_http_models()
        return http_models.get(model_key)

    @staticmethod
    def list_available_models() -> List[str]:
        """列出所有可用模型的键值"""
        return list(MODEL_CONFIGS.keys())

    @staticmethod
    def get_models_by_availability() -> Dict[str, ModelConfig]:
        """获取当前环境下可用的模型（检查API密钥）"""
        available = {}

        # 检查预定义模型
        for key, config in MODEL_CONFIGS.items():
            if config.is_available:
                available[key] = config

        # 检查动态发现的HTTP模型
        http_models = HttpModelDiscovery.discover_http_models()
        for key, config in http_models.items():
            if config.is_available:
                available[key] = config

        return available

    @staticmethod
    def add_custom_model(key: str, config: ModelConfig):
        """添加自定义模型配置"""
        MODEL_CONFIGS[key] = config

    @staticmethod
    def reload_all_api_keys():
        """重新加载所有模型的API密钥"""
        for config in MODEL_CONFIGS.values():
            config.reload_api_key()

    @staticmethod
    def get_missing_api_keys() -> Dict[str, List[str]]:
        """获取缺失的API密钥信息"""
        missing = {}
        for key, config in MODEL_CONFIGS.items():
            if not config.is_available and config.api_key_env:
                provider = config.provider.value
                if provider not in missing:
                    missing[provider] = []
                missing[provider].append(
                    {
                        "model_key": key,
                        "env_var": config.api_key_env,
                        "model_name": config.name,
                    }
                )
        return missing

    @staticmethod
    def validate_environment() -> Dict[str, any]:
        """验证环境配置"""
        all_models = ModelManager.get_available_models()
        available_models = ModelManager.get_models_by_availability()
        missing_keys = ModelManager.get_missing_api_keys()

        return {
            "total_models": len(all_models),
            "available_models": len(available_models),
            "missing_keys": missing_keys,
            "availability_rate": (
                len(available_models) / len(all_models) if all_models else 0
            ),
        }

    @staticmethod
    def is_model_available(model_key: str) -> bool:
        """检查指定模型是否可用（包括动态发现的HTTP模型）"""
        # 获取模型配置（包括HTTP模型）
        config = ModelManager.get_model_config(model_key)
        if not config:
            return False

        # 检查模型的API密钥是否正确配置
        return config.is_available

    @staticmethod
    def get_model_availability_status(model_key: str) -> Dict[str, any]:
        """获取模型可用性的详细状态信息"""
        status = {
            "model_key": model_key,
            "is_available": False,
            "exists_in_config": False,
            "has_api_key": False,
            "api_key_valid_format": False,
            "provider": None,
            "issues": [],
        }

        # 检查条件1: 模型是否在预定义配置中
        config = MODEL_CONFIGS.get(model_key)
        if not config:
            status["issues"].append("模型未在预定义配置中找到")
            return status

        status["exists_in_config"] = True
        status["provider"] = config.provider.value

        # 对于不需要API密钥的模型（如Ollama）
        if config.provider == ModelProvider.OLLAMA:
            status["is_available"] = True
            status["has_api_key"] = True  # 不需要API密钥
            status["api_key_valid_format"] = True
            return status

        # 检查条件2: API密钥配置
        if not config.api_key_env:
            status["issues"].append("未定义API密钥环境变量")
            return status

        if not config.api_key or not config.api_key.strip():
            status["issues"].append(
                f"API密钥未配置，请设置环境变量: {config.api_key_env}"
            )
            return status

        status["has_api_key"] = True

        # 检查API密钥格式
        if not config._validate_api_key_format():
            status["issues"].append("API密钥格式不正确或为占位符")
            return status

        status["api_key_valid_format"] = True
        status["is_available"] = True

        return status


def display_available_models():
    """显示当前可用的模型"""
    all_models = ModelManager.get_available_models()
    available_models = ModelManager.get_models_by_availability()

    print("🤖 模型配置状态：")
    print("=" * 80)

    # 按提供商分组显示
    providers = {}
    for key, config in all_models.items():
        provider = config.provider.value
        if provider not in providers:
            providers[provider] = []
        providers[provider].append((key, config))

    for provider, models in providers.items():
        print(f"\n📡 {provider.upper()}:")
        for key, config in models:
            info = config.get_model_info()
            status_icon = info["available"]
            print(f"  {status_icon} {key}: {config.name}")
            print(f"    📝 {config.description}")
            print(f"    🔑 API密钥: {info['api_key_status']}")
            if not config.is_available and config.api_key_env:
                print(f"    ⚠️  请设置环境变量: {config.api_key_env}")

    print("\n" + "=" * 80)
    print(f"📊 统计: {len(available_models)}/{len(all_models)} 个模型可用")

    if not available_models:
        print("\n❌ 当前没有可用的模型，请配置API密钥后重试")
        print("💡 提示: 编辑 .env 文件，添加相应的API密钥")


def display_detailed_model_info():
    """显示详细的模型配置信息（调试用）"""
    print("🔍 详细模型配置信息：")
    print("=" * 100)

    for key, config in MODEL_CONFIGS.items():
        info = config.get_model_info()
        print(f"\n🤖 {key}:")
        print(f"  名称: {info['name']}")
        print(f"  提供商: {info['provider']}")
        print(f"  模型ID: {config.model_id}")
        print(f"  环境变量: {config.api_key_env}")
        print(f"  API密钥状态: {info['api_key_status']}")
        print(f"  可用状态: {info['available']}")
        if config.base_url:
            print(f"  基础URL: {config.base_url}")
        print(f"  描述: {info['description']}")

    print("\n" + "=" * 100)


def display_environment_setup_guide():
    """显示环境设置指南"""
    validation = ModelManager.validate_environment()
    missing_keys = validation["missing_keys"]

    print("🔧 环境配置指南")
    print("=" * 60)

    print(
        f"📊 模型可用性: {validation['available_models']}/{validation['total_models']} "
        f"({validation['availability_rate']:.1%})"
    )

    if not missing_keys:
        print("✅ 所有需要的API密钥都已配置！")
        return

    print("\n❌ 缺失的API密钥配置：")

    for provider, models in missing_keys.items():
        print(f"\n📡 {provider.upper()}:")
        env_vars = set(model["env_var"] for model in models)
        for env_var in env_vars:
            print(f"  🔑 {env_var}")
            affected_models = [
                m["model_name"] for m in models if m["env_var"] == env_var
            ]
            print(f"    影响模型: {', '.join(affected_models)}")

    print("\n💡 配置步骤：")
    print("1. 编辑项目根目录的 .env 文件")
    print("2. 添加或更新相应的API密钥：")

    for provider, models in missing_keys.items():
        env_vars = set(model["env_var"] for model in models)
        for env_var in env_vars:
            print(f"   {env_var}=your-api-key-here")

    print("3. 重新运行程序")
    print("\n📝 获取API密钥的地址：")
    print("   - OpenAI: https://platform.openai.com/api-keys")
    print("   - Anthropic: https://console.anthropic.com/")
    print("   - Groq: https://console.groq.com/keys")
    print("   - DeepSeek: https://platform.deepseek.com/")
    print("   - HuggingFace: https://huggingface.co/settings/tokens")

    print("\n" + "=" * 60)


def quick_setup_check():
    """快速设置检查"""
    validation = ModelManager.validate_environment()

    if validation["availability_rate"] >= 1.0:
        print("✅ 环境配置完善！所有模型都可用")
        return True
    elif validation["available_models"] > 0:
        print(
            f"⚠️  部分模型可用 ({validation['available_models']}/{validation['total_models']})"
        )
        return True
    else:
        print("❌ 没有可用的模型，请配置API密钥")
        display_environment_setup_guide()
        return False


if __name__ == "__main__":
    # 测试模型管理器
    display_available_models()
