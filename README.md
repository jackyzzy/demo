# AI Agent框架对比Demo

本项目提供了基于LangGraph、Agno和Parlant三个框架开发的AI agent实现，展示统一的模型配置管理和多云服务商支持能力。

## ✨ 项目亮点

- **统一模型配置管理**: 支持15+模型，6大提供商，统一配置和管理
- **多云服务商支持**: 华为云、阿里云、百度云等多个云服务商的HTTP模型集成
- **三大框架对比**: LangGraph、Agno、Parlant三个主流框架的完整实现
- **统一架构设计**: 通过适配器模式和工厂模式实现框架无关的模型管理
- **开箱即用**: 完善的配置模板、测试用例和文档

## 项目结构

```
comp_agent/
├── langgraph-agent/        # LangGraph版本实现
│   ├── src/
│   │   ├── agents/         # Agent核心类
│   │   ├── states/         # 状态定义
│   │   ├── tools/          # 工具模块
│   │   └── utils/          # 工具函数
│   ├── main.py             # 主程序
│   └── requirements.txt    # 依赖文件
│
├── agno-agent/             # Agno版本实现
│   ├── src/
│   │   ├── agents/         # Agent核心类
│   │   ├── teams/          # 多智能体团队
│   │   ├── tools/          # 工具模块
│   │   ├── utils/          # 工具函数
│   │   └── workflows/      # 工作流
│   ├── main.py             # 主程序
│   └── requirements.txt    # 依赖文件
│
├── parlant-agent/          # Parlant版本实现
│   ├── src/
│   │   ├── agents/         # Agent核心类
│   │   ├── guidelines/     # Guidelines配置
│   │   ├── tools/          # 工具模块
│   │   └── utils/          # 工具函数
│   ├── main.py             # 主程序
│   └── requirements.txt    # 依赖文件
│
├── tests/                  # 统一测试目录
│   ├── integration/        # 集成测试
│   ├── unit/              # 单元测试
│   ├── http_models/       # HTTP模型测试
│   ├── demos/             # 演示程序
│   └── README.md          # 测试文档
│
├── docs/                   # 项目文档
│   ├── guides/            # 使用指南
│   ├── summaries/         # 功能总结
│   └── api/               # API文档
│
├── model_config.py         # 🎯 核心：统一模型配置管理
├── http_model_client.py    # 🔧 HTTP模型客户端基类
├── http_langchain_adapter.py # 🔌 LangGraph HTTP适配器
├── http_agno_adapter.py    # 🔌 Agno HTTP适配器
├── http_parlant_adapter.py # 🔌 Parlant HTTP适配器
├── .env.template           # 📋 环境变量模板
├── .env                    # 🔐 环境变量配置
└── README.md              # 📖 项目说明
```

## 🏗️ 架构设计

本项目采用**统一配置 + 框架适配器**的架构模式：

```
┌─────────────────────────────────────────────────────────┐
│                    Model Config Layer                    │
│  (model_config.py - 统一的模型配置和管理)                  │
│  • ModelProvider: 6大提供商枚举                           │
│  • ModelConfig: 模型配置数据类                            │
│  • HttpModelDiscovery: 动态发现HTTP模型                   │
│  • ModelManager: 统一管理接口                             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    HTTP Client Layer                     │
│  (http_model_client.py - HTTP模型客户端)                  │
│  • HttpModelClient: 统一HTTP调用接口                      │
│  • create_http_client(): 工厂方法                         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌──────────────┬──────────────────┬────────────────────────┐
│   LangGraph  │      Agno        │       Parlant          │
│   Adapter    │      Adapter     │       Adapter          │
│ (http_lang   │  (http_agno      │   (http_parlant        │
│  chain_      │   _adapter.py)   │    _adapter.py)        │
│  adapter.py) │                  │                        │
└──────────────┴──────────────────┴────────────────────────┘
                            ↓
┌──────────────┬──────────────────┬────────────────────────┐
│  LangGraph   │   Agno Agent     │   Parlant Agent        │
│  Agent       │  (316 LOC)       │   (193 LOC)            │
│  (568 LOC)   │  • 三层架构      │   • 简化直接调用        │
│  • 图状态机  │  • 高性能        │   • 轻量级              │
│  • 复杂工作流│  • 多智能体      │   • 易用性              │
└──────────────┴──────────────────┴────────────────────────┘
```

## 环境准备

### 1. API密钥配置

复制 `.env.template` 为 `.env`，然后配置相关API密钥：

```bash
# API密钥配置
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here  
TAVILY_API_KEY=tvly-your-tavily-key-here

# 可选配置
GROQ_API_KEY=gsk-your-groq-key-here

# DeepSeek配置
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# HTTP模型配置 (支持多云服务商)
HTTP_DEEPSEEK_R1_API_KEY_HUAWEI=your-huawei-api-key-here
HTTP_QWEN_API_KEY_ALIBABA=your-alibaba-api-key-here
HTTP_ERNIE_API_KEY_BAIDU=your-baidu-api-key-here

# Agno配置
AGNO_TELEMETRY=false
```

### 2. 虚拟环境配置

#### LangGraph环境
```bash
conda create -n langgraph-agent python=3.11
conda activate langgraph-agent
cd langgraph-agent
pip install -r requirements.txt
```

#### Agno环境
```bash
conda create -n agno-agent python=3.11
conda activate agno-agent
cd agno-agent
pip install -r requirements.txt
```

#### Parlant环境
```bash
conda create -n parlant-agent python=3.11
conda activate parlant-agent
cd parlant-agent
pip install -r requirements.txt
```

## 运行测试

### LangGraph Agent测试
```bash
conda activate langgraph-agent
cd langgraph-agent
python main.py
```

### Agno Agent测试
```bash
conda activate agno-agent
cd agno-agent
python main.py
```

### Parlant Agent测试
```bash
conda activate parlant-agent
cd parlant-agent
python main.py
```

## 功能特性对比

| 功能特性 | LangGraph Agent | Agno Agent | Parlant Agent |
|---------|----------------|------------|---------------|
| 对话功能 | ✅ 支持多轮对话 | ✅ 支持多轮对话 | ✅ 支持多轮对话 |
| 工具调用 | ✅ 灵活的工具系统 | ✅ 丰富的预建工具 | 🔶 系统提示词 |
| 多步推理 | ✅ 复杂的图状推理 | ✅ 内置推理工具 | 🔶 依赖模型能力 |
| 任务规划 | ✅ 动态计划生成 | ✅ 智能体团队协作 | 🔶 简化实现 |
| 搜索集成 | ✅ 多搜索引擎 | ✅ 多搜索引擎 | ✅ 多搜索引擎 |
| HTTP模型 | ✅ 支持多云服务商 | ✅ 支持多云服务商 | ✅ 支持多云服务商 |
| 模型支持 | ✅ 12+模型提供商 | ✅ 23+模型提供商 | ✅ 多模型提供商 |
| 性能 | 🔶 中等 | ✅ 高性能 | ✅ 轻量高效 |
| 学习曲线 | 🔶 较陡峭 | ✅ 相对简单 | ✅ 非常简单 |
| 特色功能 | 图状工作流 | 多智能体团队 | 简化直接调用 |

## 🤖 支持的模型提供商

本项目支持 **15+ 模型**，覆盖 **6 大提供商**：

### 1. OpenAI 模型
| 模型Key | 模型名称 | 特点 |
|---------|----------|------|
| `gpt-4o` | GPT-4o | 最新旗舰模型，多模态 |
| `gpt-4-turbo` | GPT-4 Turbo | 高性能，128K上下文 |
| `gpt-3.5-turbo` | GPT-3.5 Turbo | 快速响应，成本低 |

### 2. Anthropic Claude 模型
| 模型Key | 模型名称 | 特点 |
|---------|----------|------|
| `claude-3.5-sonnet` | Claude 3.5 Sonnet | 最新版本，推理能力强 |
| `claude-3-opus` | Claude 3 Opus | 最强模型，适合复杂任务 |
| `claude-3-haiku` | Claude 3 Haiku | 快速响应，成本优化 |

### 3. DeepSeek 模型
| 模型Key | 模型名称 | 特点 |
|---------|----------|------|
| `deepseek-chat` | DeepSeek Chat | 通用对话模型 |
| `deepseek-coder` | DeepSeek Coder | 代码专用模型 |

### 4. Groq 模型
| 模型Key | 模型名称 | 特点 |
|---------|----------|------|
| `groq-llama-70b` | Llama 3.1 70B | 超快推理速度 |
| `groq-mixtral` | Mixtral 8x7B | MoE架构，高效 |

### 5. Ollama 本地模型
| 模型Key | 模型名称 | 特点 |
|---------|----------|------|
| `ollama-llama3.1` | Llama 3.1 | 本地运行，隐私保护 |
| `ollama-qwen2.5` | Qwen 2.5 | 中文优化 |
| `ollama-deepseek-r1` | DeepSeek R1 | 推理模型本地版 |

### 6. HTTP 云服务商模型 🆕

支持**动态发现**环境变量中的HTTP模型配置：

#### 华为云 ModelArts
```bash
HTTP_DEEPSEEK_R1_API_KEY_HUAWEI=your-huawei-api-key
# 自动发现为: http-deepseek-r1-huawei
```

#### 阿里云 DashScope
```bash
HTTP_QWEN_API_KEY_ALIBABA=your-alibaba-api-key
# 自动发现为: http-qwen-alibaba
```

#### 百度智能云
```bash
HTTP_ERNIE_API_KEY_BAIDU=your-baidu-api-key
# 自动发现为: http-ernie-baidu
```

#### 自定义HTTP模型
```bash
HTTP_YOUR_MODEL_API_KEY_VENDOR=your-api-key
# 自动发现为: http-your-model-vendor
```

**命名规则**: `HTTP_{MODEL_NAME}_API_KEY_{VENDOR}`
- `MODEL_NAME`: 模型名称（如DEEPSEEK_R1、QWEN、ERNIE）
- `VENDOR`: 服务商名称（如HUAWEI、ALIBABA、BAIDU）

## 📚 使用指南

### 快速开始

1. **选择合适的Agent框架**:
   ```bash
   # LangGraph: 适合复杂工作流
   cd langgraph-agent && python main.py

   # Agno: 适合高性能场景
   cd agno-agent && python main.py

   # Parlant: 适合快速原型
   cd parlant-agent && python main.py
   ```

2. **选择模型**: 启动后根据提示选择模型，支持所有配置的模型

3. **运行模式**:
   - **演示模式**: 自动运行预设测试案例
   - **交互模式**: 手动输入问题进行对话

### 框架选择建议

| 场景 | 推荐框架 | 原因 |
|------|----------|------|
| 复杂工作流控制 | LangGraph | 图状态机，精细控制 |
| 高性能低延迟 | Agno | 优化的执行引擎 |
| 快速原型开发 | Parlant | 简化直接调用 |
| 多智能体协作 | Agno | 内置团队机制 |
| 学习AI Agent | Parlant → Agno → LangGraph | 渐进式学习 |

### 代码示例

#### 使用统一模型配置
```python
from model_config import ModelManager

# 获取所有可用模型
available_models = ModelManager.get_models_by_availability()

# 获取模型配置
config = ModelManager.get_model_config("gpt-4o")
print(f"模型: {config.name}, 提供商: {config.provider.value}")

# 检查模型是否可用
if ModelManager.is_model_available("claude-3.5-sonnet"):
    print("Claude模型已配置")
```

#### 使用HTTP模型
```python
from utils.model_factory import create_model_client

# 创建HTTP模型客户端（所有三个Agent框架统一接口）
client = create_model_client("http-deepseek-r1-huawei")

# 调用模型（根据框架自动适配）
response = agent.chat("你好，请介绍一下你自己")
```

## 🔧 核心组件说明

### model_config.py
**作用**: 统一的模型配置和管理中心

**核心功能**:
- `ModelProvider`: 枚举所有支持的模型提供商
- `ModelConfig`: 模型配置数据类，自动加载API密钥
- `HttpModelDiscovery`: 动态发现环境变量中的HTTP模型
- `ModelManager`: 统一管理接口，提供模型查询、验证等功能

**使用场景**: 所有Agent框架在初始化时都通过此模块获取模型配置

### http_model_client.py
**作用**: HTTP模型的统一客户端实现

**核心功能**:
- `HttpModelClient`: 封装HTTP API调用逻辑
- 支持多种云服务商的请求格式适配
- 统一的错误处理和重试机制

**关键特性**: 不会修改base_url路径，直接向配置的URL发送请求

### http_*_adapter.py
**作用**: 框架特定的HTTP模型适配器

**三个适配器**:
- `http_langchain_adapter.py`: 为LangGraph提供ChatOpenAI兼容接口
- `http_agno_adapter.py`: 为Agno提供模型客户端
- `http_parlant_adapter.py`: 为Parlant提供HttpModelClient实例

**设计原则**: 适配器模式，统一HttpModelClient但适配不同框架的接口要求

### model_factory.py (各Agent内)
**作用**: 每个Agent框架的模型工厂

**核心功能**:
```python
def create_model_client(model_key: str) -> Any:
    """根据model_key创建对应的模型客户端

    自动处理:
    - OpenAI模型 → OpenAI客户端
    - Anthropic模型 → Anthropic客户端
    - HTTP模型 → HttpModelClient (通过适配器)
    - 其他提供商 → 对应客户端
    """
```

**统一接口**: 三个Agent的model_factory接口完全一致

## 🐛 故障排查

### 问题1: "ValueError: 未找到API密钥"
**原因**: API密钥未配置或环境变量名称错误

**解决方案**:
1. 检查 `.env` 文件是否存在且配置正确
2. 验证环境变量命名格式（参考 `.env.template`）
3. 重启终端或重新加载环境变量

### 问题2: HTTP模型返回404错误
**原因**: URL路径重复（已在v1.1中修复）

**验证修复**:
```bash
# 确认使用最新版本的 http_parlant_adapter.py
grep "HttpModelClient" parlant-agent/src/utils/model_factory.py
# 应该看到 HTTP 分支返回 HttpModelClient
```

### 问题3: 模型选择后无响应
**原因**: API密钥无效或网络问题

**解决方案**:
1. 验证API密钥是否有效
2. 检查网络连接和防火墙设置
3. 尝试切换到其他可用模型

### 问题4: HTTP模型在LangGraph/Agno可用但Parlant失败
**原因**: Parlant适配器未正确使用HttpModelClient（已在v1.1中修复）

**验证修复**:
```python
# parlant_agent.py 应包含 HTTP 分支
elif self.model_config.provider == ModelProvider.HTTP:
    response = self.client.chat_completion(...)
```

## 📖 详细文档

- [HTTP模型集成指南](docs/guides/HTTP_MODEL_INTEGRATION.md)
- [Parlant HTTP模型修复说明](docs/summaries/PARLANT_HTTP_404_FIX.md)
- [测试用例文档](tests/README.md)
- [API参考文档](docs/api/)

## 🗺️ 开发路线图

### v1.0 ✅
- [x] 统一模型配置管理
- [x] LangGraph、Agno、Parlant三框架实现
- [x] 基础工具集成（搜索、数学、金融）

### v1.1 ✅
- [x] HTTP模型多云服务商支持
- [x] 动态模型发现机制
- [x] Parlant HTTP模型适配修复
- [x] 统一架构文档

### v1.2 (计划中)
- [ ] Web界面支持
- [ ] 流式输出支持
- [ ] 更多工具集成
- [ ] 性能基准测试

### v2.0 (规划中)
- [ ] 向量数据库集成
- [ ] RAG功能支持
- [ ] 多模态能力扩展
- [ ] 生产环境部署方案

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📮 联系方式

- Issues: [GitHub Issues](https://github.com/yourusername/comp_agent/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/comp_agent/discussions)