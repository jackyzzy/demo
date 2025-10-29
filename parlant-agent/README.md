# Parlant Agent

基于Parlant项目结构的智能助手实现（简化版本），直接使用模型客户端API。

## 特点

- ✅ **简洁高效**: 直接使用模型客户端，避免复杂的服务器架构
- ✅ **统一接口**: 与agno-agent和langgraph-agent相同的使用方式
- ✅ **统一模型配置**: 与项目统一的model_config.py集成
- ✅ **多模型支持**: OpenAI、Anthropic、DeepSeek、HTTP模型等
- ✅ **中文支持**: 完整的UTF-8中文输入输出
- ✅ **会话管理**: 支持多会话并发，自动管理历史记录

## 目录结构

```
parlant-agent/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── parlant_agent.py      # 核心Agent类
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_tools.py        # Tavily & DuckDuckGo搜索
│   │   ├── math_tools.py          # 数学计算工具
│   │   └── finance_tools.py       # 股票分析工具
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_factory.py       # 模型工厂
│   └── guidelines/                # Guidelines配置目录
│       └── __init__.py
├── tests/
│   ├── integration/
│   └── unit/
├── main.py                        # 主程序入口
├── requirements.txt               # 依赖配置
└── README.md                      # 本文件
```

## 环境准备

### 1. 创建Conda环境

```bash
conda create -n parlant-agent python=3.11
conda activate parlant-agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 演示模式

运行预设的测试案例：

```bash
python main.py
```

然后选择 `1` 进入演示模式。

### 交互模式

与agent进行自由对话：

```bash
python main.py
```

然后选择 `2` 进入交互模式。

## 测试案例

项目包含5个标准测试案例：

1. **简单对话** - 测试基本对话能力
2. **数学计算** - 测试计算器工具调用
3. **信息研究** - 测试搜索引擎集成
4. **股票分析** - 测试金融数据获取
5. **复杂推理** - 测试多步推理能力

## 实现说明

### 简化设计

本实现采用简化设计，直接使用模型客户端API：

```python
# 直接初始化agent
agent = ParlantAgent(model_key="gpt-4o")

# 开始对话
response = agent.chat("你好！请介绍一下你的能力")
```

### 模型客户端集成

支持多种模型提供商的直接集成：

```python
# OpenAI模型
agent = ParlantAgent(model_key="gpt-4o")

# Anthropic模型
agent = ParlantAgent(model_key="claude-3-5-sonnet")

# DeepSeek模型
agent = ParlantAgent(model_key="deepseek-chat")

# HTTP自定义模型
agent = ParlantAgent(model_key="http-deepseek-r1-huawei")
```

### 会话管理

支持多个独立会话，自动管理历史记录：

```python
# 不同的会话ID
response1 = agent.chat("Hello", session_id="user1")
response2 = agent.chat("你好", session_id="user2")

# 获取会话历史
history = agent.get_chat_history("user1")

# 清除会话历史
agent.clear_history("user1")
```

## 模型支持

通过统一的model_config.py支持：

- OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5-turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)
- DeepSeek (DeepSeek Chat, DeepSeek Coder)
- HTTP模型 (华为云、阿里云、百度云等)
- Groq (Llama 3, Mixtral, Gemma)

## 与其他实现对比

| 特性 | Parlant Agent | LangGraph Agent | Agno Agent |
|-----|---------|-----------|------|
| 实现方式 | 直接模型调用 | 图状态机 | Agent类 |
| 异步支持 | 🔶 可选 | 🔶 部分支持 | ✅ 支持 |
| 工具定义 | 系统提示词 | 类/函数 | 类继承 |
| 行为定义 | 系统提示词 | 图节点 | Workflows |
| 学习曲线 | 低 | 高 | 中 |
| 接口统一性 | ✅ 统一 | ✅ 统一 | ✅ 统一 |

## 注意事项

1. 本实现为简化版本，未使用完整Parlant服务器架构
2. 适合快速原型开发和小规模应用
3. 如需完整Parlant功能（Guidelines、Journey等），请参考Parlant官方文档
4. 模型调用需要配置相应的API密钥

## 参考资源

- [Parlant官方文档](https://parlant.io/)
- [Parlant GitHub](https://github.com/emcie-co/parlant)
- [项目model_config.py](../model_config.py)
