# 测试文档

本目录包含所有测试文件，按功能分类组织。

## 目录结构

```
tests/
├── README.md                 # 本文档
├── integration/              # 集成测试
│   ├── test_agno_main.py    # Agno Agent集成测试
│   ├── test_langgraph_main.py # LangGraph Agent集成测试
│   ├── test_framework.py    # 框架集成测试
│   ├── agno_quick_test.py   # Agno快速测试
│   └── langgraph_quick_test.py # LangGraph快速测试
├── unit/                     # 单元测试
│   ├── test_multi_model.py  # 多模型测试
│   ├── test_missing_keys.py # 缺失密钥测试
│   └── test_all_deepseek_models.py # DeepSeek模型测试
├── http_models/             # HTTP模型专项测试
│   ├── test_http_discovery.py # HTTP模型发现测试
│   ├── test_http_integration_final.py # HTTP集成最终测试
│   └── test_http_models.py  # HTTP模型基础测试
└── demos/                   # 演示程序
    ├── HTTP_MODEL_DEMO_NEW.py # HTTP模型新演示
    ├── http_model_demo.py   # HTTP模型演示
    └── api_key_demo.py      # API密钥演示
```

## 运行测试

### 集成测试

测试完整的Agent功能：

```bash
# 测试Agno Agent (需要agno-agent环境)
conda activate agno-agent
python tests/integration/agno_quick_test.py

# 测试LangGraph Agent (需要langgraph-agent环境) 
conda activate langgraph-agent
python tests/integration/langgraph_quick_test.py
```

### 单元测试

测试特定功能模块：

```bash
# 测试多模型支持
python tests/unit/test_multi_model.py

# 测试缺失API密钥处理
python tests/unit/test_missing_keys.py
```

### HTTP模型测试

测试HTTP模型集成：

```bash
# 测试HTTP模型发现
python tests/http_models/test_http_discovery.py

# 测试HTTP模型集成
python tests/http_models/test_http_integration_final.py
```

### 演示程序

查看功能演示：

```bash
# HTTP模型演示
python tests/demos/http_model_demo.py

# API密钥配置演示
python tests/demos/api_key_demo.py
```

## 测试环境要求

### Agno Agent测试
- 环境: `agno-agent`
- 依赖: agno-agent/requirements.txt

### LangGraph Agent测试  
- 环境: `langgraph-agent`
- 依赖: langgraph-agent/requirements.txt

### HTTP模型测试
- 需要配置环境变量中的HTTP API密钥
- 参考: docs/guides/API_KEY_INTEGRATION_GUIDE.md

## 注意事项

1. **环境隔离**: 不同的Agent需要在对应的conda环境中测试
2. **API密钥**: HTTP模型测试需要有效的API密钥配置
3. **网络连接**: 部分测试需要网络连接访问外部API
4. **超时设置**: HTTP模型测试设置了合理的超时时间

## 故障排除

### 常见问题

1. **模块导入错误**: 确保在正确的conda环境中运行
2. **API密钥错误**: 检查.env文件中的密钥配置
3. **网络超时**: 检查网络连接，HTTP模型可能需要代理
4. **依赖缺失**: 运行 `pip install -r requirements.txt`

### 联系支持

如遇到测试问题，请检查：
1. 环境变量配置
2. 网络连接状态  
3. API密钥有效性
4. 依赖包版本兼容性