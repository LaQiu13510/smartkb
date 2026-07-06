# Python 项目开发规范 v2.0

## 代码风格

遵循 PEP 8 规范，使用以下工具:
- **Ruff**: 代码格式化和 Lint (替代 Flake8 + isort + Black)
- **mypy**: 静态类型检查
- **pytest**: 单元测试框架

## 项目结构

```
project/
├── src/               # 源代码
│   ├── __init__.py
│   ├── models/        # 数据模型
│   ├── services/      # 业务逻辑
│   ├── api/           # API 路由
│   └── utils/         # 工具函数
├── tests/             # 测试代码
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docs/              # 文档
├── scripts/           # 脚本
├── pyproject.toml     # 项目配置
└── README.md
```

## 依赖管理

使用 `pyproject.toml` 管理依赖:
```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "langchain>=0.3.0",
    "langgraph>=0.2.0",
    "pymilvus>=2.4.0",
]
```

## 测试规范

- 单元测试覆盖率 > 80%
- 集成测试覆盖核心 API
- E2E 测试覆盖关键用户流程
- 使用 fixtures 管理测试数据

## Git 工作流

1. `main`: 生产环境分支
2. `develop`: 开发分支
3. `feature/*`: 功能分支
4. `fix/*`: 修复分支

Commit 规范 (Conventional Commits):
- `feat: 添加新功能`
- `fix: 修复某个Bug`
- `docs: 更新文档`
- `refactor: 重构代码`
