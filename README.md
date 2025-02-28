# Medicine Redefine

药物重定位项目：寻找已有药物的新用途

## 项目结构

```
medicine_redefine/
├── src/                      # 源代码目录
│   ├── data/                 # 数据处理相关
│   │   ├── adapters/        # 数据适配器
│   │   ├── crawlers/        # 数据爬虫
│   │   └── manager/         # 数据管理
│   ├── docking/             # 分子对接相关
│   │   ├── core/           # 核心对接算法
│   │   └── utils/          # 对接工具函数
│   ├── scoring/             # 评分系统
│   └── visualization/       # 可视化模块
├── tests/                    # 测试目录
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试
├── notebooks/                # Jupyter notebooks
├── docs/                     # 文档
│   ├── api/                # API文档
│   └── guides/             # 使用指南
├── scripts/                  # 工具脚本
├── requirements.txt          # 项目依赖
├── setup.py                 # 安装脚本
├── README.md                # 项目说明
└── LICENSE                  # 许可证
```

## 功能模块

1. 数据处理模块
   - 从多个数据源获取数据
   - 数据清洗和标准化
   - 数据整合和管理

2. 分子对接模块
   - 蛋白质结构预处理
   - 配体准备
   - 分子对接计算
   - 结果分析

3. 评分系统
   - 对接得分计算
   - 多维评分整合
   - 结果排序和筛选

4. 可视化模块
   - 对接结果可视化
   - 数据分析图表
   - 交互式界面

## 安装

1. 克隆仓库
```bash
git clone https://github.com/QLi007/medicine_redefine.git
cd medicine_redefine
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明

详细使用说明请参考 `docs/guides/` 目录下的文档。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 