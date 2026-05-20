# scMAGCL_system: 基于多尺度图神经网络的单细胞智能分析系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.1-red.svg)

## 项目简介

《基于多尺度图神经网络的单细胞智能分析系统》是一个集成了图神经网络和对比学习算法的单细胞RNA测序（scRNA-seq）数据分析平台。该系统提供了一个用户友好的Web界面，用于处理、分析和可视化高通量生物学数据。

核心算法基于多尺度图构造和图对比学习（Graph Contrastive Learning），结合BYOL（Bootstrap Your Own Latent）框架，为单细胞转录组学研究提供强大的特征学习和聚类能力。

## 主要功能

- **用户认证系统**: 安全的登录和注册功能
- **工作流管理**: 创建、管理和跟踪数据分析工作流
- **数据上传**: 支持多种格式的scRNA-seq数据上传
- **实时分析**: 使用图神经网络进行快速特征学习
- **数据浏览**: 交互式浏览分析结果和聚类结果
- **结果导出**: 支持嵌入向量和聚类标签的导出

## 技术栈

### 核心库
- **PyTorch**: 深度学习框架
- **PyTorch Geometric**: 图神经网络库
- **Streamlit**: Web用户界面框架
- **MySQL**: 数据持久化存储
- **Scanpy**: 单细胞分析工具包
- **Scikit-learn**: 机器学习算法

### 关键依赖版本
```
python: 3.8+
torch: 1.8.1
torch-geometric: 2.2.0
scanpy: 1.9.6
scikit-learn: 1.2.2
streamlit: 最新版本
mysql-connector-python: 8.0+
```

## 安装指南

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd scMAGCL-main

# 创建Conda环境
conda create -n scmagcl python=3.8
conda activate scmagcl
```

### 2. 安装依赖

```bash
# 安装PyTorch（根据您的系统选择）
# CUDA版本
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU版本
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# 安装PyTorch Geometric
pip install torch-geometric

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 数据库配置

详见[MYSQL_SETUP.md](./MYSQL_SETUP.md)

```bash
# 初始化MySQL数据库
python -c "from mysql_backend import init_mysql_database; init_mysql_database()"
```

### 4. 配置文件

编辑 `config.py` 调整模型超参数：

```python
config = {
    'graph_head': 5,           # 多头注意力头数
    'phi': 0.45,              # 图阈值
    'gcn_dim': 277,           # GCN隐藏维度
    'mlp_dim': 118,           # MLP维度
    'epochs': 200,            # 训练轮数
    'lr': 0.001,              # 学习率
    'use_byol': True,         # 是否使用BYOL框架
    # ... 更多配置项
}
```

## 使用指南

### 启动Web应用

```bash
# 激活环境
conda activate scmagcl

# 启动Streamlit应用
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。



### 工作流示例

1. **注册/登录**: 在Web界面中创建账户
2. **上传数据**: 在"上传"页面上传单细胞数据文件
3. **创建工作流**: 设置模型参数并启动分析
4. **查看结果**: 在"Explorer"页面查看聚类结果、嵌入向量等
5. **导出结果**: 下载分析结果用于后续分析

## 项目结构

```
scMAGCL-main/
├── scMAGCL.py              # 核心模型定义（图构造、GCN、对比学习）
├── main.py                 # 命令行入口点
├── app.py                  # Streamlit应用主入口和工具函数
├── config.py               # 模型配置参数
├── auth_utils.py           # 用户认证相关函数
├── mysql_backend.py        # MySQL数据库操作
├── utils.py                # 通用工具函数
├── MYSQL_SETUP.md          # MySQL数据库设置说明
├── README.md               # 原始项目说明（论文相关）
├── README_CN.md            # 本文件（中文说明）
├── pages/                  # Streamlit多页面应用
│   ├── 0_Login.py          # 登录页面
│   ├── 0_Register.py       # 注册页面
│   ├── 1_Create_Workflow.py# 工作流创建页面
│   └── 2_Explorer.py       # 结果浏览页面
└── save_file/              # 模型保存目录
```

## 核心算法说明

### 图构造（GraphConstructor）
使用多头注意力机制构造动态图，学习样本之间的相似性关系。

### 多尺度图构造（MultiScaleGraphConstructor）
在不同粒度上构造图，捕捉多层次的细胞间相互作用。

### 图对比学习（Graph Contrastive Learning）
通过最大化相同节点不同增强视图的相似性，学习鲁棒的细胞表征。

### BYOL框架
自监督学习框架，无需负样本对即可学习有效的特征表示。

## 数据格式要求

### 支持的格式
- **H5AD**: Scanpy AnnData对象格式（推荐）
- **H5**: HDF5格式
- **MTX**: Matrix Market格式
- **CSV**: 逗号分隔值格式

### 数据结构
```
基因表达矩阵: (细胞数 × 基因数)
示例: (5000个细胞 × 2000个基因)
```


## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至[230002@stu.hebut.edu.cn]

## 更新日志

### v1.0.0 (2024)
- ✅ 发布初始版本
- ✅ Streamlit Web界面
- ✅ MySQL后端集成
- ✅ 多尺度图构造
- ✅ BYOL框架支持

## 致谢

感谢所有贡献者和测试人员的支持！
