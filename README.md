# scSimGCL - 单细胞RNA-seq数据智能分析系统

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

## 📋 项目概述

**scSimGCL** 是一个基于图对比学习（Graph Contrastive Learning）的单细胞RNA-seq数据分析系统，集成了先进的深度学习模型和交互式Web应用界面。该系统提供了从数据上传、模型训练到结果可视化的完整工作流，是生物信息学数据分析的多功能基础平台。

### 核心技术
- **图对比学习** (scSimGCL/scMAGCL)：利用多尺度图构造和对比学习进行细胞表示学习
- **Web框架**：Streamlit 快速构建交互式数据分析应用
- **数据持久化**：MySQL 数据库支持用户和工作流管理
- **深度学习框架**：PyTorch + PyTorch Geometric

---

## ✨ 主要功能

| 功能模块 | 描述 |
|---------|------|
| **用户认证** | 安全的用户注册与登录系统（支持本地和MySQL存储） |
| **工作流管理** | 创建、查看、管理多个数据分析任务 |
| **数据上传** | 支持多种单细胞数据格式，自动管理文件生命周期 |
| **模型训练** | 可配置的图对比学习模型训练 |
| **结果分析** | 聚类评估、表示学习可视化 |
| **数据探索** | 交互式数据浏览和结果查看 |

---

## 🔧 系统要求

### Python环境
- **Python**: 3.8.17+
- **Conda**: 推荐使用 Anaconda/Miniconda

### 依赖库
```
scanpy==1.9.6          # 单细胞数据处理
scikit-learn==1.2.2    # 机器学习工具
torch==1.8.1           # 深度学习框架
torch-geometric==2.2.0 # 图神经网络
streamlit>=1.0         # Web应用框架
pymysql>=1.0           # MySQL连接（可选）
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
```

### 系统要求
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议 16GB+（用于大规模数据处理）
- **显存**: 推荐 CUDA 显卡（4GB+）用于加速训练

### 可选依赖
- **MySQL Server**: 用于数据持久化（推荐版本 5.7+）

---

## 📦 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd scMAGCL_system/scSimGCL-main
```

### 2. 创建Python虚拟环境
```bash
# 使用Conda
conda create -n scMAGCL python=3.8.17 -y
conda activate scMAGCL
```

### 3. 安装依赖包
```bash
# 基础依赖
pip install scanpy==1.9.6 scikit-learn==1.2.2 numpy pandas scipy

# PyTorch (CPU版本)
pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# 或 PyTorch (CUDA 11.1版本)
# pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# PyTorch Geometric
pip install torch-geometric==2.2.0

# Web应用框架
pip install streamlit streamlit-session-state

# 数据库支持 (可选)
pip install pymysql
```

### 4. 配置MySQL (可选)

#### 4.1 在Navicat中创建数据库
```sql
CREATE DATABASE IF NOT EXISTS scMSDCL
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_general_ci;
```

#### 4.2 设置环境变量 (Windows PowerShell)
```powershell
# 使用连接URL（推荐）
$env:MYSQL_URL="mysql+pymysql://root:密码@localhost:3306/scMSDCL?charset=utf8mb4"
$env:MYSQL_ENABLED="1"

# 或使用环境变量参数
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="你的密码"
$env:MYSQL_DATABASE="scMSDCL"
$env:MYSQL_ENABLED="1"
```

#### 4.3 验证连接
系统会在首次启动时自动创建必要的表（users、workflows）。

> 详见 [MYSQL_SETUP.md](./MYSQL_SETUP.md) 获取更多配置细节

---

## 🚀 快速开始

### 方式一：启动Web应用 (推荐)

```bash
# 激活虚拟环境
conda activate scMAGCL

# 启动Streamlit应用
cd scSimGCL-main
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动

**工作流**：
1. 注册账户 → 2. 登录系统 → 3. 上传数据 → 4. 创建工作流 → 5. 训练模型 → 6. 查看结果

### 方式二：命令行模式 (高级用户)

```bash
# 使用Baron数据集训练模型
python main.py \
    --data_path './data/Baron.h5' \
    --save_model_path './save_file' \
    --n_clusters 14 \
    --epochs 200 \
    --lr 0.001
```

**参数说明**：
- `--data_path`: 输入数据文件路径 (H5/AnnData格式)
- `--save_model_path`: 模型保存目录
- `--n_clusters`: 目标聚类数
- `--epochs`: 训练轮数 (默认: 200)
- `--lr`: 学习率 (默认: 0.001)

### 获取表示向量

```python
import scanpy as sc
from scMAGCL import Model

# 训练后的表示向量z可用于下游分析
z = model_output  # scSimGCL输出的细胞表示

# 创建AnnData对象
adata = sc.AnnData(z)

# 进行标准的单细胞分析
sc.pp.normalize_total(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color='leiden')
```

---

## 📁 项目结构

```
scSimGCL-main/
├── app.py                    # Streamlit主应用程序
├── main.py                   # 命令行模式训练脚本
├── scMAGCL.py                # 核心模型定义
├── config.py                 # 全局配置参数
├── utils.py                  # 工具函数库
├── auth_utils.py             # 用户认证工具
├── mysql_backend.py          # MySQL数据库接口
│
├── pages/                    # Streamlit多页面应用
│   ├── 0_Login.py            # 登录页面
│   ├── 0_Register.py         # 注册页面
│   ├── 1_Create_Workflow.py  # 创建工作流
│   ├── 2_Explorer.py         # 数据浏览器
│
├── baseline/                 # 对比方法实现
│   └── README.md             # 基准方法文档
│
├── data/                     # 数据存储目录
│   └── (示例数据文件)
│
├── MYSQL_SETUP.md            # MySQL配置指南
└── README.md                 # 本文件
```

### 核心模块说明

#### `scMAGCL.py` - 模型核心
- `GraphConstructor`: 图构造模块（多头注意力）
- `MultiScaleGraphConstructor`: 多尺度图构造
- `Model`: 完整的图对比学习模型

#### `main.py` - 训练流程
- 数据加载和预处理
- 模型训练循环
- 聚类评估 (CA, NMI, ARI)
- 模型检查点保存

#### `config.py` - 超参配置
```python
{
    'graph_head': 5,              # 图注意力头数
    'phi': 0.45,                  # 注意力阈值
    'gcn_dim': 277,               # GCN隐层维度
    'mlp_dim': 118,               # MLP维度
    'prob_feature': 0.1,          # 特征掩码概率
    'prob_edge': 0.5,             # 边掩码概率
    'tau': 0.8,                   # 温度参数
    'lambda_cl': 0.8871,          # 对比损失权重
    'use_byol': True,             # 使用BYOL增强
    'epochs': 200,                # 训练轮数
    'lr': 0.001                   # 学习率
}
```

---

## ⚙️ 配置说明

### 模型超参调整

编辑 `config.py` 调整模型参数：

```python
config = {
    # 图构造参数
    'graph_head': 5,        # ↑ 增加可捕捉更多关系
    'phi': 0.45,           # ↑ 更严格的边选择
    'phi1': 0.06,          # 多尺度参数
    
    # 数据增强参数
    'prob_feature': 0.1,   # 特征掩码强度
    'prob_edge': 0.5,      # 边掩码强度
    
    # 对比学习参数
    'tau': 0.8,            # 温度（↓ 更难的任务）
    'lambda_cl': 0.8871,   # 对比损失权重
    
    # 优化参数
    'lr': 0.001,           # 学习率
    'epochs': 200,         # 总训练轮数
    'dropout': 0.4,        # Dropout比率
}
```

### 数据库配置

支持两种方式配置MySQL连接：

**方式一：URL配置 (推荐)**
```powershell
$env:MYSQL_URL="mysql+pymysql://root:password@localhost:3306/scMSDCL?charset=utf8mb4"
```

**方式二：环境变量配置**
```powershell
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="password"
$env:MYSQL_DATABASE="scMSDCL"
```

**禁用MySQL** (仅本地模式)
```powershell
$env:MYSQL_ENABLED="0"
```

---

## 📊 使用示例

### 例1：训练单个数据集

```bash
python main.py \
    --data_path './data/Baron.h5' \
    --save_model_path './models/baron_model' \
    --n_clusters 14 \
    --epochs 300 \
    --lr 0.0005
```

### 例2：Web应用工作流

```
1. 打开 http://localhost:8501
2. 点击 "Register" 创建账户
3. 登录系统
4. 在 "Create Workflow" 页面：
   - 选择数据文件
   - 配置聚类数、学习率等参数
   - 点击"提交"开始训练
5. 在 "Explorer" 页面查看训练结果和可视化
```

### 例3：集成到分析流程

```python
import scanpy as sc
from scMAGCL import Model

# 加载数据
adata = sc.read_h5ad('data.h5ad')

# 构建数据加载器（见utils.py）
train_loader, test_loader = loader_construction(
    adata.X, 
    adata.obs['cell_type'], 
    n_clusters=14
)

# 初始化模型
model = Model(
    input_dim=adata.n_vars,
    graph_head=5,
    phi=0.45,
    # ... 更多参数
)

# 训练
z = model.train(train_loader, test_loader, epochs=200)

# 结果分析
adata.obsm['X_scMAGCL'] = z
sc.pl.umap(adata, color='cell_type')
```

---

## 📈 评估指标

系统使用以下指标评估聚类质量：

| 指标 | 含义 | 范围 | 解释 |
|-----|------|------|------|
| **CA** (Clustering Accuracy) | 聚类准确度 | [0, 1] | ↑ 越高越好 |
| **NMI** (Normalized Mutual Information) | 归一化互信息 | [0, 1] | ↑ 越高越好 |
| **ARI** (Adjusted Rand Index) | 调整兰德指数 | [-1, 1] | ↑ 越高越好 |

---

## 🔬 支持的数据集

系统已在以下公开数据集上验证：

| 数据集 | 来源 | 细胞数 | 基因数 |
|--------|------|--------|--------|
| Shekhar | GSE81904 | ~44K | ~15K |
| Baron | GSE84133 | ~8K | ~10K |
| 10X PBMC | 10X Genomics | ~4K | ~2K |
| Camp | GSE81252 | ~60K | ~13K |
| Mouse bladder | Figshare | ~50K | ~16K |
| Zeisel | GSE60361 | ~5K | ~18K |
| Tabula Sapiens | Figshare | ~600K+ | ~30K+ |
| Chien | GSE247988 | ~100K+ | ~16K+ |

> 可从各来源下载数据并转换为 H5/HDF5 格式使用

---

## 🐛 故障排除

### 问题1：MySQL连接失败
```
错误: "Can't connect to MySQL server on 'localhost'"

解决方案:
1. 检查MySQL服务是否运行：
   Get-Service | grep -i mysql  (PowerShell)
   
2. 验证连接字符串:
   $env:MYSQL_URL  # 检查是否正确设置
   
3. 重试连接:
   mysql -u root -p -h localhost
```

### 问题2：内存不足
```
错误: "RuntimeError: CUDA out of memory" 或 "MemoryError"

解决方案:
1. 减小batch size (config.py)
2. 使用CPU: CUDA_VISIBLE_DEVICES="" python main.py
3. 增加虚拟内存
```

### 问题3：数据格式不支持
```
错误: "Unsupported file format"

支持格式: .h5, .h5ad, .csv, .txt
转换方法:
import scanpy as sc
adata = sc.read_csv('data.csv', first_column_names=True)
adata.write('data.h5ad')
```

### 问题4：Streamlit页面加载缓慢
```
解决方案:
1. 清除缓存:
   streamlit cache clear
   
2. 减小数据量进行测试
3. 增加系统内存
```

---

## 📚 学习资源

### 相关论文
- Zhang et al. "Graph contrastive learning as a versatile foundation for advanced scRNA-seq data analysis" ***Briefings in Bioinformatics*** 25.6 (2024)

### 工具文档
- [Scanpy文档](https://scanpy.readthedocs.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Streamlit文档](https://docs.streamlit.io/)
- [MySQL文档](https://dev.mysql.com/doc/)

### 单细胞分析教程
- [Orchestrating Single-Cell Analysis](https://bioconductor.org/books/OSCA/)
- [Single-Cell Best Practices](https://www.sc-best-practices.org/)

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 提交步骤：
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件

---

## ✉️ 联系方式

- **问题反馈**: 在GitHub上创建Issue
- **功能建议**: 欢迎通过Issue讨论
- **学术咨询**: 参考相关论文

---

## 🙏 致谢

感谢以下开源项目的支持：
- PyTorch & PyTorch Geometric
- Scanpy
- Streamlit
- 数据提供者和参与者

---

**最后更新**: 2026年5月
**版本**: 1.0.0
**维护者**: scMAGCL Team
