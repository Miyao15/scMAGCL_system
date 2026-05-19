# MySQL 连接与配置说明（scSimGCL）

## 1) 安装 Python 驱动

在你当前环境（`my_pytorch`）执行：

```powershell
pip install pymysql
```

## 2) 准备数据库

你在 Navicat 里已经有 `scMSDCL` 库的话可以直接用。
如果没有，先创建：

```sql
CREATE DATABASE IF NOT EXISTS scMSDCL
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_general_ci;
```

> 代码会自动创建表：`users`、`workflows`。

## 3) 设置连接环境变量（PowerShell）

按你截图的连接信息：`localhost:3306`, `root`, database=`scMSDCL`

```powershell
$env:MYSQL_ENABLED="1"
$env:MYSQL_URL="mysql+pymysql://root:你的密码@localhost:3306/scMSDCL?charset=utf8mb4"
```

如果 root 没密码，可写成：

```powershell
$env:MYSQL_URL="mysql+pymysql://root@localhost:3306/scMSDCL?charset=utf8mb4"
```

> 兼容旧参数方式（`MYSQL_HOST`/`MYSQL_PORT`/`MYSQL_USER`/`MYSQL_PASSWORD`/`MYSQL_DATABASE`），但现在推荐优先使用 `MYSQL_URL`。

## 4) 启动项目

```powershell
cd D:\scSimGCL-main6_cara\scSimGCL-main
streamlit run app.py
```

## 5) 如何确认已连上 MySQL

- 注册新用户后，`scMSDCL.users` 会出现记录。
- 新建工作流后，`scMSDCL.workflows` 会出现对应记录，状态会在 `运行中/已完成/已查看/失败` 间变化。
- 重启 Streamlit 后再次登录，工作流仍会从 MySQL 读出（不丢失）。

## 6) 回退机制（避免启动失败）

- 当 `MYSQL_ENABLED=1` 时，系统会严格要求 MySQL 可连接；连接失败会在页面明确报错并停止，不再静默回退。
- 如需强制禁用 MySQL：

```powershell
$env:MYSQL_ENABLED="0"
```
