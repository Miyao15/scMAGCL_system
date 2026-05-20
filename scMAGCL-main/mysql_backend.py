import os
import importlib
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote

try:
    pymysql = importlib.import_module("pymysql")
except Exception:
    pymysql = None


MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "scMSDCL")
MYSQL_URL = os.getenv("MYSQL_URL", "").strip()
MYSQL_ENABLED = os.getenv("MYSQL_ENABLED", "1") == "1"


def mysql_ready() -> bool:
    return MYSQL_ENABLED and pymysql is not None


def mysql_requested() -> bool:
    return MYSQL_ENABLED


def get_mysql_runtime_config(mask_password: bool = True) -> Dict:
    cfg = {
        "enabled": MYSQL_ENABLED,
        "mode": "url" if MYSQL_URL else "params",
    }

    if MYSQL_URL:
        parsed = urlparse(MYSQL_URL)
        pwd = unquote(parsed.password or "")
        cfg.update(
            {
                "host": parsed.hostname or MYSQL_HOST,
                "port": parsed.port or MYSQL_PORT,
                "user": unquote(parsed.username or MYSQL_USER),
                "password": "***" if (mask_password and pwd) else pwd,
                "database": parsed.path.lstrip("/") or MYSQL_DATABASE,
                "url": MYSQL_URL,
            }
        )
    else:
        cfg.update(
            {
                "host": MYSQL_HOST,
                "port": MYSQL_PORT,
                "user": MYSQL_USER,
                "password": "***" if (mask_password and MYSQL_PASSWORD) else MYSQL_PASSWORD,
                "database": MYSQL_DATABASE,
            }
        )
    return cfg


def _connection_kwargs() -> Dict:
    if MYSQL_URL:
        parsed = urlparse(MYSQL_URL)
        if not parsed.scheme.startswith("mysql"):
            raise ValueError("MYSQL_URL 协议必须是 mysql:// 或 mysql+pymysql://")

        query = parse_qs(parsed.query)
        database = parsed.path.lstrip("/") or MYSQL_DATABASE
        if not database:
            raise ValueError("MYSQL_URL 缺少数据库名")

        charset = query.get("charset", ["utf8mb4"])[0]

        return {
            "host": parsed.hostname or MYSQL_HOST,
            "port": parsed.port or MYSQL_PORT,
            "user": unquote(parsed.username or MYSQL_USER),
            "password": unquote(parsed.password or MYSQL_PASSWORD),
            "database": database,
            "charset": charset,
            "autocommit": True,
            "cursorclass": pymysql.cursors.DictCursor,
        }

    return {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "database": MYSQL_DATABASE,
        "charset": "utf8mb4",
        "autocommit": True,
        "cursorclass": pymysql.cursors.DictCursor,
    }


def _connect():
    if not mysql_ready():
        return None
    return pymysql.connect(**_connection_kwargs())


def _connect_server_only():
    if not mysql_ready():
        return None
    kwargs = _connection_kwargs()
    kwargs.pop("database", None)
    return pymysql.connect(**kwargs)


def ensure_mysql_database_exists() -> Tuple[bool, str]:
    if not MYSQL_ENABLED:
        return False, "MYSQL_ENABLED=0，已禁用 MySQL"
    if pymysql is None:
        return False, "未安装 pymysql"

    try:
        cfg = get_mysql_runtime_config(mask_password=True)
        database_name = cfg.get("database") or MYSQL_DATABASE
        if not database_name:
            return False, "数据库名为空"

        conn = _connect_server_only()
        if conn is None:
            return False, "MySQL 服务未就绪"
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_general_ci"
            )
        conn.close()
        return True, f"数据库已就绪: {database_name}"
    except Exception as exc:
        return False, f"创建数据库失败: {exc}"


def check_mysql_connection() -> Tuple[bool, str]:
    if not MYSQL_ENABLED:
        return False, "MYSQL_ENABLED=0，已禁用 MySQL"
    if pymysql is None:
        return False, "未安装 pymysql"

    try:
        conn = _connect()
        if conn is None:
            return False, "MySQL 未就绪"
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
        conn.close()
        if not row or row.get("ok") != 1:
            return False, "MySQL 心跳检查失败"
        cfg = get_mysql_runtime_config(mask_password=True)
        return True, f"MySQL 已连接: {cfg.get('host')}:{cfg.get('port')}/{cfg.get('database')}"
    except Exception as exc:
        return False, f"MySQL 连接失败: {exc}"


def init_mysql_database() -> Tuple[bool, str]:
    ok_db, msg_db = ensure_mysql_database_exists()
    if not ok_db:
        return False, msg_db

    ok_conn, msg_conn = check_mysql_connection()
    if not ok_conn:
        return False, msg_conn

    try:
        conn = _connect()
        if conn is None:
            return False, "MySQL 未就绪"
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(128) NOT NULL UNIQUE,
                    password_hash VARCHAR(128) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id VARCHAR(64) PRIMARY KEY,
                    username VARCHAR(128) NOT NULL,
                    workflow VARCHAR(255),
                    dataset VARCHAR(255),
                    epochs INT NULL,
                    best_epoch INT NULL,
                    lr DOUBLE NULL,
                    clusters INT NULL,
                    byol VARCHAR(16),
                    status VARCHAR(32),
                    created_at VARCHAR(32),
                    ca DOUBLE NULL,
                    nmi DOUBLE NULL,
                    ari DOUBLE NULL,
                    mae DOUBLE NULL,
                    embed_path VARCHAR(512) NULL,
                    label_path VARCHAR(512) NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_workflows_user (username)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            try:
                cur.execute("ALTER TABLE workflows ADD COLUMN embed_path VARCHAR(512) NULL")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE workflows ADD COLUMN label_path VARCHAR(512) NULL")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE workflows ADD COLUMN best_epoch INT NULL")
            except Exception:
                pass
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(128) NOT NULL,
                    file_name VARCHAR(255) NOT NULL,
                    file_path VARCHAR(512) NOT NULL,
                    size_mb DOUBLE NULL,
                    uploaded_at DATETIME NOT NULL,
                    expires_at DATETIME NOT NULL,
                    md5 CHAR(32) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_uploads_user (username),
                    INDEX idx_uploads_exp (expires_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
        conn.close()
        return True, "MySQL 初始化成功"
    except Exception as exc:
        return False, f"MySQL 初始化失败: {exc}"


def register_user_mysql(username: str, password_hash: str) -> Tuple[bool, str, bool]:
    if not mysql_requested():
        return False, "MySQL 不可用", False
    if pymysql is None:
        return False, "未安装 pymysql", True

    try:
        ok, msg = init_mysql_database()
        if not ok:
            return False, msg, True
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s", (username,))
            if cur.fetchone() is not None:
                conn.close()
                return False, "用户名已存在", True
            cur.execute(
                "INSERT INTO users(username, password_hash) VALUES(%s, %s)",
                (username, password_hash),
            )
        conn.close()
        return True, "注册成功", True
    except Exception as exc:
        return False, f"MySQL 注册失败: {exc}", True


def authenticate_user_mysql(username: str, password_hash: str) -> Tuple[bool, bool]:
    if not mysql_requested():
        return False, False
    if pymysql is None:
        return False, True
    try:
        ok, _ = init_mysql_database()
        if not ok:
            return False, True
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash FROM users WHERE username=%s", (username,))
            row = cur.fetchone()
        conn.close()
        if row is None:
            return False, True
        return row.get("password_hash") == password_hash, True
    except Exception:
        return False, True


def upsert_workflow_mysql(username: str, record: Dict) -> bool:
    if not mysql_requested() or pymysql is None:
        return False
    try:
        ok, _ = init_mysql_database()
        if not ok:
            return False
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workflows(
                    workflow_id, username, workflow, dataset, epochs, best_epoch, lr, clusters, byol, status,
                    created_at, ca, nmi, ari, mae, embed_path, label_path
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    workflow=VALUES(workflow),
                    dataset=VALUES(dataset),
                    epochs=VALUES(epochs),
                    best_epoch=VALUES(best_epoch),
                    lr=VALUES(lr),
                    clusters=VALUES(clusters),
                    byol=VALUES(byol),
                    status=VALUES(status),
                    created_at=VALUES(created_at),
                    ca=VALUES(ca),
                    nmi=VALUES(nmi),
                    ari=VALUES(ari),
                    mae=VALUES(mae),
                    embed_path=VALUES(embed_path),
                    label_path=VALUES(label_path)
                """,
                (
                    str(record.get("WorkflowId", "")),
                    username,
                    record.get("Workflow"),
                    record.get("Dataset"),
                    _to_int(record.get("Epochs")),
                    _to_int(record.get("Best Epoch")),
                    _to_float(record.get("LR")),
                    _to_int(record.get("Clusters")),
                    record.get("BYOL"),
                    record.get("Status"),
                    record.get("Created At"),
                    _to_float(record.get("CA")),
                    _to_float(record.get("NMI")),
                    _to_float(record.get("ARI")),
                    _to_float(record.get("MAE")),
                    record.get("embed_path"),
                    record.get("label_path"),
                ),
            )
        conn.close()
        return True
    except Exception:
        return False


def load_workflows_mysql(username: str) -> Optional[List[Dict]]:
    if not mysql_requested() or pymysql is None:
        return None
    try:
        ok, _ = init_mysql_database()
        if not ok:
            return None
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT workflow_id, workflow, dataset, epochs, lr, clusters, byol,
                      best_epoch, status, created_at, ca, nmi, ari, mae, embed_path, label_path
                FROM workflows
                WHERE username=%s
                ORDER BY updated_at DESC
                """,
                (username,),
            )
            rows = cur.fetchall()
        conn.close()

        workflows = []
        for row in rows:
            workflows.append(
                {
                    "WorkflowId": row.get("workflow_id"),
                    "Workflow": row.get("workflow"),
                    "Dataset": row.get("dataset"),
                    "Epochs": row.get("epochs"),
                    "Best Epoch": row.get("best_epoch"),
                    "LR": row.get("lr"),
                    "Clusters": row.get("clusters"),
                    "BYOL": row.get("byol"),
                    "Status": row.get("status"),
                    "Created At": row.get("created_at"),
                    "CA": row.get("ca"),
                    "NMI": row.get("nmi"),
                    "ARI": row.get("ari"),
                    "MAE": row.get("mae"),
                    "embed_path": row.get("embed_path"),
                    "label_path": row.get("label_path"),
                }
            )
        return workflows
    except Exception:
        return None


def insert_upload_mysql(username: str, file_name: str, file_path: str, size_mb: float,
                        uploaded_at, expires_at, md5: Optional[str]) -> bool:
    if not mysql_requested() or pymysql is None:
        return False
    try:
        ok, _ = init_mysql_database()
        if not ok:
            return False
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO uploads(username, file_name, file_path, size_mb, uploaded_at, expires_at, md5)
                VALUES(%s,%s,%s,%s,%s,%s,%s)
                """,
                (username, file_name, file_path, size_mb, uploaded_at, expires_at, md5),
            )
        conn.close()
        return True
    except Exception:
        return False


def load_uploads_mysql(username: str) -> Optional[List[Dict]]:
    if not mysql_requested() or pymysql is None:
        return None
    try:
        ok, _ = init_mysql_database()
        if not ok:
            return None
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_name, file_path, size_mb, uploaded_at, expires_at, md5
                FROM uploads
                WHERE username=%s
                ORDER BY uploaded_at DESC
                """,
                (username,),
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return None


def purge_expired_uploads() -> int:
    if not mysql_requested() or pymysql is None:
        return 0
    removed = 0
    try:
        ok, _ = init_mysql_database()
        if not ok:
            return 0
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, file_path FROM uploads WHERE expires_at < NOW()"
            )
            rows = cur.fetchall()
            ids = [r.get("id") for r in rows if r.get("id") is not None]
            for r in rows:
                fp = r.get("file_path")
                try:
                    if fp and os.path.exists(fp):
                        os.remove(fp)
                except Exception:
                    pass
            if ids:
                cur.execute(
                    "DELETE FROM uploads WHERE id IN (" + ",".join(["%s"] * len(ids)) + ")",
                    ids,
                )
                removed = len(ids)
        conn.close()
    except Exception:
        return removed
    return removed


def _to_float(v):
    try:
        if v is None or v == "--":
            return None
        return float(v)
    except Exception:
        return None


def _to_int(v):
    try:
        if v is None or v == "--":
            return None
        return int(v)
    except Exception:
        return None
