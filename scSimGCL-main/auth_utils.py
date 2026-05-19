import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple

from mysql_backend import register_user_mysql, authenticate_user_mysql, init_mysql_database, mysql_requested

USERS_FILE = Path(__file__).resolve().parent / ".users.json"


def _ensure_users_file() -> None:
    if not USERS_FILE.exists():
        USERS_FILE.write_text("{}", encoding="utf-8")


def _load_users() -> Dict[str, str]:
    _ensure_users_file()
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users: Dict[str, str]) -> None:
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> Tuple[bool, str]:
    username = (username or "").strip()
    if len(username) < 3:
        return False, "用户名至少 3 个字符"
    if len(password or "") < 6:
        return False, "密码至少 6 位"

    pwd_hash = hash_password(password)

    # 优先 MySQL
    init_mysql_database()
    ok_mysql, msg_mysql, used_mysql = register_user_mysql(username, pwd_hash)
    if used_mysql:
        return ok_mysql, msg_mysql

    if mysql_requested():
        return False, "MySQL 已启用但连接失败，请检查 MYSQL_URL 或连接参数"

    # 回退本地文件
    users = _load_users()
    if username in users:
        return False, "用户名已存在"

    users[username] = pwd_hash
    _save_users(users)
    return True, "注册成功（本地模式）"


def authenticate_user(username: str, password: str) -> bool:
    username = (username or "").strip()
    pwd_hash = hash_password(password or "")

    # 优先 MySQL
    init_mysql_database()
    ok_mysql, used_mysql = authenticate_user_mysql(username, pwd_hash)
    if used_mysql:
        return ok_mysql

    if mysql_requested():
        return False

    # 回退本地文件
    users = _load_users()
    if username not in users:
        return False
    return users[username] == pwd_hash
