import streamlit as st
import base64
from pathlib import Path
from auth_utils import authenticate_user
from auth_utils import register_user


def _go(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        pass


def _get_hero_image_src() -> str:
    root_dir = Path(__file__).resolve().parents[1]
    candidates = [
        root_dir / "assets" / "login_hero.png",
        root_dir / "assets" / "login_hero.jpg",
        root_dir / "assets" / "login_hero.jpeg",
    ]
    for image_path in candidates:
        if image_path.exists():
            suffix = image_path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    return "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=1400&q=80"


st.set_page_config(page_title="登录 - 基于多尺度图神经网络的单细胞智能分析系统", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {display: none;}
        div[data-testid="stToolbar"] {display: none;}
        header {visibility: hidden; height: 0;}
        div.block-container {padding-top: 0.8rem; max-width: 1280px;}
        .auth-shell {border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;background:#fff;}
        .hero {
            height: 640px;
            border: 1px solid #e5ebf7;
            border-radius: 16px;
            overflow: hidden;
            background: #eef4ff;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
            display:flex;align-items:center;justify-content:center;position:relative;
        }
        .hero-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: saturate(0.86) brightness(1.14) contrast(0.9);
            transform: scale(1.01);
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(180deg, rgba(248, 251, 255, 0.46), rgba(231, 240, 255, 0.34)),
                radial-gradient(circle at 75% 22%, rgba(59,130,246,0.16), transparent 48%);
            pointer-events: none;
        }
        .right-wrap {padding: 44px 48px;}
        .sys-title {font-size:2rem;font-weight:800;color:#111827;margin-bottom:6px;}
        .sys-sub {font-size:0.95rem;color:#6b7280;margin-bottom:18px;}
        .download-pill {float:right;background:#3b82f6;color:#fff;border-radius:999px;padding:6px 12px;font-size:0.78rem;}
        div.stButton > button {
            height: 42px;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            background: #ffffff;
            color: #1f2937;
            font-weight: 600;
        }
        div.stButton > button[kind="primary"] {
            background: #3b82f6 !important;
            border-color: #3b82f6 !important;
            color: #ffffff !important;
        }
        .link-row {display:flex;justify-content:space-between;align-items:center;margin-top:8px;}
        .auth-link {color:#3b82f6;text-decoration:none;font-weight:600;font-size:1.05rem;}
        .auth-link:hover {text-decoration:underline;}
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("logged_in"):
    st.query_params.clear()
    _go("app.py")
    st.stop()

if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"

auth_qs = st.query_params.get("auth", "")
if auth_qs in ["login", "register"]:
    st.session_state.auth_view = auth_qs

hero_src = _get_hero_image_src()

left, right = st.columns([1.65, 1.0], gap="small")

with left:
    st.markdown(
        f"""
        <div class='hero'>
            <img class='hero-img' src='{hero_src}' alt='数据可视化背景图' />
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("<div class='right-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='sys-title'>基于多尺度图神经网络的单细胞智能分析系统</div>", unsafe_allow_html=True)
    is_register = st.session_state.auth_view == "register"

    if is_register:
        st.markdown("<div class='sys-sub'>创建账号后即可登录</div>", unsafe_allow_html=True)

        reg_username = st.text_input("用户名", placeholder="至少 3 个字符", key="reg_username")
        reg_password = st.text_input("密码", type="password", placeholder="至少 6 位", key="reg_password")
        reg_confirm = st.text_input("确认密码", type="password", placeholder="再次输入密码", key="reg_confirm")

        if st.button("注  册", type="primary", width="stretch", key="register_submit"):
            if reg_password != reg_confirm:
                st.error("两次输入的密码不一致")
            else:
                ok, msg = register_user(reg_username, reg_password)
                if ok:
                    st.success("注册成功，请登录")
                    st.session_state.auth_view = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown(
            "<div class='link-row'><span class='sys-sub' style='margin:0;'>已有账号？</span><a class='auth-link' href='?auth=login'>返回登录</a></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='sys-sub'>请输入账号密码进行登录</div>", unsafe_allow_html=True)

        username = st.text_input("用户名", placeholder="请输入用户名", key="login_username")
        password = st.text_input("密码", type="password", placeholder="请输入密码", key="login_password")
        remember = st.checkbox("记住密码", value=False, key="remember_me")

        if st.button("登  录", type="primary", width="stretch", key="login_submit"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.query_params.clear()
                _go("app.py")
            else:
                st.error("用户名或密码错误")

        st.markdown(
            "<div class='link-row'><span class='sys-sub' style='margin:0;'>忘记密码</span><a class='auth-link' href='?auth=register'>注册</a></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
