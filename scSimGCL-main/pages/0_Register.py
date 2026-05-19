import streamlit as st
from auth_utils import register_user


def _go(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        pass


st.set_page_config(page_title="注册 - 基于多尺度图神经网络的单细胞智能分析系统", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {display: none;}
        div[data-testid="stToolbar"] {display: none;}
        header {visibility: hidden; height: 0;}
        div.block-container {padding-top: 0.8rem; max-width: 1280px;}
        .hero {
            height: 640px;
            background:
                radial-gradient(circle at 20% 20%, rgba(56,189,248,0.28), transparent 35%),
                radial-gradient(circle at 75% 70%, rgba(59,130,246,0.25), transparent 40%),
                linear-gradient(135deg, #eef4ff 0%, #f7fbff 55%, #eef6ff 100%);
            border-right: 1px solid #e5e7eb;
            display:flex;align-items:center;justify-content:center;position:relative;
        }
        .hero-card {
            width: 68%; min-height: 280px; border-radius: 18px;
            background: linear-gradient(180deg, #c8efff 0%, #82c9ff 45%, #4b7bff 100%);
            box-shadow: 0 18px 35px rgba(59,130,246,0.18);
            opacity: 0.9;
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
    </style>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.65, 1.0], gap="small")

with left:
    st.markdown(
        """
        <div class='hero'>
          <div class='hero-card'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("<div class='right-wrap'>", unsafe_allow_html=True)
    st.markdown("<span class='download-pill'>Download APP</span>", unsafe_allow_html=True)
    st.markdown("<div class='sys-title'>基于多尺度图神经网络的单细胞智能分析系统</div>", unsafe_allow_html=True)
    st.markdown("<div class='sys-sub'>创建账号后即可登录</div>", unsafe_allow_html=True)

    username = st.text_input("用户名", placeholder="至少 3 个字符")
    password = st.text_input("密码", type="password", placeholder="至少 6 位")
    confirm = st.text_input("确认密码", type="password", placeholder="再次输入密码")

    if st.button("注  册", type="primary", width="stretch"):
        if password != confirm:
            st.error("两次输入的密码不一致")
        else:
            ok, msg = register_user(username, password)
            if ok:
                st.success("注册成功，请登录")
                _go("pages/0_Login.py")
            else:
                st.error(msg)

    if st.button("返回登录", width="stretch"):
        _go("pages/0_Login.py")

    st.markdown("</div>", unsafe_allow_html=True)
