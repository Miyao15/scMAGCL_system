import streamlit as st
import pandas as pd
import datetime as dt
import math
import hashlib
import os
from mysql_backend import (
    load_workflows_mysql,
    upsert_workflow_mysql,
    init_mysql_database,
    mysql_requested,
    insert_upload_mysql,
    load_uploads_mysql,
    purge_expired_uploads,
)


def _normalize_dataset_record(record: dict) -> dict:
    return {
        "File Name": record.get("File Name", record.get("文件名", "--")),
        "File Size": record.get("File Size", record.get("大小", "--")),
        "Upload Date": record.get("Upload Date", record.get("上传时间", "--")),
        "Expires": record.get("Expires", record.get("过期时间", "--")),
        "Status": record.get("Status", record.get("状态", "就绪")),
        "Action": record.get("Action", record.get("操作", "创建工作流")),
    }

def _normalize_workflow_record(record: dict) -> dict:
    return {
        "WorkflowId": record.get("WorkflowId") or record.get("workflow_id"),
        "Workflow": record.get("Workflow", record.get("任务名称", "--")),
        "Dataset": record.get("Dataset", record.get("数据", "--")),
        "Epochs": record.get("Epochs", "--"),
        "Best Epoch": record.get("Best Epoch", record.get("best_epoch")),
        "LR": record.get("LR", "--"),
        "Clusters": record.get("Clusters", "--"),
        "BYOL": record.get("BYOL", "--"),
        "Status": record.get("Status", record.get("状态", "--")),
        "Created At": record.get("Created At", record.get("创建时间", "--")),
        "CA": record.get("CA"),
        "NMI": record.get("NMI"),
        "ARI": record.get("ARI"),
        "embed_path": record.get("embed_path"),
        "label_path": record.get("label_path"),
    }


def _go(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        pass

st.set_page_config(
        page_title="单细胞智能分析系统（scMAGCL）- 工作台",
        page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UPLOAD_ROOT = os.path.join("D:/", "scMAGCL_uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

if "datasets" not in st.session_state:
        st.session_state.datasets = []
if "dataset_store" not in st.session_state:
        st.session_state.dataset_store = {}
if "workflows" not in st.session_state:
        st.session_state.workflows = []
if "uploaded_signatures" not in st.session_state:
    st.session_state.uploaded_signatures = set()
if "show_upload_panel" not in st.session_state:
    st.session_state.show_upload_panel = False
if "last_completed_workflow" not in st.session_state:
    st.session_state.last_completed_workflow = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

st.session_state.datasets = [_normalize_dataset_record(r) for r in st.session_state.datasets]
st.session_state.workflows = [_normalize_workflow_record(r) for r in st.session_state.workflows]

if not st.session_state.logged_in:
    _go("pages/0_Login.py")
    st.stop()

mysql_ok, mysql_msg = init_mysql_database()
if mysql_requested() and not mysql_ok:
    st.error(f"MySQL 连接失败：{mysql_msg}")
    st.stop()

purge_expired_uploads()

db_uploads = load_uploads_mysql(st.session_state.username)
if db_uploads is not None:
    st.session_state.datasets = []
    st.session_state.dataset_store = {}
    for r in db_uploads:
        fname = r.get("file_name") or "--"
        size_mb = r.get("size_mb") or 0
        uploaded_at = r.get("uploaded_at")
        expires_at = r.get("expires_at")
        record = {
            "File Name": fname,
            "File Size": f"{size_mb:.2f} MB",
            "Upload Date": uploaded_at.strftime("%b %d, %Y") if uploaded_at else "--",
            "Expires": expires_at.strftime("%b %d, %Y %I:%M %p") if expires_at else "--",
            "Status": "就绪",
            "Action": "创建工作流",
            "File Path": r.get("file_path"),
        }
        st.session_state.datasets.append(record)
        fp = r.get("file_path")
        if fp and os.path.exists(fp):
            try:
                with open(fp, "rb") as f:
                    st.session_state.dataset_store[fname] = f.read()
            except Exception:
                pass

db_workflows = load_workflows_mysql(st.session_state.username)
if db_workflows is not None:
    seen = {}
    deduped = []
    for r in db_workflows:
        key = r.get("WorkflowId") or r.get("workflow_id") or ""
        if not key:
            continue
        if key in seen:
            continue
        seen[key] = True
        deduped.append(r)
    st.session_state.workflows = [_normalize_workflow_record(r) for r in deduped]

st.markdown(
    """
    <style>
    .headline {font-size:2.2rem;font-weight:800;margin:0;}
    .sub {color:#4b5563;font-size:0.95rem;margin-top:4px;}
    .card {background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,0.04);}    
    div[data-testid="stButton"][id*="status_btn_"] > button {
        background: transparent !important;
        border: none !important;
        color: #2563eb !important;
        padding: 0 !important;
        height: auto !important;
        min-height: auto !important;
        box-shadow: none !important;
        text-decoration: underline;
    }
    div[data-testid="stButton"][id*="status_btn_"] > button:hover {
        color: #1d4ed8 !important;
    }
    div[data-testid="stButton"][id*="status_btn_"] > button:disabled {
        color: #9ca3af !important;
        text-decoration: none;
    }
    div[data-testid="stButton"] > button { white-space: nowrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {display: none;}
            div[data-testid="stToolbar"] {display: none;}
            header {visibility: hidden; height: 0;}
            div.block-container {padding-top: 0.6rem; max-width: 1400px;}
            div[data-testid="stFileUploader"] {max-width: 560px;}
        </style>
        """,
        unsafe_allow_html=True,
)

col_title, col_actions = st.columns([3, 2])
with col_title:
    st.markdown("<div class='headline'>工作台 <span style='font-size:0.9rem;color:#6366f1;'>Beta</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>用零代码工作流探索单细胞数据</div>", unsafe_allow_html=True)

with col_actions:
    c1, c2, c3 = st.columns([1.1, 1.2, 1.7])
    with c1:
        if st.button("+ 上传数据", key="top_upload_btn", width="stretch"):
            st.session_state.show_upload_panel = True
    with c2:
        if st.button("+ 新建工作流", key="top_create_btn", width="stretch"):
            _go("pages/1_Create_Workflow.py")
    with c3:
                avatar = "https://avatars.githubusercontent.com/u/9919?v=4"
                username = st.session_state.username or "用户"
                prof_col, logout_col = st.columns([2.0, 1.2])
                with prof_col:
                        st.markdown(
                                f"""
                                <div style='display:flex;align-items:center;justify-content:flex-end;gap:10px;'>
                                    <div style='display:flex;align-items:center;gap:8px;padding:6px 12px;border:1px solid #e5e7eb;border-radius:999px;background:#f8fafc;'>
                                        <img src='{avatar}' style='width:24px;height:24px;border-radius:50%;object-fit:cover;border:1px solid #e5e7eb;' />
                                        <span style='font-weight:600;color:#111827;'>{username}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                        )
                with logout_col:
                    if st.button("退出", key="logout_btn", type="secondary", use_container_width=True):
                        st.session_state.logged_in = False
                        st.session_state.username = ""
                        _go("pages/0_Login.py")

st.divider()

if st.session_state.show_upload_panel:
    st.markdown("**上传数据文件**")
    uploaded = st.file_uploader("支持 .h5 / .h5ad", type=["h5", "h5ad"], key="top_upload_uploader")
    if uploaded is not None:
        content = uploaded.getvalue()
        sig = hashlib.md5(content).hexdigest()
        if sig not in st.session_state.uploaded_signatures:
            fname = uploaded.name
            size_mb = len(content) / (1024 * 1024)
            upload_time = dt.datetime.now()
            expires = upload_time + dt.timedelta(days=180)

            user_dir = os.path.join(UPLOAD_ROOT, st.session_state.username or "default")
            os.makedirs(user_dir, exist_ok=True)
            stored_name = f"{upload_time.strftime('%Y%m%d%H%M%S')}_{fname}"
            file_path = os.path.join(user_dir, stored_name)
            with open(file_path, "wb") as f:
                f.write(content)

            record = {
                "File Name": fname,
                "File Size": f"{size_mb:.2f} MB",
                "Upload Date": upload_time.strftime("%b %d, %Y"),
                "Expires": expires.strftime("%b %d, %Y %I:%M %p"),
                "Status": "就绪",
                "Action": "创建工作流",
                "File Path": file_path,
            }
            st.session_state.datasets.append(record)
            st.session_state.dataset_store[fname] = content
            st.session_state.uploaded_signatures.add(sig)
            insert_upload_mysql(
                st.session_state.username,
                fname,
                file_path,
                size_mb,
                upload_time,
                expires,
                sig,
            )
            st.success(f"已上传 {fname} 到 {file_path}")
            st.session_state.show_upload_panel = False

tab_flows, tab_data = st.tabs(["我的工作流", "我的数据"])

with tab_data:
    st.caption("数据将在上传后 6 个月自动删除。")
    if st.session_state.datasets:
        df = pd.DataFrame(st.session_state.datasets)
        df = df[["File Name", "File Size", "Upload Date", "Expires", "Status", "Action"]]
        df_show = df.rename(columns={
            "File Name": "文件名",
            "File Size": "文件大小",
            "Upload Date": "上传日期",
            "Expires": "过期时间",
            "Status": "状态",
            "Action": "操作",
        })
        st.dataframe(df_show, width="stretch", hide_index=True)
    else:
        st.info("暂无数据，请点击右上角“上传数据”。")

with tab_flows:
    st.caption("可在此查看工作流状态，并点击进入可视化页。")
    if st.session_state.workflows:
        last_done = st.session_state.last_completed_workflow

        for idx, row in enumerate(st.session_state.workflows):
            if not last_done:
                break
            same_job = (
                row.get("Workflow") == last_done.get("Workflow")
                and row.get("Dataset") == last_done.get("Dataset")
            )
            missing_metrics = any(row.get(metric) is None for metric in ["CA", "NMI", "ARI", "MAE"])
            if same_job and missing_metrics:
                for metric in ["CA", "NMI", "ARI", "MAE"]:
                    st.session_state.workflows[idx][metric] = last_done.get(metric)

        h1, h2, h3, h4, h5, h6, h7 = st.columns([2.0, 2.8, 1.1, 1.6, 0.8, 0.8, 0.8])
        h1.markdown("**工作流**")
        h2.markdown("**数据集**")
        h3.markdown("**状态**")
        h4.markdown("**创建时间**")
        h5.markdown("**CA**")
        h6.markdown("**NMI**")
        h7.markdown("**ARI**")
        st.markdown("---")

        for i, row in enumerate(st.session_state.workflows):
            r1, r2, r3, r4, r5, r6, r7 = st.columns([2.0, 2.8, 1.1, 1.6, 0.8, 0.8, 0.8])
            row_name = row.get("Workflow", "--")
            row_data = row.get("Dataset", "--")
            row_status = row.get("Status", "--")

            if row_status == "已完成" or row_status == "已查看":
                status_label = "已完成"
            else:
                status_label = row_status

            ca = "--" if row.get("CA") is None else f"{row.get('CA')}"
            nmi = "--" if row.get("NMI") is None else f"{row.get('NMI')}"
            ari = "--" if row.get("ARI") is None else f"{row.get('ARI')}"
            with r1:
                st.markdown(row_name)
            with r2:
                st.markdown(row_data)
            with r3:
                if row_status == "运行中":
                    st.button("运行中", key=f"status_btn_{i}", width="stretch", disabled=True)
                else:
                    if st.button(status_label, key=f"status_btn_{i}", width="stretch"):
                        st.session_state.active_workflow = st.session_state.workflows[i]
                        _go("pages/2_Explorer.py")
            with r4:
                st.caption(row.get("Created At", "--"))
            with r5:
                st.caption(ca)
            with r6:
                st.caption(nmi)
            with r7:
                st.caption(ari)
            st.markdown("---")
    else:
        st.info("暂无工作流，请点击“新建工作流”。")

col_l, col_r = st.columns([3,1])
with col_l:
    st.caption("流程：上传数据 → 新建工作流 → 打开可视化页。")
with col_r:
    st.page_link("pages/1_Create_Workflow.py", label="新建工作流", icon=None, width="stretch")

st.divider()

st.markdown("**提示：** 点击右上角“上传数据”导入文件，然后运行工作流并打开可视化页。")
