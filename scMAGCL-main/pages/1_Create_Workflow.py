import streamlit as st
import datetime as dt
import os
import uuid
import tempfile
import numpy as np


def _go(page_path: str):
  try:
    st.switch_page(page_path)
  except Exception:
    pass

from utils import loader_construction, device
from main import train as train_model, test as test_cluster
from config import config
from mysql_backend import upsert_workflow_mysql, init_mysql_database

UPLOAD_ROOT = os.path.join("D:/", "scSimGCL_uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)


def _persist_embedding(username: str, workflow_id: str, embed_mat: np.ndarray, labels: np.ndarray):
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    user_dir = os.path.join(UPLOAD_ROOT, username or "default", "embeds")
    os.makedirs(user_dir, exist_ok=True)
    embed_path = os.path.join(user_dir, f"{workflow_id}_{ts}_embed.npy")
    label_path = os.path.join(user_dir, f"{workflow_id}_{ts}_label.npy")
    np.save(embed_path, embed_mat)
    np.save(label_path, labels)
    return embed_path, label_path

st.set_page_config(page_title="新建工作流 - scMAGCL", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {display: none;}
            div[data-testid="stToolbar"] {display: none;}
            header {visibility: hidden; height: 0;}
            div.block-container {padding-top: 0.8rem; max-width: 1400px;}
            .label-small {font-size:0.9rem;color:#4b5563;font-weight:600;margin-bottom:4px;}
            .hint {font-size:0.85rem;color:#6b7280;margin-top:4px;}
            .card {background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;box-shadow:0 3px 10px rgba(0,0,0,0.03);}    
        </style>
        """,
        unsafe_allow_html=True,
)

if "datasets" not in st.session_state:
    st.session_state.datasets = []
if "dataset_store" not in st.session_state:
    st.session_state.dataset_store = {}
if "workflows" not in st.session_state:
    st.session_state.workflows = []
if "active_workflow" not in st.session_state:
    st.session_state.active_workflow = None
if "training_in_progress" not in st.session_state:
    st.session_state.training_in_progress = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    _go("pages/0_Login.py")
    st.stop()

init_mysql_database()

col_nav, col_title = st.columns([1,6])
with col_nav:
    if st.button("返回", width="content"):
        _go("app.py")
with col_title:
    st.markdown("<div style='font-size:1.8rem;font-weight:800;margin-bottom:6px;'>新建工作流</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#4b5563;font-size:0.98rem;'>配置 GCN + BYOL 任务，右侧实时摘要。</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

if not st.session_state.datasets:
    st.warning("请先在工作台上传数据，再创建工作流。")
    st.stop()

col_form, col_summary = st.columns([2.2, 1])

with col_form:
    st.markdown("<div class='label-small'>参数配置</div>", unsafe_allow_html=True)
    st.markdown("<div class='label-small'>工作流名称</div>", unsafe_allow_html=True)
    wf_name = st.text_input(label="工作流名称", value="Workflow_01", label_visibility="collapsed", key="wf_name")

    st.markdown("<div class='label-small' style='margin-top:8px;'>选择数据</div>", unsafe_allow_html=True)
    dataset_names = [d.get("File Name", d.get("文件名", "")) for d in st.session_state.datasets]
    default_idx = 0
    if dataset_names and st.session_state.get("wf_data") in dataset_names:
        default_idx = dataset_names.index(st.session_state.get("wf_data"))
    data_choice = st.selectbox(label="选择数据", options=dataset_names, index=default_idx, label_visibility="collapsed", key="wf_data")
    st.markdown("<div class='hint'>查看工作流文件要求</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='label-small'>模型参数</div>", unsafe_allow_html=True)
    epochs = st.slider("训练轮数", 10, 500, 50, 10, key="wf_epochs")
    lr = st.selectbox("学习率", [0.01, 0.005, 0.001, 0.0001], index=2, key="wf_lr")
    clusters = st.number_input("聚类簇数", min_value=2, max_value=30, value=9, key="wf_clusters")
    byol = st.checkbox("启用 BYOL", value=True, key="wf_byol")
    
    # 超参数微调
    st.markdown("<div class='label-small' style='margin-top:12px;'>超参数微调</div>", unsafe_allow_html=True)
    tau = st.slider("温度参数 (tau)", 0.1, 2.0, config['tau'], 0.1, key="wf_tau", help="对比学习损失的温度参数，值越小对比越强")
    lambda_byol = st.slider("BYOL损失权重 (λ_BYOL)", 0.0, 2.0, config['lambda_byol'], 0.1, key="wf_lambda_byol", help="BYOL损失项的权重，0表示不使用BYOL")

    has_running = any(w.get("Status") == "运行中" for w in st.session_state.workflows)
    submitted = st.button("运行工作流", type="primary", disabled=has_running or st.session_state.training_in_progress)
    if has_running or st.session_state.training_in_progress:
        st.info("已有任务在运行中，请等待完成后再启动新的工作流。")

    if submitted:
        st.session_state.training_in_progress = True
        if data_choice not in st.session_state.dataset_store:
            st.error("找不到所选数据，请返回工作台重新上传。")
            st.session_state.training_in_progress = False
        else:
            workflow_id = str(uuid.uuid4())
            pending_record = {
                "WorkflowId": workflow_id,
                "Workflow": wf_name,
                "Dataset": data_choice,
                "Epochs": epochs,
                "Best Epoch": None,
                "LR": lr,
                "Clusters": clusters,
                "BYOL": "On" if byol else "Off",
                "Status": "运行中",
                "Created At": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "CA": None,
                "NMI": None,
                "ARI": None,
                "MAE": None,
            }
            st.session_state.workflows.append(pending_record)
            upsert_workflow_mysql(st.session_state.username, pending_record)
            row_index = len(st.session_state.workflows) - 1
            st.info("任务已提交，当前状态：运行中")
            tmp_file = None
            try:
                with st.spinner("工作流运行中，请稍候..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                        tmp.write(st.session_state.dataset_store[data_choice])
                        tmp_file = tmp.name

                    graph_head = config['graph_head']
                    phi = config['phi']
                    gcn_dim = config['gcn_dim']
                    mlp_dim = config['mlp_dim']
                    prob_feature = config['prob_feature']
                    prob_edge = config['prob_edge']
                    alpha = config['alpha']
                    beta = config['beta']
                    lambda_cl = config['lambda_cl']
                    dropout = config['dropout']
                    gamma = config.get('gamma', 0.0)
                    seed = config['seed']
                    lambda_byol_val = config['lambda_byol'] if byol else 0.0

                    train_loader, test_loader, input_dim = loader_construction(tmp_file)

                    best_epoch, min_loss, best_z_test, best_y_test, best_x_imp_test, best_l1, best_pccs, _ = train_model(
                        train_loader,
                        test_loader,
                        input_dim,
                        graph_head,
                        phi,
                        gcn_dim,
                        mlp_dim,
                        prob_feature,
                        prob_edge,
                        tau,
                        alpha,
                        beta,
                        lambda_cl,
                        dropout,
                        lr,
                        seed,
                        epochs,
                        os.path.join(UPLOAD_ROOT, f"model_{workflow_id}.pth"),
                        device,
                        knn_k=15,
                        phi1=config.get('phi1'),
                        n_clusters=clusters,
                        gamma=gamma,
                        lambda_byol=lambda_byol_val,
                    )

                    results = test_cluster([best_z_test], [best_y_test], 0, clusters, seed)
                    z_mat = np.vstack(best_z_test)
                    y_vec = np.hstack(best_y_test)
                    embed_path, label_path = _persist_embedding(st.session_state.username, workflow_id, z_mat, y_vec)

                record = {
                    "WorkflowId": workflow_id,
                    "Workflow": wf_name,
                    "Dataset": data_choice,
                    "Epochs": epochs,
                    "Best Epoch": int(best_epoch) if best_epoch is not None else None,
                    "LR": lr,
                    "Clusters": clusters,
                    "BYOL": "On" if byol else "Off",
                    "Status": "已完成",
                    "Created At": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "CA": round(float(results.get("CA", 0)), 2) if results.get("CA") is not None else None,
                    "NMI": round(float(results.get("NMI", 0)), 2) if results.get("NMI") is not None else None,
                    "ARI": round(float(results.get("ARI", 0)), 2) if results.get("ARI") is not None else None,
                    "embed_path": embed_path,
                    "label_path": label_path,
                }

                st.session_state.workflows[row_index] = record
                st.session_state.active_workflow = record
                st.session_state.last_completed_workflow = record
                st.session_state.best_embedding = {
                    "z": [z_mat],
                    "y": [y_vec],
                    "epoch": best_epoch,
                    "clusters": clusters,
                }
                upsert_workflow_mysql(st.session_state.username, record)
                st.success("训练完成，已生成真实嵌入并写入数据库。")
                _go("pages/2_Explorer.py")
            except Exception as e:
                failed_record = dict(st.session_state.workflows[row_index])
                failed_record["Status"] = "失败"
                st.session_state.workflows[row_index] = failed_record
                upsert_workflow_mysql(st.session_state.username, failed_record)
                st.error(f"训练失败：{e}")
            finally:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except Exception:
                        pass
                st.session_state.training_in_progress = False

with col_summary:
    st.markdown("<div class='label-small'>配置摘要</div>", unsafe_allow_html=True)
    wf_val = st.session_state.get("wf_name", "--")
    data_val = st.session_state.get("wf_data", "--")
    epochs_val = st.session_state.get("wf_epochs", "--")
    lr_val = st.session_state.get("wf_lr", "--")
    clus_val = st.session_state.get("wf_clusters", "--")
    byol_val = "On" if st.session_state.get("wf_byol", True) else "Off"
    tau_val = st.session_state.get("wf_tau", config['tau'])
    lambda_byol_val = st.session_state.get("wf_lambda_byol", config['lambda_byol'])

    st.markdown(
        f"""
        <div class='card'>
          <div style='font-weight:700;margin-bottom:6px;'>Workflow Summary</div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
                        <span style='color:#6b7280;'>名称</span><span style='font-weight:600;color:#111827;'>{wf_val}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
                        <span style='color:#6b7280;'>数据</span><span style='color:#111827;'>{data_val}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
                        <span style='color:#6b7280;'>轮数</span><span style='color:#111827;'>{epochs_val}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
            <span style='color:#6b7280;'>LR</span><span style='color:#111827;'>{lr_val}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
                        <span style='color:#6b7280;'>簇数</span><span style='color:#111827;'>{clus_val}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
            <span style='color:#6b7280;'>BYOL</span><span style='color:#111827;'>{byol_val}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
            <span style='color:#6b7280;'>τ (tau)</span><span style='color:#111827;'>{tau_val:.2f}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #f1f5f9;'>
            <span style='color:#6b7280;'>λ_BYOL</span><span style='color:#111827;'>{lambda_byol_val:.2f}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
st.caption("提示：右侧卡片实时展示当前输入或最近一次完成的工作流配置。")
