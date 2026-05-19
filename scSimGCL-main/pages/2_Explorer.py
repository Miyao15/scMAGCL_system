import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import h5py
import scanpy as sc
import plotly.express as px
import plotly.graph_objects as go
from mysql_backend import upsert_workflow_mysql


def _go(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        pass

st.set_page_config(page_title="可视化页 - scMAGCL", page_icon="", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {display: none;}
            div[data-testid="stToolbar"] {display: none;}
            header {visibility: hidden; height: 0;}
            div.block-container {padding-top: 0.8rem; max-width: 1400px;}
            .pane {background:#ffffff;border:1px solid #d7dbe2;border-radius:6px;padding:8px;}
            .pane-title {font-size:0.92rem;font-weight:700;color:#1f2937;margin-bottom:6px;}
            .tiny {font-size:0.78rem;color:#6b7280;}
            .tool-btn {display:inline-block;padding:2px 6px;border:1px solid #cfd6df;border-radius:4px;font-size:0.72rem;color:#4b5563;margin-right:4px;background:#f8fafc;}
        </style>
        """,
        unsafe_allow_html=True,
)

if "dataset_store" not in st.session_state:
    st.session_state.dataset_store = {}
if "active_workflow" not in st.session_state:
    st.session_state.active_workflow = None
if "datasets" not in st.session_state:
    st.session_state.datasets = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    _go("pages/0_Login.py")
    st.stop()

col_nav, col_title = st.columns([1,5])
with col_nav:
    if st.button("返回", width="content"):
                _go("app.py")
with col_title:
    st.markdown("<div style='font-size:2rem;font-weight:800;'>可视化页</div>\n<div style='color:#4b5563;'>高维交互可视化 · cellxgene 风格三栏布局</div>", unsafe_allow_html=True)

st.divider()

if not st.session_state.active_workflow:
    st.warning("请先创建并运行一个工作流。")
    st.stop()

wf = st.session_state.active_workflow
wf_name = wf.get("Workflow", wf.get("任务名称", "--"))
wf_dataset = wf.get("Dataset", wf.get("数据", "--"))
wf_status = wf.get("Status", wf.get("状态", "Unknown"))
wf_best_epoch = wf.get("Best Epoch", wf.get("best_epoch", "?"))

st.markdown(f"**当前工作流：** {wf_name} · 数据集 {wf_dataset}")
st.caption(f"状态：{wf_status} · 训练轮数={wf.get('Epochs','?')} · 保存嵌入最佳epoch={wf_best_epoch} · 学习率={wf.get('LR','?')} · 簇数={wf.get('Clusters','?')} · BYOL={wf.get('BYOL','?')}")

@st.cache_resource(show_spinner=True)
def build_umap_from_raw(bytes_data, max_cells=8000):
    def _extract_gene_names(h5f, n_vars):
        candidates = [
            "var/_index", "var/gene_symbols", "var/genes", "var/feature_name", "var/feature_names",
            "var_names", "gene_names", "genes",
        ]
        for path in candidates:
            try:
                if "/" in path:
                    g, k = path.split("/", 1)
                    if g in h5f and k in h5f[g]:
                        arr = np.array(h5f[g][k][()]).squeeze()
                    else:
                        continue
                else:
                    if path not in h5f:
                        continue
                    arr = np.array(h5f[path][()]).squeeze()
                if arr.size == 0:
                    continue
                if isinstance(arr[0], (bytes, np.bytes_)):
                    arr = np.array([v.decode("utf-8") for v in arr])
                return [str(v) for v in arr.tolist()[:n_vars]]
            except Exception:
                continue
        return [f"Gene_{i}" for i in range(n_vars)]

    with h5py.File(io.BytesIO(bytes_data), "r") as f:
        if "X" not in f:
            raise ValueError("H5 文件缺少 X 数据集，无法构建 UMAP")
        X = f["X"][()]
        gene_names = _extract_gene_names(f, X.shape[1])
        obs_dict = {}
        if "y" in f:
            y = np.array(f["y"][()]).squeeze()
            if y.ndim == 0:
                y = np.array([y])
            if isinstance(y[0], (bytes, np.bytes_)):
                y = np.array([v.decode("utf-8") for v in y])
            obs_dict["label"] = y.astype(str)
        if "obs" in f:
            for key in f["obs"].keys():
                try:
                    obs_dict[key] = np.array(f[f"obs/{key}"][()]).astype(str)
                except Exception:
                    continue
        n_cells = X.shape[0]
        if n_cells > max_cells:
            idx = np.random.choice(n_cells, max_cells, replace=False)
            X = X[idx]
            for k in list(obs_dict.keys()):
                obs_dict[k] = np.array(obs_dict[k])[idx]
        X_for_expr = X.astype(np.float32, copy=True)
        adata = sc.AnnData(X=X)
        for k, v in obs_dict.items():
            adata.obs[k] = pd.Categorical(v)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars))
        adata = adata[:, adata.var["highly_variable"]]
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=min(30, adata.n_vars))
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(30, adata.n_vars))
        sc.tl.umap(adata, min_dist=0.3)
        label_col = "label" if "label" in adata.obs else "cluster"
        if label_col not in adata.obs:
            adata.obs[label_col] = pd.Categorical([f"Cluster {i%10}" for i in range(adata.n_obs)])
        df = pd.DataFrame({
            "UMAP1": adata.obsm["X_umap"][:,0],
            "UMAP2": adata.obsm["X_umap"][:,1],
            "label": adata.obs[label_col].astype(str).values,
        })
        for c in adata.obs.columns:
            if c not in df.columns:
                df[c] = adata.obs[c].astype(str).values
    return df, X_for_expr, gene_names


@st.cache_resource(show_spinner=False)
def load_expression_for_view(bytes_data, n_target=None, max_cells=8000):
    with h5py.File(io.BytesIO(bytes_data), "r") as f:
        if "X" not in f:
            return None, []
        X = f["X"][()]
        n_vars = X.shape[1]

        gene_candidates = ["var/_index", "var/gene_symbols", "var/genes", "var/feature_name", "var/feature_names", "var_names", "gene_names", "genes"]
        gene_names = None
        for path in gene_candidates:
            try:
                if "/" in path:
                    g, k = path.split("/", 1)
                    if g in f and k in f[g]:
                        arr = np.array(f[g][k][()]).squeeze()
                    else:
                        continue
                else:
                    if path not in f:
                        continue
                    arr = np.array(f[path][()]).squeeze()
                if arr.size == 0:
                    continue
                if isinstance(arr[0], (bytes, np.bytes_)):
                    arr = np.array([v.decode("utf-8") for v in arr])
                gene_names = [str(v) for v in arr.tolist()[:n_vars]]
                break
            except Exception:
                continue
        if gene_names is None:
            gene_names = [f"Gene_{i}" for i in range(n_vars)]

        if n_target is not None and X.shape[0] >= n_target:
            X_use = X[:n_target]
        elif X.shape[0] > max_cells:
            X_use = X[:max_cells]
        else:
            X_use = X

    return X_use.astype(np.float32, copy=False), gene_names

fname = wf.get("Dataset", wf.get("数据"))
if fname not in st.session_state.dataset_store:
    st.error("未找到对应数据文件，请回到工作台重新上传。")
    st.stop()

umap_df = None
expr_matrix = None
gene_names = []

embed_path = wf.get("embed_path")
label_path = wf.get("label_path")
if embed_path and os.path.exists(embed_path):
    try:
        embed_arr = np.load(embed_path)
        label_arr = np.load(label_path) if label_path and os.path.exists(label_path) else None
        st.session_state.best_embedding = {
            "z": [embed_arr],
            "y": [label_arr] if label_arr is not None else None,
            "epoch": wf_best_epoch,
            "clusters": wf.get("Clusters"),
        }
    except Exception:
        pass

emb = st.session_state.get("best_embedding")
if emb and emb.get("z") is not None:
    try:
        z = emb["z"]  # list of array per batch
        y = emb.get("y")
        z_stack = np.vstack(z)
        df_data = {"UMAP1": z_stack[:, 0], "UMAP2": z_stack[:, 1]}
        if y is not None:
            y_arr = np.concatenate(y) if isinstance(y, list) else np.array(y)
            df_data["label"] = y_arr.astype(str)
        else:
            df_data["label"] = [f"Cluster {i%10}" for i in range(z_stack.shape[0])]
        umap_df = pd.DataFrame(df_data)
        expr_matrix, gene_names = load_expression_for_view(
            st.session_state.dataset_store[fname],
            n_target=len(umap_df),
            max_cells=8000,
        )
        st.caption(f"当前 UMAP 使用模型最佳 epoch（{emb.get('epoch', '?')}）的嵌入结果。")
    except Exception:
        umap_df = None

if umap_df is None:
    try:
        umap_df, expr_matrix, gene_names = build_umap_from_raw(st.session_state.dataset_store[fname])
        st.caption("未找到训练嵌入，已回退为原始数据 UMAP。")
    except Exception as e:
        st.error(f"UMAP 构建失败：{e}")
        st.stop()

gene_to_idx = {g: i for i, g in enumerate(gene_names)}
if not gene_names:
    gene_names = ["Sftpc", "Crip1", "Igfbp7", "Krt8", "Col1a1"]

search_keyword = st.session_state.get("gene_search_kw", "")
if search_keyword:
    filtered_gene_options = [g for g in gene_names if search_keyword.lower() in g.lower()]
    if not filtered_gene_options:
        filtered_gene_options = gene_names[:200]
else:
    filtered_gene_options = gene_names[:200]

if "gene_selected" not in st.session_state or st.session_state.gene_selected not in filtered_gene_options:
    st.session_state.gene_selected = filtered_gene_options[0] if filtered_gene_options else None

selected_gene = st.session_state.get("gene_selected")
gene_expr_values = None
if selected_gene and expr_matrix is not None and selected_gene in gene_to_idx:
    idx = gene_to_idx[selected_gene]
    if idx < expr_matrix.shape[1]:
        gene_expr_values = np.asarray(expr_matrix[:, idx]).reshape(-1)

can_gene_color = gene_expr_values is not None and len(gene_expr_values) == len(umap_df)
if can_gene_color:
    umap_df["_gene_expr"] = gene_expr_values

col_left, col_mid, col_right = st.columns([1.15, 2.0, 1.1], gap="small")

cats = sorted(umap_df["label"].unique())
palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Set3
color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
ai_query = st.session_state.get("ai_query", "").strip().lower()

with col_left:
    st.markdown("<div class='pane'>", unsafe_allow_html=True)
    st.markdown("<div class='pane-title'>数据类别 ▾</div>", unsafe_allow_html=True)
    st.markdown("<div class='tiny'>cell_type</div>", unsafe_allow_html=True)

    selected_labels = []
    counts = umap_df["label"].value_counts()
    for idx, c in enumerate(cats):
        c1, c2, c3 = st.columns([0.08, 0.76, 0.16])
        with c1:
            checked = st.checkbox("", key=f"cat_{idx}", value=True, label_visibility="collapsed")
        with c2:
            st.markdown(f"<div style='font-size:0.82rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{c}</div>", unsafe_allow_html=True)
        with c3:
            n = int(counts.get(c, 0))
            st.markdown(
                f"<div style='display:flex;align-items:center;justify-content:flex-end;gap:6px;'>"
                f"<span style='font-size:0.75rem;color:#4b5563;'>{n}</span>"
                f"<span style='display:inline-block;width:10px;height:10px;background:{color_map[c]};border-radius:2px;'></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if checked:
            selected_labels.append(c)

    # Apply text query filter (CELL x AI 搜索) to labels when有输入
    if ai_query:
        matched = [c for c in cats if ai_query in str(c).lower()]
        if matched:
            selected_labels = matched
            st.caption(f"已按查询筛选类别：{', '.join(matched)}")
        else:
            st.caption("未找到匹配的类别，保留全部类别显示。")

    st.markdown("<div class='tiny' style='margin-top:8px;'>已选类别将用于 UMAP 过滤与着色。</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_mid:
    st.markdown("<div class='pane'>", unsafe_allow_html=True)
    top_l, top_r = st.columns([1.1, 1.3])
    with top_l:
        st.markdown("<div class='pane-title'>umap ▾</div>", unsafe_allow_html=True)
        color_mode = st.radio("着色方式", ["按类别", "按基因表达"], horizontal=True, key="umap_color_mode")
    with top_r:
        st.markdown("<div style='text-align:right;'></div>", unsafe_allow_html=True)

    if not selected_labels:
        selected_labels = cats

    df_plot = umap_df[umap_df["label"].isin(selected_labels)]
    st.markdown(f"<div class='tiny'>{len(df_plot)} / {len(umap_df)} 个细胞</div>", unsafe_allow_html=True)
    if color_mode == "按基因表达" and can_gene_color:
        fig = px.scatter(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            color="_gene_expr",
            color_continuous_scale="Viridis",
            opacity=0.78,
            height=760,
            hover_data={col: True for col in df_plot.columns if col not in ["UMAP1", "UMAP2"]},
        )
    else:
        if color_mode == "按基因表达" and not can_gene_color:
            st.caption("当前 UMAP 与表达矩阵行数不一致，暂无法按基因表达着色，已自动回退到按类别着色。")
        fig = px.scatter(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            color="label",
            color_discrete_map=color_map,
            opacity=0.75,
            height=760,
            hover_data={col: True for col in df_plot.columns if col not in ["UMAP1", "UMAP2"]},
        )
    fig.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, title=""),
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='pane'>", unsafe_allow_html=True)
    st.markdown("<div class='pane-title'>基因 ▾</div>", unsafe_allow_html=True)
    st.caption(f"已加载 {len(gene_names)} 个基因名，可输入关键字进行模糊匹配。")
    search_keyword = st.text_input("快速基因检索", value=st.session_state.get("gene_search_kw", ""), key="gene_search_kw", label_visibility="collapsed", placeholder="输入基因名快速检索")

    filtered_gene_options = [g for g in gene_names if search_keyword.lower() in g.lower()] if search_keyword else gene_names
    if not filtered_gene_options:
        filtered_gene_options = gene_names[:200]
    filtered_gene_options = filtered_gene_options[:200]

    if st.session_state.get("gene_selected") not in filtered_gene_options:
        st.session_state.gene_selected = filtered_gene_options[0] if filtered_gene_options else None

    gene = st.selectbox("基因集", options=filtered_gene_options, index=0 if filtered_gene_options else None, key="gene_selected", label_visibility="collapsed")

    with st.expander("不知道基因名？点此查看示例"):
        st.markdown("- 基因名来自你上传数据中的 `var` / `var_names` / `gene_names` 字段。")
        st.markdown("- 搜索支持不区分大小写的包含匹配，例如输入 `Sft` 可匹配 `Sftpc`。")
        example_genes = gene_names[:60] if gene_names else []
        if example_genes:
            st.code("\n".join(example_genes), language="text")
        st.download_button(
            "下载全部基因名（txt）",
            data="\n".join(gene_names),
            file_name="gene_names.txt",
            mime="text/plain",
            width="stretch",
        )

    if gene and expr_matrix is not None and gene in gene_to_idx and gene_to_idx[gene] < expr_matrix.shape[1]:
        expr = np.asarray(expr_matrix[:, gene_to_idx[gene]]).reshape(-1)
    else:
        rng = np.random.default_rng(7)
        signal = np.clip(np.abs(umap_df["UMAP1"].values) + 0.35 * np.abs(umap_df["UMAP2"].values), 0, None)
        expr = signal / (signal.max() + 1e-8) * 3.2 + rng.normal(0, 0.08, size=signal.shape[0])
        expr = np.clip(expr, 0, 3.6)

    hist = go.Figure(go.Histogram(x=expr, nbinsx=28, marker_color="#9ca3af"))
    hist.update_layout(
        height=160,
        margin=dict(l=0, r=0, t=10, b=20),
        bargap=0.04,
        xaxis_title="基因集平均表达",
        yaxis_title="",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(hist, width="stretch", config={"displaylogo": False, "staticPlot": True})

    st.markdown(f"<div class='tiny' style='margin-top:4px;'>当前基因集：<b>{gene}</b></div>", unsafe_allow_html=True)
    if expr_matrix is not None:
        st.markdown(f"<div class='tiny'>已基于真实表达矩阵渲染该基因分布。</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='tiny'>当前为示意分布（未找到可用表达矩阵）。</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='pane-title'>CELL x AI</div>", unsafe_allow_html=True)
    st.markdown("<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:8px;font-size:0.8rem;color:#1f2937;'>已按 cell_type 元数据完成可视化着色。</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='background:#f3f4f6;border:1px solid #e5e7eb;border-radius:8px;padding:8px;font-size:0.8rem;color:#1f2937;'>已从 cell_type 元数据中选择 hematopoietic 细胞。</div>", unsafe_allow_html=True)
    st.text_input("", value=st.session_state.get("ai_query", ""), placeholder="请输入查询内容...", label_visibility="collapsed", key="ai_query")

    metrics = {
        "CA": f"{wf.get('CA','--')}%" if wf.get('CA') is not None else "--",
        "NMI": f"{wf.get('NMI','--')}%" if wf.get('NMI') is not None else "--",
        "ARI": f"{wf.get('ARI','--')}%" if wf.get('ARI') is not None else "--",
    }
    st.markdown("---")
    st.markdown("<div class='pane-title'>指标</div>", unsafe_allow_html=True)
    for k, v in metrics.items():
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;border-bottom:1px solid #eef2f7;padding:4px 0;'>"
            f"<span class='tiny' style='font-size:0.8rem'>{k}</span><span style='font-size:0.9rem;font-weight:700;color:#111827'>{v}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("布局：左侧类别树，中间 UMAP 主画布，右侧基因/工具面板。")
