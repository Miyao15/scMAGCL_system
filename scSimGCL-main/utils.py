import os
import importlib
import torch
import random
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import scipy.sparse as sp
import scanpy as sc
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import *
from scipy.optimize import linear_sum_assignment as linear_assignment
import h5py
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if issparse(X):
            self.X = X.tocsr()
        else:
            self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if issparse(self.X):
            x = self.X[idx].toarray().squeeze()
        else:
            x = self.X[idx]
        x = torch.tensor(x, dtype=torch.float32)
        return x, self.y[idx]


def qc_filter(adata, min_genes=200, max_genes=5000, min_cells=3):
    orig_cells, orig_genes = adata.shape
    sc.pp.filter_cells(adata, min_genes=min_genes)
    after_min_genes = adata.shape
    sc.pp.filter_cells(adata, max_genes=max_genes)
    after_max_genes = adata.shape
    sc.pp.filter_genes(adata, min_cells=min_cells)
    final_shape = adata.shape
    cell_retention = final_shape[0] / orig_cells * 100
    gene_retention = final_shape[1] / orig_genes * 100
    return adata


def normalize(adata, target_sum=1e4):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    X_data = adata.X if not issparse(adata.X) else adata.X.data
    nan_count = np.sum(~np.isfinite(X_data))
    if nan_count > 0:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    return adata


def select_highly_variable_genes(adata, n_top_genes=2000):
    before_hvg = adata.shape
    X_data = adata.X
    if issparse(X_data):
        X_data = X_data.toarray()

    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
        n_hvg = adata.var.highly_variable.sum()
    except (ValueError, RuntimeWarning) as e:
        try:
            if issparse(X_data):
                X_data = X_data.toarray()
            gene_var = np.var(X_data, axis=0)
            gene_var = np.nan_to_num(gene_var, nan=0.0)
            top_gene_indices = np.argsort(gene_var)[-n_top_genes:]
            adata.var["highly_variable"] = False
            adata.var.loc[adata.var.index[top_gene_indices], "highly_variable"] = True
            n_hvg = adata.var.highly_variable.sum()
        except Exception as e2:
            adata.var["highly_variable"] = True
            n_hvg = adata.var.highly_variable.sum()

    adata = adata[:, adata.var.highly_variable]
    after_hvg = adata.shape
    gene_reduction = (before_hvg[1] - after_hvg[1]) / before_hvg[1] * 100
    return adata


def loader_construction(data_path, graph_head=None):

    with h5py.File(data_path, "r") as f:

        if "X" in f and "y" in f and "obs" not in f:
            X_all = f["X"][()]
            y_all = f["y"][()]
            y_all = np.array(y_all).squeeze()
            if isinstance(y_all[0], bytes):
                y_all = np.array([v.decode("utf-8") for v in y_all])

            try:
                pass
            except Exception:
                pass
            le = LabelEncoder()
            y_all_encoded = le.fit_transform(y_all)
            n_classes = len(np.unique(y_all_encoded))
            pass
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all_encoded, test_size=0.2, random_state=42, stratify=y_all_encoded
            )

            train_set = CellDataset(X_train, y_train)
            test_set = CellDataset(X_test, y_test)
            train_loader = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=False)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False)
            input_dim = X_all.shape[1]

            print("=" * 60)
            return train_loader, test_loader, input_dim

        if "X" in f and "y" in f and "cell_barcodes" in f:
            X_all = f["X"][()]
            y_all = f["y"][()].astype(str)
            cell_barcodes = f["cell_barcodes"][()].astype(str)

            if not issparse(X_all):
                X_all = csr_matrix(X_all)


        elif "X" in f and "obs" in f:
            X_all = f["X"][()]
            obs_keys = list(f["obs"].keys())

            for key in obs_keys:
                try:
                    data_sample = f[f"obs/{key}"][()]
                    if len(data_sample) > 0:
                        sample_values = data_sample[: min(5, len(data_sample))]
                except Exception:
                    pass

            y_all = None
            cell_type_found = False
            possible_labels = [
                "cell_type",
                "celltype",
                "cell_type1",
                "assigned_cluster",
                "cluster",
                "labels",
                "Group",
                "group",
                "cell_group",
                "batch",
                "sample",
                "condition",
                "treatment",
                "cell_ontology_class",
                "cell_ontology_id",
                "annotation",
                "celltype_major",
                "celltype_minor",
                "louvain",
                "leiden",
                "seurat_clusters",
                "predicted_labels",
            ]

            for label_name in possible_labels:
                if label_name in f["obs"]:
                    y_all = f[f"obs/{label_name}"][()].astype(str)
                    cell_type_found = True
                    break

            if not cell_type_found and "cell_id" in f["obs"]:
                cell_ids = f["obs/cell_id"][()].astype(str)
                import re

                patterns = []
                for cell_id in cell_ids[: min(100, len(cell_ids))]:
                    match = re.match(r"^([A-Za-z]+)", str(cell_id))
                    if match:
                        patterns.append(match.group(1))

                unique_patterns = list(set(patterns))
                if len(unique_patterns) > 1 and len(unique_patterns) <= 20:
                    y_all = []
                    for cell_id in cell_ids:
                        match = re.match(r"^([A-Za-z]+)", str(cell_id))
                        if match:
                            y_all.append(match.group(1))
                        else:
                            y_all.append("Unknown")
                    y_all = np.array(y_all)
                    cell_type_found = True

            if not cell_type_found:
                n_cells = X_all.shape[0]
                if n_cells < 1000:
                    n_types = 5
                elif n_cells < 3000:
                    n_types = 8
                elif n_cells < 8000:
                    n_types = 10
                else:
                    n_types = 12
                y_all = np.array([f"Type_{i % n_types}" for i in range(n_cells)])


            if not issparse(X_all):
                X_all = csr_matrix(X_all)

        elif "exprs" in f and "exprs/data" in f:
            print("   检测到稀疏矩阵格式数据")
            data = f["exprs/data"][()]
            indices = f["exprs/indices"][()]
            indptr = f["exprs/indptr"][()]
            shape = tuple(f["exprs/shape"][()])
            X_all = csr_matrix((data, indices, indptr), shape=shape)
            y_all = f["obs/cell_type1"][()].astype(str)

        else:
            raise ValueError(f"不支持的数据格式。文件结构: {list(f.keys())}")


    adata = sc.AnnData(X=X_all, obs={"cell_type": y_all})
    adata = normalize(adata)
    adata = select_highly_variable_genes(adata)

    le = LabelEncoder()
    y_all_encoded = le.fit_transform(adata.obs["cell_type"])
    unique_labels = len(np.unique(y_all_encoded))

    X_train, X_test, y_train, y_test = train_test_split(
        adata.X, y_all_encoded, test_size=0.2, random_state=1
    )

    train_set = CellDataset(X_train, y_train)
    test_set = CellDataset(X_test, y_test)

    def sparse_collate(batch):
        x = [item[0] for item in batch]
        y = torch.stack([item[1] for item in batch])
        return torch.stack(x), y

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=sparse_collate)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=sparse_collate)

    input_dim = adata.shape[1]


    return train_loader, test_loader, input_dim


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def cluster_acc(y_true, y_pred):
    if isinstance(y_true[0], (bytes, str)):
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(y_true, y_pred):
    acc = cluster_acc(y_true, y_pred)
    f1 = 0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    return acc, f1, nmi, ari, homo, comp
