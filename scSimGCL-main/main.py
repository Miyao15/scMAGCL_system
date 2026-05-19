import argparse
import warnings
import importlib
import torch
import numpy as np
import pandas as pd
import os
import gc
from utils import setup_seed, loader_construction, evaluate, device
from scMAGCL import Model
from sklearn.cluster import KMeans
from config import config
from scipy.stats import pearsonr
import time
try:
    import psutil
except Exception:
    psutil = None

if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)





def train(train_loader,
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
          save_model_path,
          device,
          knn_k=15,
          phi1: float = None,
          n_clusters: int = 0,
          gamma: float = 0.0,
          lambda_byol: float = 1.0):
    setup_seed(seed)
    model = Model(input_dim=input_dim, graph_head=graph_head, phi=phi, gcn_dim=gcn_dim,
                  mlp_dim=mlp_dim, prob_feature=prob_feature, prob_edge=prob_edge, tau=tau,
                  alpha=alpha, beta=beta, dropout=dropout,
                  phi1=(config['phi1'] if phi1 is None else phi1),
                  use_dec=True, cluster_num=n_clusters,
                  use_adaptive_aug=config['use_adaptive_aug'],
                  use_semantic_preserve=config['use_semantic_preserve'],
                  preserve_ratio=config['preserve_ratio'],
                  base_prob_feature=config['base_prob_feature'],
                  base_prob_edge=config['base_prob_edge'],
                  adaptive_noise_std=config['adaptive_noise_std'],
                  use_byol=config['use_byol'],
                  byol_hidden_dim=config['byol_hidden_dim'],
                  byol_output_dim=config['byol_output_dim'],
                  momentum_tau=config['momentum_tau']).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    
    test_loss = []
    best_epoch = 0
    min_loss = 999

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    best_z_test = None
    best_y_test = None
    best_x_imp_test = None

    print(f"开始训练 {epochs} 个epoch...")
    
    for each_epoch in range(epochs):
        batch_loss = []
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_z, x_imp, loss_cl, loss_dec, loss_byol = model(batch_x)
            mask = torch.where(batch_x != 0, torch.ones(batch_x.shape).to(device),
                               torch.zeros(batch_x.shape).to(device))
            mae_f = torch.nn.L1Loss(reduction='mean')
            loss_mae = mae_f(mask * x_imp, mask * batch_x)
            loss = loss_mae + lambda_cl * loss_cl + lambda_byol * loss_byol
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()
            model.update_target_network()
            batch_loss.append(loss.cpu().detach().numpy())

        with torch.no_grad():
            model.eval()
            z_test = []
            y_test = []
            x_imp_test = []
            batch_loss = []
            for step, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_z, x_imp, loss_cl, loss_dec, loss_byol = model(batch_x)
                mask = torch.where(batch_x != 0, torch.ones(batch_x.shape).to(device),
                                   torch.zeros(batch_x.shape).to(device))
                loss_mae = mae_f(mask * x_imp, mask * batch_x)
                loss = loss_mae + lambda_cl * loss_cl + lambda_byol * loss_byol
                z_test.append(batch_z.cpu().detach().numpy())
                y_test.append(batch_y)
                x_imp_test.append(x_imp.cpu().detach().numpy())
                batch_loss.append(loss.cpu().detach().numpy())

        cur_loss = np.mean(np.array(batch_loss))
        test_loss.append(cur_loss)

        if cur_loss < min_loss:
            min_loss = cur_loss
            best_epoch = each_epoch
            best_z_test = z_test.copy()
            best_y_test = y_test.copy()
            best_x_imp_test = x_imp_test.copy()
            print(f"Epoch {each_epoch:2d}/{epochs-1}: 测试损失={cur_loss:.4f} (新最佳)")
        else:
            print(f"Epoch {each_epoch:2d}/{epochs-1}: 测试损失={cur_loss:.4f} (最佳: {min_loss:.4f})")
        
        del z_test, y_test, x_imp_test, batch_loss
        if (each_epoch + 1) % 10 == 0:
            gc.collect()

    print(f"训练完成! 最佳epoch: {best_epoch}, 最佳损失: {min_loss:.4f}")
    del model, opt_model, test_loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for _ in range(3):
        gc.collect()
    
    return best_epoch, min_loss, best_z_test, best_y_test, best_x_imp_test, None, None, None
    

def test(z_test_epoch,
         y_test_epoch,
         best_epoch,
         n_clusters,
         seed):
    z_test = z_test_epoch[best_epoch]
    y_test = y_test_epoch[best_epoch]
    z_test = np.vstack(z_test)
    y_test = np.hstack(y_test)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_test)
    y_kmeans_test = kmeans.labels_
    acc, f1, nmi, ari, homo, comp = evaluate(y_test, y_kmeans_test)
    results = {'CA': acc, 'NMI': nmi, 'ARI': ari}
    try:
        del kmeans, z_test, y_test, y_kmeans_test
    except:
        pass
    gc.collect()
    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_head", type=int, default=config['graph_head'])
    parser.add_argument("--phi", type=float, default=config['phi'])
    parser.add_argument("--gcn_dim", type=int, default=config['gcn_dim'])
    parser.add_argument("--mlp_dim", type=int, default=config['mlp_dim'])
    parser.add_argument("--prob_feature", type=float, default=config['prob_feature'])
    parser.add_argument("--prob_edge", type=float, default=config['prob_edge'])
    parser.add_argument("--tau", type=float, default=config['tau'])
    parser.add_argument("--alpha", type=float, default=config['alpha'])
    parser.add_argument("--beta", type=float, default=config['beta'])
    parser.add_argument("--lambda_cl", type=float, default=config['lambda_cl'])
    parser.add_argument("--dropout", type=float, default=config['dropout'])
    parser.add_argument("--knn_k", type=int, default=15, help="Number of nearest neighbors for KNN graph")
    parser.add_argument("--n_clusters", type=int, default=9, help="Number of clusters")
    parser.add_argument("--gamma", type=float, default=config.get('gamma', 0.0), help="Weight for KL loss")
    parser.add_argument("--lambda_byol", type=float, default=config['lambda_byol'], help="Weight for BYOL loss")
    parser.add_argument("--phi1", type=float, default=config['phi1'], help="Multi-scale graph threshold margin (phi1)")
    # 已移除消融实验参数
    parser.add_argument("--lr", type=float, default=config['lr'])
    parser.add_argument("--seed", type=int, default=config['seed'])
    parser.add_argument("--epochs", type=int, default=config['epochs'])
    parser.add_argument("--n_runs", type=int, default=10, help="Number of repeated runs per setting")
    parser.add_argument("--sweep_param", type=str, default=None,
                        help="Sweep one of: phi1, tau, lambda_cl, lambda_byol (requires --sweep_values)")
    parser.add_argument("--sweep_values", type=str, default=None,
                        help="Comma-separated values for sweep_param, e.g. '0,0.2,0.4'")
    parser.add_argument("--data_path", type=str, default="D://data2//Zeisel.h5", help="Path to data file")
    parser.add_argument("--save_model_path", type=str, default="./model.pth", help="Path to save model")
    args = parser.parse_args()

    graph_head = args.graph_head
    phi = args.phi
    gcn_dim = args.gcn_dim
    mlp_dim = args.mlp_dim
    prob_feature = args.prob_feature
    prob_edge = args.prob_edge
    tau = args.tau
    alpha = args.alpha
    beta = args.beta
    lambda_cl = args.lambda_cl
    dropout = args.dropout
    knn_k = args.knn_k
    n_clusters = args.n_clusters
    gamma = args.gamma
    lambda_byol = args.lambda_byol
    phi1_cli = args.phi1
    # 已移除消融实验参数
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    n_runs = args.n_runs
    sweep_param = args.sweep_param
    sweep_values_str = args.sweep_values
    data_path = args.data_path
    save_model_path = args.save_model_path

    if sweep_param is not None and sweep_values_str is not None:
        sweep_values = [float(v) for v in sweep_values_str.split(",") if v.strip() != ""]
    else:
        sweep_param = None
        sweep_values = [None]

    for sweep_val in sweep_values:
        phi1_current = phi1_cli
        tau_current = tau
        lambda_cl_current = lambda_cl
        lambda_byol_current = lambda_byol
        setting_tag = "base"
        if sweep_param is not None and sweep_val is not None:
            if sweep_param == "phi1":
                phi1_current = sweep_val
                setting_tag = f"phi1_{sweep_val}"
            elif sweep_param == "tau":
                tau_current = sweep_val
                setting_tag = f"tau_{sweep_val}"
            elif sweep_param == "lambda_cl":
                lambda_cl_current = sweep_val
                setting_tag = f"lambda_cl_{sweep_val}"
            elif sweep_param == "lambda_byol":
                lambda_byol_current = sweep_val
                setting_tag = f"lambda_byol_{sweep_val}"
            else:
                raise SystemExit(
                    f"Unknown --sweep_param {sweep_param!r}; use one of: phi1, tau, lambda_cl, lambda_byol"
                )


        results_list = []
        # l1_list = []
        # pccs_list = []
        runtime_list = []
        gpu_mem_list = []
        cpu_mem_list = []
        best_overall_score = -1.0
        best_overall_index = -1
        best_overall_z_path = None
        best_overall_y_path = None

        for i in range(n_runs):

            if psutil is not None:
                pass

            start_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid()) if psutil is not None else None

            cur_seed = seed + i
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            train_loader, test_loader, input_dim = loader_construction(data_path)

            best_epoch, min_loss, best_z_test, best_y_test, best_x_imp_test, _, _, _ = train(
                train_loader, test_loader, input_dim, graph_head, phi, gcn_dim, mlp_dim, prob_feature, prob_edge,
                tau_current, alpha, beta, lambda_cl_current, dropout, lr, cur_seed, epochs, f"{save_model_path}_{setting_tag}_{i}.pth", device, 
                knn_k, phi1=phi1_current, n_clusters=n_clusters, gamma=gamma, lambda_byol=lambda_byol_current)

            results = test([best_z_test], [best_y_test], 0, n_clusters, cur_seed)
            results_list.append(results)
            gc.collect()

            try:
                z_test_mat = np.vstack(best_z_test)
                y_test_vec = np.hstack(best_y_test)
                import os
                dataset_name = os.path.splitext(os.path.basename(data_path))[0]
                emb_dir = os.path.join("embeddings", dataset_name, setting_tag)
                os.makedirs(emb_dir, exist_ok=True)
                run_id = f"run{cur_seed}"
                z_path = os.path.join(emb_dir, f"{setting_tag}_{run_id}_z_test.npy")
                y_path = os.path.join(emb_dir, f"{setting_tag}_{run_id}_y_test.npy")
                np.save(z_path, z_test_mat)
                np.save(y_path, y_test_vec)
                pd.DataFrame(z_test_mat).to_csv(z_path.replace('.npy', '.csv'), index=False)
                pd.DataFrame(y_test_vec, columns=["label"]).to_csv(y_path.replace('.npy', '.csv'), index=False)
                cur_score = float(results["CA"])
                if cur_score > best_overall_score:
                    best_overall_score = cur_score
                    best_overall_index = i + 1
                    best_overall_z_path = z_path
                    best_overall_y_path = y_path
            except Exception as e:
                pass
            
            del results

            try:
                del best_z_test, best_y_test, best_x_imp_test
                del z_test_mat, y_test_vec
                del best_epoch, min_loss
                del train_loader, test_loader
            except Exception:
                pass
            
            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            try:
                duration_s = float(time.perf_counter() - start_time)
            except Exception:
                duration_s = float('nan')
            if torch.cuda.is_available():
                try:
                    gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                except Exception:
                    gpu_peak_mb = float('nan')
            else:
                gpu_peak_mb = float('nan')
            if process is not None:
                try:
                    cpu_rss_mb = float(process.memory_info().rss / (1024 ** 2))
                except Exception:
                    cpu_rss_mb = float('nan')
            else:
                cpu_rss_mb = float('nan')
            runtime_list.append(duration_s)
            gpu_mem_list.append(gpu_peak_mb)
            cpu_mem_list.append(cpu_rss_mb)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_results_with_stats = {}
        for key in ['CA', 'NMI', 'ARI']:
            values = [result[key] for result in results_list]
            avg_results_with_stats[f'avg_{key}'] = np.mean(values)
            avg_results_with_stats[f'std_{key}'] = np.std(values)
        
        try:
            if best_overall_z_path is not None and best_overall_y_path is not None:
                import shutil
                dataset_name = os.path.splitext(os.path.basename(data_path))[0]
                emb_dir = os.path.join("embeddings", dataset_name)
                best_z_dst = os.path.join(emb_dir, f"emb_best_{setting_tag}_z_test.npy")
                best_y_dst = os.path.join(emb_dir, f"emb_best_{setting_tag}_y_test.npy")
                shutil.copy(best_overall_z_path, best_z_dst)
                shutil.copy(best_overall_y_path, best_y_dst)
                shutil.copy(best_overall_z_path.replace('.npy', '.csv'), best_z_dst.replace('.npy', '.csv'))
                shutil.copy(best_overall_y_path.replace('.npy', '.csv'), best_y_dst.replace('.npy', '.csv'))
            else:
                print(f"设置 {setting_tag}：未能确定最佳表现 run，跳过最佳 embedding 保存。")
        except Exception as e:
            print(f"设置 {setting_tag}：保存最佳 embedding 失败: {e}")
        
        avg_results_with_params = avg_results_with_stats.copy()
        avg_results_with_params.update({
            'n_clusters': n_clusters,
            'lr': lr,
            'epochs': epochs,
            'knn_k': knn_k,
            'n_experiments': n_runs,
            'timestamp': pd.Timestamp.now(),
            'avg_runtime_s': float(np.nanmean(runtime_list)) if len(runtime_list)>0 else float('nan'),
            'std_runtime_s': float(np.nanstd(runtime_list)) if len(runtime_list)>0 else float('nan'),
            'avg_gpu_mem_mb': float(np.nanmean(gpu_mem_list)) if len(gpu_mem_list)>0 else float('nan'),
            'std_gpu_mem_mb': float(np.nanstd(gpu_mem_list)) if len(gpu_mem_list)>0 else float('nan'),
            'avg_cpu_rss_mb': float(np.nanmean(cpu_mem_list)) if len(cpu_mem_list)>0 else float('nan'),
            'std_cpu_rss_mb': float(np.nanstd(cpu_mem_list)) if len(cpu_mem_list)>0 else float('nan'),
        })
        print(f"设置 {setting_tag} 的结果如下：")
        for k, v in avg_results_with_params.items():
            print(f"  {k}: {v}")
    
    print("所有设置下的实验完成")