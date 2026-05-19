import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
import argparse
import os

def determine_n_clusters(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"文件 {data_path} 不存在")

    try:
        with h5py.File(data_path, 'r') as f:
            print("检测数据格式...")
            
            if 'obs' in f and 'cell_type1' in f['obs']:
                print("   找到 obs/cell_type1 标签")
                y_all = f['obs/cell_type1'][()].astype(str)
                le = LabelEncoder()
                y_all_encoded = le.fit_transform(y_all)
                n_clusters = len(np.unique(y_all_encoded))
                print(f"   基于真实标签的聚类数: {n_clusters}")
                
            elif 'obs' in f and 'cell_type' in f['obs']:
                print("   找到 obs/cell_type 标签")
                y_all = f['obs/cell_type'][()].astype(str)
                le = LabelEncoder()
                y_all_encoded = le.fit_transform(y_all)
                n_clusters = len(np.unique(y_all_encoded))
                print(f"   基于真实标签的聚类数: {n_clusters}")
                
            elif 'X' in f and 'obs' in f:
                print("   检测到 Baron 格式数据")
                X = f['X']
                n_cells = X.shape[0]
                print(f"   细胞数量: {n_cells:,}")
                obs_keys = list(f['obs'].keys())
                print(f"   obs组中的键: {obs_keys}")
                label_found = False
                possible_labels = [
                    'cell_type', 'celltype', 'cell_type1', 'assigned_cluster', 'cluster', 'labels',
                    'Group', 'group', 'cell_group', 'batch', 'sample', 'condition', 'treatment',
                    'cell_ontology_class', 'cell_ontology_id', 'annotation', 'celltype_major',
                    'celltype_minor', 'louvain', 'leiden', 'seurat_clusters', 'predicted_labels'
                ]
                
                for label_name in possible_labels:
                    if label_name in f['obs']:
                        print(f"   找到细胞类型标签: {label_name}")
                        y_all = f[f'obs/{label_name}'][()].astype(str)
                        le = LabelEncoder()
                        y_all_encoded = le.fit_transform(y_all)
                        n_clusters = len(np.unique(y_all_encoded))
                        print(f"   基于真实标签的聚类数: {n_clusters}")
                        label_found = True
                        break
                
                if not label_found and 'cell_id' in f['obs']:
                    print("   尝试从cell_id中提取分组信息...")
                    cell_ids = f['obs/cell_id'][()].astype(str)
                    
                    import re
                    patterns = []
                    for cell_id in cell_ids[:min(100, len(cell_ids))]:
                        match = re.match(r'^([A-Za-z]+)', str(cell_id))
                        if match:
                            patterns.append(match.group(1))
                    
                    unique_patterns = list(set(patterns))
                    if len(unique_patterns) > 1 and len(unique_patterns) <= 20:
                        n_clusters = len(unique_patterns)
                        print(f"   从cell_id中发现 {n_clusters} 个分组模式")
                        print(f"   基于cell_id模式的聚类数: {n_clusters}")
                        label_found = True
                
                if not label_found:
                    print("   未找到任何可用标签，基于细胞数量估算")
                    if n_cells < 1000:
                        n_clusters = 5
                    elif n_cells < 3000:
                        n_clusters = 8
                    elif n_cells < 8000:
                        n_clusters = 10
                    else:
                        n_clusters = 12
                    
                    print(f"   基于细胞数量估算的聚类数: {n_clusters}")
                    print(f"   建议: 可以根据生物学先验知识调整聚类数")
                
            else:
                print(f"   未识别的数据格式，文件包含: {list(f.keys())}")
                n_clusters = 8
                print(f"   使用默认聚类数: {n_clusters}")
                
    except Exception as e:
        print(f"读取文件时出错: {e}")
        print("使用默认聚类数: 8")
        n_clusters = 8

    if n_clusters <= 1:
        print("聚类数过小，设置为最小值 2")
        n_clusters = 2
    elif n_clusters > 20:
        print("聚类数过大，限制为 20")
        n_clusters = 20

    return n_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动确定聚类数量 - 支持多种HDF5数据格式")
    parser.add_argument("--data_path", type=str, default="D://scMGCA_data//scMGCA_data//20 datasets//Bach//data.h5", help="HDF5 数据文件路径")
    args = parser.parse_args()

    print("自动聚类数量确定工具")
    print("=" * 40)
    
    try:
        n_clusters = determine_n_clusters(args.data_path)
        print("\n" + "=" * 40)
        print(f"建议的聚类数量: {n_clusters}")
        print("=" * 40)
        print(f"\n使用方法:")
        print(f"   python main.py --data_path {args.data_path} --n_clusters {n_clusters}")
    except Exception as e:
        print(f"错误：{e}")