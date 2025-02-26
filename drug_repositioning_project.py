# 全人类蛋白质组药物重定位项目

import os
import json
import pickle
import time
import hashlib
import pandas as pd
import numpy as np
import subprocess
import requests
import argparse
from google.colab import drive
import warnings
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from data_crawler import DataCrawler
warnings.filterwarnings('ignore')

# 全局设置
BASE_DIR = '/content/drive/MyDrive/full_docking_project'
DRUGS_DIR = f'{BASE_DIR}/drugs'
PROTEINS_DIR = f'{BASE_DIR}/proteins'
RESULTS_DIR = f'{BASE_DIR}/results'
CHECKPOINTS_DIR = f'{BASE_DIR}/checkpoints'
LOG_DIR = f'{BASE_DIR}/logs'
ANALYSIS_DIR = f'{BASE_DIR}/analysis'
RECOMMENDATIONS_DIR = f'{BASE_DIR}/recommendations'

# 创建项目目录结构
def setup_project_directories():
    """创建项目所需的所有目录"""
    for directory in [BASE_DIR, DRUGS_DIR, PROTEINS_DIR, RESULTS_DIR, 
                      CHECKPOINTS_DIR, LOG_DIR, ANALYSIS_DIR, RECOMMENDATIONS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # 创建README文件
    readme_path = f'{BASE_DIR}/README.md'
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as f:
            f.write("""# 全人类蛋白质组药物重定位项目

## 项目概述
本项目旨在通过分子对接技术，探索已批准药物与人类蛋白质组之间的潜在相互作用，用于药物重定位和副作用预测。

## 目录结构
- `/drugs`: 药物结构文件
- `/proteins`: 蛋白质结构文件
- `/results`: 对接结果
- `/checkpoints`: 项目检查点
- `/logs`: 运行日志
- `/analysis`: 数据分析结果
- `/recommendations`: 药物重定位和副作用预测
- `/data_sources`: 原始数据源文件

## 使用方法
在Google Colab中运行以下模式：
- 数据获取模式: `--mode fetch --limit 20`
- 对接模式: `--mode docking`
- 分析模式: `--mode analysis`
- 推荐模式: `--mode recommend --use-baseline`
- 基准模式: `--mode baseline`
- 仪表板模式: `--mode dashboard`

## 数据来源
- 药物数据: DrugBank
- 蛋白质数据: UniProt
- 蛋白质结构: AlphaFold
- 疾病关联: DisGeNET
- 药物适应症: DrugCentral
- 结合数据: BindingDB
            """)
    
    # 创建development_log.md
    dev_log_path = f'{BASE_DIR}/development_log.md'
    if not os.path.exists(dev_log_path):
        with open(dev_log_path, 'w') as f:
            f.write("""# Development Log

## Project Name: 全人类蛋白质组药物重定位项目

### Date: 2023-06-01

#### 初始设置
- 创建项目目录结构
- 设计核心数据管理系统
- 实现任务跟踪器
            """)

# 任务跟踪系统
class TaskTracker:
    def __init__(self, tracker_file=f'{BASE_DIR}/task_tracker.json'):
        self.tracker_file = tracker_file
        self.tasks = self._load_tracker()
    
    def _load_tracker(self):
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'current_batch': 0,
            'drug_progress': {},
            'protein_progress': {},
            'last_update': time.time()
        }
    
    def save(self):
        self.tasks['last_update'] = time.time()
        with open(self.tracker_file, 'w') as f:
            json.dump(self.tasks, f)
    
    def update_progress(self, completed=0, failed=0):
        self.tasks['completed_tasks'] += completed
        self.tasks['failed_tasks'] += failed
        self.save()
    
    def get_progress(self):
        total = self.tasks['total_tasks']
        completed = self.tasks['completed_tasks']
        if total == 0:
            return 0
        return (completed / total) * 100
    
    def log_batch_completion(self, batch_id, time_taken):
        batch_log_file = f'{LOG_DIR}/batch_{batch_id}_log.json'
        log_data = {
            'batch_id': batch_id,
            'time_taken': time_taken,
            'completed_at': time.time(),
            'tasks_completed': self.tasks['completed_tasks'],
            'progress_percentage': self.get_progress()
        }
        with open(batch_log_file, 'w') as f:
            json.dump(log_data, f)

# 分块任务管理系统
class ChunkManager:
    def __init__(self, all_drugs, all_proteins, chunk_size=100):
        self.all_drugs = all_drugs
        self.all_proteins = all_proteins
        self.chunk_size = chunk_size
        self.tracker = TaskTracker()
        
        # 初始化任务总数
        if self.tracker.tasks['total_tasks'] == 0:
            self.tracker.tasks['total_tasks'] = len(all_drugs) * len(all_proteins)
            self.tracker.save()
    
    def generate_task_id(self, drug_id, protein_id):
        """生成唯一的任务ID"""
        return hashlib.md5(f"{drug_id}_{protein_id}".encode()).hexdigest()
    
    def get_next_chunk(self):
        """获取下一个要处理的分块"""
        completed_tasks = set()
        
        # 加载已完成的任务
        completed_file = f'{CHECKPOINTS_DIR}/completed_tasks.pkl'
        if os.path.exists(completed_file):
            with open(completed_file, 'rb') as f:
                completed_tasks = pickle.load(f)
        
        # 生成所有任务
        all_tasks = []
        for drug_id in self.all_drugs:
            for protein_id in self.all_proteins:
                task_id = self.generate_task_id(drug_id, protein_id)
                if task_id not in completed_tasks:
                    all_tasks.append((drug_id, protein_id, task_id))
        
        # 如果没有剩余任务，返回空列表
        if not all_tasks:
            return []
        
        # 返回下一个分块
        return all_tasks[:self.chunk_size]
    
    def mark_completed(self, completed_task_ids):
        """标记任务为已完成"""
        completed_file = f'{CHECKPOINTS_DIR}/completed_tasks.pkl'
        completed_tasks = set()
        
        # 加载已完成的任务
        if os.path.exists(completed_file):
            with open(completed_file, 'rb') as f:
                completed_tasks = pickle.load(f)
        
        # 添加新完成的任务
        completed_tasks.update(completed_task_ids)
        
        # 保存更新后的已完成任务集
        with open(completed_file, 'wb') as f:
            pickle.dump(completed_tasks, f)
        
        # 更新追踪器
        self.tracker.update_progress(completed=len(completed_task_ids))
        
        return len(completed_tasks)

# 主工作流执行器
def run_full_docking_project():
    """运行完整的对接项目"""
    start_time = time.time()
    
    # 1. 数据准备
    print("准备药物和蛋白质数据...")
    drugs_df = fetch_all_drugs()
    proteins_df = fetch_all_proteins()
    
    # 2. 初始化管理器
    chunk_manager = ChunkManager(
        all_drugs=drugs_df['drug_id'].tolist(),
        all_proteins=proteins_df['uniprot_id'].tolist(),
        chunk_size=50  # 每批50个任务
    )
    
    docking_engine = LightweightDocking()
    
    # 3. 持续处理分块，直到完成所有任务
    batch_id = 0
    while True:
        batch_start_time = time.time()
        
        # 获取下一个分块
        next_chunk = chunk_manager.get_next_chunk()
        
        # 如果没有剩余任务，退出循环
        if not next_chunk:
            print("所有任务已完成！")
            break
        
        print(f"处理批次 {batch_id}，包含 {len(next_chunk)} 个任务...")
        
        # 处理当前分块中的所有任务
        completed_task_ids = []
        results = []
        
        for drug_id, protein_id, task_id in next_chunk:
            print(f"处理: {drug_id} vs {protein_id}")
            
            try:
                # 获取药物SMILES
                drug_smiles = drugs_df[drugs_df['drug_id'] == drug_id]['smiles'].values[0]
                
                # 准备蛋白质和配体
                protein_pdbqt = docking_engine.prepare_protein(protein_id)
                if protein_pdbqt is None:
                    print(f"无法准备蛋白质 {protein_id}")
                    continue
                
                ligand_pdbqt = docking_engine.prepare_ligand(drug_id, drug_smiles)
                if ligand_pdbqt is None:
                    print(f"无法准备配体 {drug_id}")
                    continue
                
                # 寻找结合位点
                binding_site = docking_engine.find_binding_site(protein_pdbqt)
                if binding_site is None:
                    print(f"无法找到蛋白质 {protein_id} 的结合位点")
                    continue
                
                # 执行对接
                output_prefix = f"{RESULTS_DIR}/{drug_id}_{protein_id}"
                docking_result = docking_engine.run_docking(
                    protein_pdbqt,
                    ligand_pdbqt,
                    output_prefix,
                    binding_site['center'],
                    binding_site['size']
                )
                
                if docking_result['success']:
                    print(f"对接成功: {drug_id} vs {protein_id}, 结合能: {docking_result['binding_energy']}")
                    result_data = {
                        'drug_id': drug_id,
                        'protein_id': protein_id,
                        'binding_energy': docking_result['binding_energy'],
                        'timestamp': time.time()
                    }
                    results.append(result_data)
                    completed_task_ids.append(task_id)
                    
                    # 保存单个结果到文件
                    with open(f'{output_prefix}_result.json', 'w') as f:
                        json.dump(result_data, f)
                else:
                    print(f"对接失败: {drug_id} vs {protein_id}, 错误: {docking_result.get('error', '未知错误')}")
            
            except Exception as e:
                print(f"处理任务 {drug_id} vs {protein_id} 时出错: {str(e)}")
        
        # 标记已完成的任务
        total_completed = chunk_manager.mark_completed(completed_task_ids)
        
        # 保存当前批次的结果
        batch_results_file = f'{RESULTS_DIR}/batch_{batch_id}_results.json'
        with open(batch_results_file, 'w') as f:
            json.dump(results, f)
        
        # 计算批次执行时间
        batch_time = time.time() - batch_start_time
        print(f"批次 {batch_id} 完成，用时 {batch_time:.2f} 秒")
        print(f"总进度: {chunk_manager.tracker.get_progress():.2f}% ({total_completed}/{chunk_manager.tracker.tasks['total_tasks']})")
        
        # 记录批次完成日志
        chunk_manager.tracker.log_batch_completion(batch_id, batch_time)
        
        # 增加批次ID
        batch_id += 1
        
        # 如果执行时间接近Colab限制（11小时），退出循环
        if (time.time() - start_time) > 11 * 3600:  # 11小时
            print("接近Colab会话限制，暂停执行。重新启动Colab继续。")
            break
    
    print(f"执行完成！总用时: {(time.time() - start_time) / 3600:.2f} 小时")
    print(f"总进度: {chunk_manager.tracker.get_progress():.2f}%")

    dev_logger = DevelopmentLogger()
    dev_logger.add_entry("批次处理进度", f"- 已完成{batch_id}个批次\n- 当前进度: {chunk_manager.tracker.get_progress():.2f}%")

# 增量分析系统
class IncrementalAnalyzer:
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.results_dir = f'{base_dir}/results'
        self.analysis_dir = f'{base_dir}/analysis'
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def collect_all_results(self):
        """收集所有已完成的对接结果"""
        all_results = []
        
        # 查找所有结果文件
        result_files = []
        for root, _, files in os.walk(self.results_dir):
            for file in files:
                if file.endswith('_result.json'):
                    result_files.append(os.path.join(root, file))
        
        # 加载所有结果
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
            except:
                print(f"无法加载结果文件: {file}")
        
        return pd.DataFrame(all_results)
    
    def identify_strong_interactions(self):
        """识别强相互作用（低结合能）"""
        results_df = self.collect_all_results()
        
        if results_df.empty:
            print("没有找到结果数据")
            return pd.DataFrame()
        
        # 筛选强相互作用
        strong_interactions = results_df[results_df['binding_energy'] < -8.0].sort_values('binding_energy')
        
        # 保存结果
        output_file = f'{self.analysis_dir}/strong_interactions.csv'
        strong_interactions.to_csv(output_file, index=False)
        
        print(f"识别了 {len(strong_interactions)} 个强相互作用，已保存到 {output_file}")
        return strong_interactions
    
    def analyze_drug_promiscuity(self):
        """分析药物的多靶点特性"""
        results_df = self.collect_all_results()
        
        if results_df.empty:
            print("没有找到结果数据")
            return pd.DataFrame()
        
        # 计算每种药物的结合蛋白质数量
        drug_targets = results_df.groupby('drug_id').agg({
            'protein_id': 'nunique',
            'binding_energy': ['min', 'mean', 'count']
        })
        
        drug_targets.columns = ['target_count', 'best_energy', 'avg_energy', 'docking_count']
        drug_targets = drug_targets.reset_index().sort_values('target_count', ascending=False)
        
        # 保存结果
        output_file = f'{self.analysis_dir}/drug_promiscuity.csv'
        drug_targets.to_csv(output_file, index=False)
        
        print(f"药物多靶点分析完成，已保存到 {output_file}")
        return drug_targets
    
    def analyze_protein_druggability(self):
        """分析蛋白质的可药性"""
        results_df = self.collect_all_results()
        
        if results_df.empty:
            print("没有找到结果数据")
            return pd.DataFrame()
        
        # 计算每个蛋白质的结合药物数量
        protein_drugs = results_df.groupby('protein_id').agg({
            'drug_id': 'nunique',
            'binding_energy': ['min', 'mean', 'count']
        })
        
        protein_drugs.columns = ['drug_count', 'best_energy', 'avg_energy', 'docking_count']
        protein_drugs = protein_drugs.reset_index().sort_values('drug_count', ascending=False)
        
        # 保存结果
        output_file = f'{self.analysis_dir}/protein_druggability.csv'
        protein_drugs.to_csv(output_file, index=False)
        
        print(f"蛋白质可药性分析完成，已保存到 {output_file}")
        return protein_drugs
    
    def generate_interaction_network(self, energy_threshold=-7.0, max_edges=1000):
        """生成药物-蛋白质相互作用网络"""
        results_df = self.collect_all_results()
        
        if results_df.empty:
            print("没有找到结果数据")
            return None
        
        # 筛选强相互作用
        filtered_results = results_df[results_df['binding_energy'] < energy_threshold]
        
        # 如果边太多，取前N个最强的相互作用
        if len(filtered_results) > max_edges:
            filtered_results = filtered_results.sort_values('binding_energy').head(max_edges)
        
        # 创建网络
        G = nx.Graph()
        
        # 添加节点和边
        for _, row in filtered_results.iterrows():
            G.add_node(row['drug_id'], type='drug')
            G.add_node(row['protein_id'], type='protein')
            G.add_edge(row['drug_id'], row['protein_id'], weight=-row['binding_energy'])
        
        # 保存为GraphML格式
        output_file = f'{self.analysis_dir}/interaction_network.graphml'
        nx.write_graphml(G, output_file)
        
        print(f"相互作用网络已保存到 {output_file}")
        return G
    
    def analyze_tier_performance(self):
        """分析不同分类蛋白质的对接表现"""
        # 获取所有对接结果
        results_df = self.collect_all_results()
        
        # 加载蛋白质分类信息
        tier_file = f'{DATA_DIR}/protein_classification.csv'
        if not os.path.exists(tier_file):
            print("无法找到蛋白质分类数据")
            return None
        
        tier_df = pd.read_csv(tier_file)
        
        # 合并分类信息
        merged_df = pd.merge(
            results_df, 
            tier_df[['uniprot_id', 'tier']], 
            left_on='protein_id', 
            right_on='uniprot_id', 
            how='left'
        )
        
        # 填充缺失值
        merged_df['tier'] = merged_df['tier'].fillna(4)
        
        # 按层级分组计算统计量
        tier_stats = merged_df.groupby('tier').agg({
            'binding_energy': ['count', 'mean', 'std', 'min', 'max'],
            'protein_id': 'nunique',
            'ligand_id': 'nunique'
        })
        
        # 保存结果
        tier_stats.to_csv(f'{ANALYSIS_DIR}/tier_performance.csv')
        
        # 创建可视化
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        
        # 绘制不同层级的结合能分布
        sns.boxplot(x='tier', y='binding_energy', data=merged_df)
        plt.title('不同蛋白质分类的结合能分布')
        plt.xlabel('蛋白质分类')
        plt.ylabel('结合能 (kcal/mol)')
        plt.savefig(f'{ANALYSIS_DIR}/tier_binding_energy.png')
        
        # 打印统计数据
        print("\n不同蛋白质分类的对接结果统计:")
        print(f"一级(有结构+有结合数据): {tier_stats.loc[1]['binding_energy']['count']:.0f}个结果, {tier_stats.loc[1]['protein_id']['nunique']:.0f}个蛋白质")
        print(f"二级(有结构+无结合数据): {tier_stats.loc[2]['binding_energy']['count']:.0f}个结果, {tier_stats.loc[2]['protein_id']['nunique']:.0f}个蛋白质")
        print(f"三级(无结构+有结合数据): {tier_stats.loc[3]['binding_energy']['count']:.0f}个结果, {tier_stats.loc[3]['protein_id']['nunique']:.0f}个蛋白质")
        print(f"四级(无结构+无结合数据): {tier_stats.loc[4]['binding_energy']['count']:.0f}个结果, {tier_stats.loc[4]['protein_id']['nunique']:.0f}个蛋白质")
        
        return tier_stats

# 基准比较系统 - 在IncrementalAnalyzer类后添加
class BaselineReferenceSystem:
    """基准参考系统 - 根据蛋白质分类使用不同的评估方法"""
    
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.reference_data_file = f'{DATA_DIR}/reference_binding_data.csv'
        self.reference_data = self._load_reference_data()
        self.baseline_scores = {}  # 存储每个蛋白质的基准得分
        self.protein_tiers = self._load_protein_tiers()
    
    def _load_protein_tiers(self):
        """加载蛋白质分类数据"""
        tier_file = f'{DATA_DIR}/protein_classification.csv'
        if os.path.exists(tier_file):
            tier_df = pd.read_csv(tier_file)
            # 创建UniProt ID到tier的映射
            return dict(zip(tier_df['uniprot_id'], tier_df['tier']))
        else:
            # 如果分类文件不存在，尝试创建
            try:
                crawler = DataCrawler()
                classified_df, _ = crawler.classify_protein_data()
                return dict(zip(classified_df['uniprot_id'], classified_df['tier']))
            except:
                print("无法加载或创建蛋白质分类数据")
                return {}
    
    def _load_reference_data(self):
        """加载已知的分子-蛋白质结合数据，优先使用PDBbind"""
        reference_data_file = f'{DATA_DIR}/reference_binding_data.csv'
        pdbbind_file = f'{DATA_DIR}/pdbbind_refined_set.csv'
        
        # 优先使用PDBbind数据
        if os.path.exists(pdbbind_file):
            print(f"使用PDBbind数据作为参考: {pdbbind_file}")
            pdbbind_df = pd.read_csv(pdbbind_file)
            
            # 转换PDBbind数据格式以匹配我们的需求
            if not pdbbind_df.empty:
                # 需要获取PDBbind中配体的SMILES
                processed_data = []
                for _, row in pdbbind_df.iterrows():
                    pdb_id = row['pdb_id']
                    uniprot_id = row['uniprot_id']
                    affinity_value = row['affinity_value']
                    affinity_type = row['affinity_type']
                    ligand_name = row.get('ligand_name', f"Ligand_{pdb_id}")
                    
                    # 获取配体SMILES（理想情况下应该从PDB或其他来源获取）
                    # 这里简化处理，实际应用中需要实现
                    ligand_smiles = self._get_ligand_smiles(pdb_id, row.get('ligand_id'))
                    
                    if ligand_smiles and uniprot_id:
                        processed_data.append({
                            'protein_id': uniprot_id,
                            'ligand_name': ligand_name,
                            'ligand_smiles': ligand_smiles,
                            'binding_constant': affinity_value,
                            'constant_type': affinity_type
                        })
                
                if processed_data:
                    # 保存处理后的数据
                    processed_df = pd.DataFrame(processed_data)
                    processed_df.to_csv(reference_data_file, index=False)
                    print(f"已处理 {len(processed_df)} 条PDBbind数据为参考数据")
                    return processed_df
        
        # 如果没有PDBbind数据或处理失败，使用常规参考数据
        if os.path.exists(reference_data_file):
            return pd.read_csv(reference_data_file)
        else:
            # 尝试从BindingDB获取数据
            try:
                print("尝试从BindingDB获取参考数据...")
                crawler = DataCrawler()
                binding_df = crawler.get_binding_data()
                return binding_df
            except Exception as e:
                print(f"获取参考数据时出错: {str(e)}")
                # 创建示例数据
                print("创建示例参考数据...")
                example_data = [
                    {'protein_id': 'P30556', 'ligand_name': 'Losartan', 'ligand_smiles': 'CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1', 'binding_constant': 2.3, 'constant_type': 'Ki_nM'},
                    {'protein_id': 'P31645', 'ligand_name': 'Fluoxetine', 'ligand_smiles': 'CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1', 'binding_constant': 8.5, 'constant_type': 'IC50_nM'},
                    {'protein_id': 'P05067', 'ligand_name': 'Verubecestat', 'ligand_smiles': 'CC(C)(C)OC(=O)N1CC(C1)C(=O)N(C)C1CCN(C1)C(=O)C1=CC=C(F)C(=C1)C#N', 'binding_constant': 12.0, 'constant_type': 'IC50_nM'},
                    {'protein_id': 'P37840', 'ligand_name': 'Fasudil', 'ligand_smiles': 'CN1CC(=NNC1=O)N1CCCN(CC1)C1=CC=CC2=CC=CC=C21', 'binding_constant': 150.0, 'constant_type': 'EC50_nM'},
                    {'protein_id': 'P01308', 'ligand_name': 'Insulin Lispro', 'ligand_smiles': 'CC(C)CC1NC(=O)C(CCCNC(=N)N)NC(=O)C(CC(C)C)NC(=O)CNC(=O)C(CO)NC(=O)C(CC(C)C)NC(=O)C(CCC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(C)NC1=O', 'binding_constant': 0.2, 'constant_type': 'Kd_nM'}
                ]
                
                df = pd.DataFrame(example_data)
                df.to_csv(reference_data_file, index=False)
                print(f"已创建{len(df)}条示例参考数据")
                return df
    
    def _get_ligand_smiles(self, pdb_id, ligand_id=None):
        """从PDB或其他来源获取配体的SMILES表示"""
        # 实际应用中应该实现从PDB或PubChem获取SMILES
        # 这里简化处理，返回一些示例SMILES
        example_smiles = {
            '6sjm': 'CC1=CC(=CC(=C1)C)NC(=O)C2=NC(=CN2)C3=CC(=CC=C3)OC',  # BACE1抑制剂
            '6niv': 'CNCCC(OC1=CC=C(C=C1)C(F)(F)F)C2=CC=CC=C2',            # Fluoxetine
            '6os1': 'CCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NN[NH]N4)CO)Cl', # Losartan
            '2n0a': 'CN1CC(=NNC1=O)N1CCCN(CC1)C1=CC=CC2=CC=CC=C21',        # Fasudil
            '6pxv': 'CC(C)CC1NC(=O)C(CCCNC(=N)N)NC(=O)C(CC(C)C)NC(=O)C2CCC(=O)N2C(=O)C(CC(C)C)NC(=O)C(CCC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(C)NC1=O' # 短版胰岛素
        }
        
        if pdb_id.lower() in example_smiles:
            return example_smiles[pdb_id.lower()]
        
        # 试图从PubChem获取（实际应用中实现）
        # 如果找不到，返回一个默认SMILES
        return 'C'  # 甲烷作为默认值
    
    def evaluate_docking_result(self, protein_id, ligand_id, docking_score):
        """根据蛋白质分类评估对接结果"""
        # 获取蛋白质分类
        tier = self.protein_tiers.get(protein_id, 4)  # 默认为四级
        
        # 一级蛋白质：使用已知结合数据评估
        if tier == 1:
            return self._evaluate_with_reference_data(protein_id, ligand_id, docking_score)
        
        # 二级蛋白质：使用相似蛋白质的统计分布
        elif tier == 2:
            return self._evaluate_with_similar_proteins(protein_id, docking_score)
        
        # 三级蛋白质：使用结合数据但考虑结构不确定性
        elif tier == 3:
            return self._evaluate_with_binding_data_no_structure(protein_id, ligand_id, docking_score)
        
        # 四级蛋白质：使用全局统计分布
        else:
            return self._evaluate_with_global_statistics(docking_score)
    
    def _evaluate_with_reference_data(self, protein_id, ligand_id, docking_score):
        """使用参考结合数据评估一级蛋白质"""
        # 检查是否有此蛋白质的基准数据
        protein_data = self.reference_data[self.reference_data['protein_id'] == protein_id]
        
        if protein_data.empty:
            # 如果没有特定参考数据，使用其他方法
            return self._evaluate_with_global_statistics(docking_score)
        
        # 计算Z分数（相对于已知结合剂的分布）
        baseline_scores = list(protein_data['binding_constant'])
        if not baseline_scores:
            return {'evaluation': 'unknown', 'relative_score': 0, 'confidence': 'low'}
            
        mean_score = np.mean(baseline_scores)
        std_score = np.std(baseline_scores) if len(baseline_scores) > 1 else 1.0
        
        z_score = (docking_score - mean_score) / std_score if std_score > 0 else 0
        
        # 解释Z分数
        if z_score < -2.0:
            evaluation = 'excellent'
            confidence = 'high'
        elif z_score < -1.0:
            evaluation = 'good'
            confidence = 'high'
        elif z_score < 0:
            evaluation = 'moderate'
            confidence = 'medium'
        else:
            evaluation = 'poor'
            confidence = 'medium'
        
        return {
            'evaluation': evaluation,
            'relative_score': z_score,
            'confidence': confidence,
            'method': 'reference_data',
            'protein_tier': 1
        }
    
    def _evaluate_with_similar_proteins(self, protein_id, docking_score):
        """使用相似蛋白质评估二级蛋白质"""
        # 这里应该实现查找相似蛋白质的逻辑
        # 例如，可以根据蛋白质家族或结构相似性
        
        # 简化版：使用全局统计，但降低置信度
        result = self._evaluate_with_global_statistics(docking_score)
        result['confidence'] = 'medium' if result['confidence'] == 'high' else 'low'
        result['method'] = 'similar_proteins'
        result['protein_tier'] = 2
        return result
    
    def _evaluate_with_binding_data_no_structure(self, protein_id, ligand_id, docking_score):
        """使用结合数据评估无结构的蛋白质(三级)"""
        # 检查是否有此蛋白质的基准数据
        protein_data = self.reference_data[self.reference_data['protein_id'] == protein_id]
        
        if protein_data.empty:
            # 如果没有特定参考数据，使用其他方法
            return self._evaluate_with_global_statistics(docking_score)
        
        # 计算Z分数（相对于已知结合剂的分布）
        baseline_scores = list(protein_data['binding_constant'])
        if not baseline_scores:
            return {'evaluation': 'unknown', 'relative_score': 0, 'confidence': 'low'}
            
        mean_score = np.mean(baseline_scores)
        std_score = np.std(baseline_scores) if len(baseline_scores) > 1 else 1.0
        
        z_score = (docking_score - mean_score) / std_score if std_score > 0 else 0
        
        # 对无结构蛋白质，我们降低阈值并降低置信度
        if z_score < -2.5:  # 需要更强的信号
            evaluation = 'excellent'
            confidence = 'medium'  # 降级置信度
        elif z_score < -1.5:
            evaluation = 'good'
            confidence = 'medium'
        elif z_score < -0.5:
            evaluation = 'moderate'
            confidence = 'low'
        else:
            evaluation = 'poor'
            confidence = 'low'
        
        return {
            'evaluation': evaluation,
            'relative_score': z_score,
            'confidence': confidence,
            'method': 'binding_data_no_structure',
            'protein_tier': 3
        }
    
    def _evaluate_with_global_statistics(self, docking_score):
        """使用全局统计分布评估四级蛋白质"""
        # 使用经验阈值
        if docking_score < -10.0:
            evaluation = 'excellent'
            confidence = 'low'
        elif docking_score < -8.0:
            evaluation = 'good'
            confidence = 'low'
        elif docking_score < -6.5:
            evaluation = 'moderate'
            confidence = 'low'
        else:
            evaluation = 'poor'
            confidence = 'low'
        
        # 没有参考，所以相对分数设为0
        return {
            'evaluation': evaluation,
            'relative_score': 0,
            'confidence': confidence,
            'method': 'global_statistics',
            'protein_tier': 4
        }

# 药物数据获取
def fetch_all_drugs():
    """获取所有批准药物的数据"""
    drugs_list_file = f'{BASE_DIR}/all_drugs.csv'
    
    # 如果已经下载过，直接加载
    if os.path.exists(drugs_list_file):
        return pd.read_csv(drugs_list_file)
    
    # 使用爬取系统获取数据
    crawler = DataCrawler()
    drugs_df = crawler.get_drugbank_drugs()
    
    # 保存到主目录
    drugs_df.to_csv(drugs_list_file, index=False)
    return drugs_df

# 蛋白质数据获取
def fetch_all_proteins():
    """获取所有人类蛋白质数据"""
    proteins_list_file = f'{BASE_DIR}/all_proteins.csv'
    
    # 如果已经下载过，直接加载
    if os.path.exists(proteins_list_file):
        return pd.read_csv(proteins_list_file)
    
    # 使用爬取系统获取数据
    crawler = DataCrawler()
    proteins_df = crawler.get_uniprot_proteins()
    
    # 保存到主目录
    proteins_df.to_csv(proteins_list_file, index=False)
    return proteins_df

# 轻量级对接引擎
class LightweightDocking:
    def __init__(self):
        # 设置对接环境
        self._setup_environment()
    
    def _setup_environment(self):
        """设置对接环境"""
        try:
            # 检查是否安装了必要的软件包
            import rdkit
            
            # 下载并设置AutoDock Vina
            if not os.path.exists('/usr/local/bin/vina'):
                subprocess.run([
                    "wget", "http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz",
                    "-q", "-O", "/tmp/vina.tgz"
                ])
                subprocess.run(["tar", "-xzf", "/tmp/vina.tgz", "-C", "/tmp"])
                subprocess.run(["mv", "/tmp/autodock_vina_1_1_2_linux_x86/bin/vina", "/usr/local/bin/"])
                subprocess.run(["chmod", "+x", "/usr/local/bin/vina"])
                print("AutoDock Vina已安装")
        except Exception as e:
            print(f"环境设置失败: {str(e)}")
    
    def prepare_protein(self, protein_id, protein_file=None):
        """准备蛋白质文件，考虑蛋白质分类"""
        output_pdbqt = f'{PROTEINS_DIR}/{protein_id}.pdbqt'
        
        # 如果已经准备好，直接返回
        if os.path.exists(output_pdbqt):
            return output_pdbqt
        
        # 加载蛋白质分类信息
        tier_file = f'{DATA_DIR}/protein_classification.csv'
        protein_tier = 4  # 默认为四级
        if os.path.exists(tier_file):
            try:
                tier_df = pd.read_csv(tier_file)
                if protein_id in tier_df['uniprot_id'].values:
                    protein_tier = tier_df[tier_df['uniprot_id'] == protein_id]['tier'].iloc[0]
            except:
                pass
        
        # 如果没有提供文件，尝试找到合适的蛋白质结构
        if protein_file is None:
            # 一级和二级蛋白质：优先使用PDB实验结构
            if protein_tier <= 2:
                # 1. 首先检查是否有PDB实验结构
                # 加载PDB-UniProt映射
                pdb_structures_file = f'{DATA_DIR}/pdb_structures.csv'
                if os.path.exists(pdb_structures_file):
                    pdb_df = pd.read_csv(pdb_structures_file)
                    matching_pdbs = pdb_df[pdb_df['uniprot_id'] == protein_id]
                    
                    if not matching_pdbs.empty:
                        # 按分辨率排序，选择最好的结构
                        if 'resolution' in matching_pdbs.columns:
                            # 较低的分辨率更好，但有些NMR结构没有分辨率值
                            matching_pdbs = matching_pdbs.sort_values(
                                by='resolution', 
                                ascending=True, 
                                na_position='last'
                            )
                        
                        best_pdb = matching_pdbs.iloc[0]
                        pdb_id = best_pdb['pdb_id']
                        
                        # 检查PDB文件是否存在
                        pdb_file = f'{PROTEINS_DIR}/pdb_{pdb_id.lower()}.pdb'
                        if os.path.exists(pdb_file):
                            print(f"使用PDB实验结构 {pdb_id} 代替AlphaFold预测结构")
                            protein_file = pdb_file
            
            # 三级蛋白质：有结合数据但无PDB结构
            # 对这些蛋白质，我们可以尝试使用更保守的AlphaFold处理方式
            if protein_tier == 3:
                pdb_file = f'{PROTEINS_DIR}/{protein_id}.pdb'
                
                if not os.path.exists(pdb_file):
                    print(f"蛋白质 {protein_id} 有结合数据但无PDB结构，从AlphaFold下载...")
                    crawler = DataCrawler()
                    pdb_file = crawler.download_alphafold_structure(protein_id)
                    
                    if pdb_file is None:
                        return None
                
                # 对于有结合数据的蛋白质，我们可以尝试查找结合位点信息，提高对接精度
                # 这里可以添加查询结合位点的代码
                
                protein_file = pdb_file
            
            # 四级蛋白质：仅使用AlphaFold预测结构
            else:
                pdb_file = f'{PROTEINS_DIR}/{protein_id}.pdb'
                
                if not os.path.exists(pdb_file):
                    print(f"从AlphaFold下载蛋白质 {protein_id}...")
                    crawler = DataCrawler()
                    pdb_file = crawler.download_alphafold_structure(protein_id)
                    
                    if pdb_file is None:
                        return None
                
                protein_file = pdb_file
        
        # 转换为PDBQT格式
        try:
            print(f"准备蛋白质 {protein_id}...")
            subprocess.run([
                "python", "-m", "meeko.scripts.mk_prepare_receptor",
                "-i", protein_file,
                "-o", output_pdbqt
            ])
            
            if os.path.exists(output_pdbqt):
                return output_pdbqt
            else:
                print(f"无法转换蛋白质文件 {protein_file}")
                return None
        except Exception as e:
            print(f"准备蛋白质时出错: {str(e)}")
            return None
    
    def prepare_ligand(self, drug_id, smiles):
        """准备配体文件"""
        output_pdbqt = f'{DRUGS_DIR}/{drug_id}.pdbqt'
        
        # 如果已经准备好，直接返回
        if os.path.exists(output_pdbqt):
            return output_pdbqt
        
        try:
            # 使用RDKit生成3D构象
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            
            # 保存为PDB文件
            temp_pdb = f'{DRUGS_DIR}/{drug_id}.pdb'
            Chem.MolToPDBFile(mol, temp_pdb)
            
            # 转换为PDBQT
            subprocess.run(["obabel", temp_pdb, "-O", output_pdbqt], check=True)
            
            return output_pdbqt
        except Exception as e:
            print(f"准备配体 {drug_id} 失败: {str(e)}")
            return None
    
    def find_binding_site(self, protein_pdbqt):
        """简单的结合位点识别"""
        # 在实际应用中，这应该使用更复杂的方法
        # 这里简单地返回蛋白质几何中心和一个固定的盒子大小
        try:
            with open(protein_pdbqt, 'r') as f:
                coords = []
                for line in f:
                    if line.startswith('ATOM'):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append((x, y, z))
            
            if not coords:
                return None
            
            # 计算几何中心
            center_x = sum(c[0] for c in coords) / len(coords)
            center_y = sum(c[1] for c in coords) / len(coords)
            center_z = sum(c[2] for c in coords) / len(coords)
            
            return {
                'center': (center_x, center_y, center_z),
                'size': (20, 20, 20)  # 20埃盒子
            }
        except Exception as e:
            print(f"寻找结合位点失败: {str(e)}")
            return None
    
    def run_docking(self, receptor_pdbqt, ligand_pdbqt, output_prefix, center, box_size):
        """运行极简化的对接"""
        output_pdbqt = f"{output_prefix}_out.pdbqt"
        output_log = f"{output_prefix}_log.txt"
        
        # 超轻量级对接参数
        cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]),
            "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
            "--out", output_pdbqt,
            "--log", output_log,
            "--exhaustiveness", "1",
            "--num_modes", "1"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 解析结果
            binding_energy = None
            if os.path.exists(output_log):
                with open(output_log, 'r') as f:
                    for line in f:
                        if "Affinity:" in line:
                            binding_energy = float(line.split()[1])
                            break
            
            # 清理大文件，仅保留结果数据
            if os.path.exists(output_pdbqt):
                os.remove(output_pdbqt)
            
            return {
                'binding_energy': binding_energy,
                'success': binding_energy is not None
            }
            
        except Exception as e:
            return {
                'binding_energy': None,
                'success': False,
                'error': str(e)
            }

# 药物重定位推荐系统
class DrugRepositioningRecommender:
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.analysis_dir = f'{base_dir}/analysis'
        self.recommendations_dir = f'{base_dir}/recommendations'
        os.makedirs(self.recommendations_dir, exist_ok=True)
        
        # 加载疾病-蛋白质关联数据
        self.disease_protein_file = f'{base_dir}/disease_protein_associations.csv'
        self.disease_protein_df = self._load_disease_associations()
        
        # 加载已知药物适应症
        self.drug_indication_file = f'{base_dir}/drug_indications.csv'
        self.drug_indication_df = self._load_drug_indications()
    
    def _load_disease_associations(self):
        """加载疾病-蛋白质关联数据"""
        if os.path.exists(self.disease_protein_file):
            return pd.read_csv(self.disease_protein_file)
        else:
            # 尝试从DisGeNET下载疾病-蛋白质关联数据
            try:
                print("尝试从DisGeNET下载疾病-蛋白质关联数据...")
                url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz"
                response = requests.get(url)
                
                if response.status_code == 200:
                    # 保存并解压文件
                    with open(f"{BASE_DIR}/disease_gene.tsv.gz", 'wb') as f:
                        f.write(response.content)
                    
                    # 使用gunzip解压
                    subprocess.run(["gunzip", "-f", f"{BASE_DIR}/disease_gene.tsv.gz"], 
                                  check=True, capture_output=True)
                    
                    # 读取并处理数据
                    disease_gene_df = pd.read_csv(f"{BASE_DIR}/disease_gene.tsv", sep="\t")
                    
                    # 重命名列
                    disease_gene_df = disease_gene_df.rename(columns={
                        'diseaseId': 'disease_id',
                        'diseaseName': 'disease_name',
                        'geneSymbol': 'gene_name',
                        'score': 'score'
                    })
                    
                    # 选择需要的列
                    if all(col in disease_gene_df.columns for col in ['disease_id', 'disease_name', 'gene_name', 'score']):
                        disease_gene_df = disease_gene_df[['disease_id', 'disease_name', 'gene_name', 'score']]
                        
                        # 加载蛋白质数据以获取gene_name到protein_id的映射
                        proteins_df = fetch_all_proteins()
                        
                        # 合并数据以将gene_name映射到protein_id
                        if 'gene_name' in proteins_df.columns:
                            disease_protein_df = pd.merge(
                                disease_gene_df, 
                                proteins_df[['uniprot_id', 'gene_name']], 
                                on='gene_name', 
                                how='inner'
                            )
                            
                            disease_protein_df = disease_protein_df.rename(columns={'uniprot_id': 'protein_id'})
                            disease_protein_df = disease_protein_df[['disease_id', 'disease_name', 'protein_id', 'score']]
                            
                            # 保存结果
                            disease_protein_df.to_csv(self.disease_protein_file, index=False)
                            print(f"已下载并处理{len(disease_protein_df)}条疾病-蛋白质关联数据")
                            return disease_protein_df
            except Exception as e:
                print(f"获取疾病-蛋白质关联数据时出错: {str(e)}")
            
            # 创建示例数据
            print("创建示例疾病-蛋白质关联数据...")
            example_associations = [
                {'disease_id': 'DOID:10652', 'disease_name': 'Alzheimer disease', 'protein_id': 'P05067', 'score': 0.9},
                {'disease_id': 'DOID:14330', 'disease_name': 'Parkinson disease', 'protein_id': 'P37840', 'score': 0.8},
                {'disease_id': 'DOID:1470', 'disease_name': 'Major depressive disorder', 'protein_id': 'P31645', 'score': 0.7},
                {'disease_id': 'DOID:2531', 'disease_name': 'Hypertension', 'protein_id': 'P30556', 'score': 0.85},
                {'disease_id': 'DOID:9953', 'disease_name': 'Diabetes mellitus', 'protein_id': 'P01308', 'score': 0.95},
            ]
            
            df = pd.DataFrame(example_associations)
            df.to_csv(self.disease_protein_file, index=False)
            print(f"已创建{len(df)}条示例疾病-蛋白质关联数据")
            return df
    
    def _load_drug_indications(self):
        """加载药物适应症数据"""
        if os.path.exists(self.drug_indication_file):
            return pd.read_csv(self.drug_indication_file)
        else:
            # 创建示例数据
            print("创建示例药物适应症数据...")
            example_indications = [
                {'drug_id': 'DB00316', 'disease_id': 'DOID:2531', 'disease_name': 'Hypertension'},
                {'drug_id': 'DB00945', 'disease_id': 'DOID:8398', 'disease_name': 'Osteoarthritis'},
                {'drug_id': 'DB00273', 'disease_id': 'DOID:7910', 'disease_name': 'Hypercholesterolemia'},
                {'drug_id': 'DB00316', 'disease_id': 'DOID:653', 'disease_name': 'Rheumatoid arthritis'},
                {'drug_id': 'DB00945', 'disease_id': 'DOID:8398', 'disease_name': 'Rheumatoid arthritis'},
            ]
            
            df = pd.DataFrame(example_indications)
            df.to_csv(self.drug_indication_file, index=False)
            print(f"已创建{len(df)}条示例药物适应症数据")
            return df
    
    def generate_repositioning_recommendations(self, binding_threshold=-8.0, use_baseline=True, tier_threshold=3):
        """生成药物重定位推荐，考虑蛋白质分类"""
        analyzer = IncrementalAnalyzer()
        results_df = analyzer.collect_all_results()
        
        if results_df.empty or self.disease_protein_df.empty:
            print("缺少必要数据，无法生成推荐")
            return pd.DataFrame()
        
        # 使用基准比较系统
        if use_baseline:
            baseline_system = BaselineReferenceSystem()
            protein_tiers = baseline_system.protein_tiers
            
            # 评估每个对接结果
            evaluations = []
            for _, row in results_df.iterrows():
                protein_id = row['protein_id']
                ligand_id = row['ligand_id']
                docking_score = row['binding_energy']
                
                eval_result = baseline_system.evaluate_docking_result(
                    protein_id, ligand_id, docking_score
                )
                
                # 添加到评估列表
                evaluations.append({
                    'protein_id': protein_id,
                    'ligand_id': ligand_id,
                    'docking_score': docking_score,
                    'evaluation': eval_result['evaluation'],
                    'confidence': eval_result['confidence'],
                    'protein_tier': eval_result.get('protein_tier', 4)
                })
            
            # 转换为DataFrame
            eval_df = pd.DataFrame(evaluations)
            
            # 合并评估结果和原始对接结果
            results_df = pd.merge(
                results_df, 
                eval_df, 
                on=['protein_id', 'ligand_id'], 
                how='left'
            )
            
            # 筛选强相互作用，根据评估结果和蛋白质层级
            strong_interactions = results_df[
                ((results_df['evaluation'].isin(['excellent', 'good'])) |
                 ((results_df['evaluation'] == 'moderate') & (results_df['protein_tier'] <= 2)) |
                 (results_df['binding_energy'] < binding_threshold)) &
                (results_df['protein_tier'] <= tier_threshold)  # 仅考虑指定层级以内的蛋白质
            ]
        else:
            # 不使用基准系统，仅使用阈值，但仍考虑蛋白质层级
            if 'protein_tier' in results_df.columns:
                strong_interactions = results_df[
                    (results_df['binding_energy'] < binding_threshold) &
                    (results_df['protein_tier'] <= tier_threshold)
                ]
            else:
                strong_interactions = results_df[results_df['binding_energy'] < binding_threshold]
        
        # 合并与疾病相关的蛋白质
        merged_data = pd.merge(
            strong_interactions,
            self.disease_protein_df,
            on='protein_id',
            how='inner'
        )
        
        # 如果合并后没有数据，返回空DataFrame
        if merged_data.empty:
            print("没有找到与疾病相关的蛋白质相互作用")
            return pd.DataFrame()
        
        # 排除已知适应症
        if not self.drug_indication_df.empty:
            # 获取每种药物的已知疾病
            current_indications = set()
            for _, row in self.drug_indication_df.iterrows():
                current_indications.add((row['drug_id'], row['disease_id']))
            
            # 过滤出新的适应症（不在已知适应症中）
            new_indications = []
            for _, row in merged_data.iterrows():
                if (row['drug_id'], row['disease_id']) not in current_indications:
                    new_indications.append(row)
        
            recommendations_df = pd.DataFrame(new_indications)
        else:
            recommendations_df = merged_data
        
        # 如果没有推荐结果
        if recommendations_df.empty:
            print("没有找到符合条件的药物重定位推荐")
            return pd.DataFrame()
        
        # 按结合能和疾病关联评分排序
        recommendations_df = recommendations_df.sort_values('binding_energy', ascending=False)
        
        # 保存推荐结果
        output_file = f'{self.recommendations_dir}/repositioning_recommendations.csv'
        recommendations_df.to_csv(output_file, index=False)
        
        print(f"生成了 {len(recommendations_df)} 个药物重定位推荐，已保存到 {output_file}")
        
        dev_logger = DevelopmentLogger()
        dev_logger.add_entry("药物重定位推荐", f"- 生成了{len(recommendations_df)}个药物重定位推荐\n- 使用基准系统: {use_baseline}\n- 结果已保存到{output_file}")
        return recommendations_df
    
    def predict_side_effects(self, binding_threshold=-7.0):
        """预测潜在副作用"""
        analyzer = IncrementalAnalyzer()
        results_df = analyzer.collect_all_results()
        
        if results_df.empty or self.disease_protein_df.empty:
            print("缺少必要数据，无法预测副作用")
            return pd.DataFrame()
        
        # 筛选中等强度结合能（不需要特别强，但有明显相互作用）
        interactions = results_df[results_df['binding_energy'] < binding_threshold]
        
        # 合并与疾病相关的蛋白质
        merged_data = pd.merge(
            interactions,
            self.disease_protein_df,
            on='protein_id',
            how='inner'
        )
        
        # 排除已知适应症
        if not self.drug_indication_df.empty:
            # 获取每种药物的已知疾病
            drug_known_diseases = {}
            for _, row in self.drug_indication_df.iterrows():
                if row['drug_id'] not in drug_known_diseases:
                    drug_known_diseases[row['drug_id']] = set()
                drug_known_diseases[row['drug_id']].add(row['disease_id'])
            
            # 过滤出可能的副作用（与药物已知适应症不同的疾病）
            potential_side_effects = []
            for _, row in merged_data.iterrows():
                drug_id = row['drug_id']
                disease_id = row['disease_id']
                
                # 如果药物有已知适应症，且当前疾病不是已知适应症
                if drug_id in drug_known_diseases and disease_id not in drug_known_diseases[drug_id]:
                    potential_side_effects.append(row)
            
            side_effects_df = pd.DataFrame(potential_side_effects)
        else:
            side_effects_df = merged_data
        
        # 如果没有预测结果
        if side_effects_df.empty:
            print("没有找到潜在副作用预测")
            return pd.DataFrame()
        
        # 按结合能和疾病关联评分排序
        side_effects_df['risk_score'] = -side_effects_df['binding_energy'] * side_effects_df['score']
        side_effects_df = side_effects_df.sort_values('risk_score', ascending=False)
        
        # 保存预测结果
        output_file = f'{self.recommendations_dir}/potential_side_effects.csv'
        side_effects_df.to_csv(output_file, index=False)
        
        print(f"预测了 {len(side_effects_df)} 个潜在副作用，已保存到 {output_file}")
        return side_effects_df

# 脚本入口点
def main():
    """主函数 - 项目执行入口"""
    parser = argparse.ArgumentParser(description='全人类蛋白质组药物重定位项目')
    parser.add_argument('--mode', type=str, default='docking',
                       choices=['fetch', 'docking', 'analysis', 'recommend', 'dashboard', 'baseline'],
                       help='运行模式: fetch (获取数据), docking (对接), analysis (分析), recommend (推荐), dashboard (仪表板), baseline (基准)')
    parser.add_argument('--show-log', action='store_true',
                       help='显示开发日志')
    parser.add_argument('--use-baseline', action='store_true',
                       help='使用基准系统评估对接结果')
    parser.add_argument('--limit', type=int, default=10,
                       help='数据获取模式下每类基础数据的最大记录数')
    parser.add_argument('--include-pdb', action='store_true',
                       help='包括PDB实验结构')
    parser.add_argument('--include-pdbbind', action='store_true',
                       help='包括PDBbind结合数据')
    parser.add_argument('--pdb-only', action='store_true',
                       help='只使用PDB实验结构（忽略AlphaFold）')
    parser.add_argument('--pdbbind-limit', type=int, default=50,
                       help='PDBbind数据的最大记录数')
    parser.add_argument('--pdbbind-dataset', type=str, default='refined',
                       choices=['general', 'refined', 'core'],
                       help='使用的PDBbind数据集')
    parser.add_argument('--classify-proteins', action='store_true',
                      help='对蛋白质进行分类')
    parser.add_argument('--tier-threshold', type=int, default=3,
                      help='用于推荐的蛋白质分类阈值 (1-4)')
    
    args = parser.parse_args()
    
    # 挂载Google Drive
    try:
        drive.mount('/content/drive')
    except:
        print("无法挂载Google Drive，可能不在Colab环境中运行")
    
    # 创建项目目录
    setup_project_directories()
    
    # 初始化开发日志管理器
    dev_logger = DevelopmentLogger()
    
    # 记录运行模式
    dev_logger.log_mode_execution(args.mode)
    
    if args.mode == 'fetch':
        print("启动数据获取模式...")
        crawler = DataCrawler()
        data = crawler.fetch_all_data(
            limit=args.limit,
            include_pdb=args.include_pdb or args.pdb_only,
            include_pdbbind=args.include_pdbbind,
            pdbbind_limit=args.pdbbind_limit
        )
        
        print("\n数据摘要:")
        for key, df in data.items():
            print(f"{key}: {len(df)} 条记录")
        
        dev_logger.add_entry("数据获取完成", f"- 药物: {len(data['drugs'])}条\n- 蛋白质: {len(data['proteins'])}条\n- 疾病-蛋白质关联: {len(data['disease_protein'])}条\n- 药物适应症: {len(data['drug_indication'])}条\n- 结合数据: {len(data['binding_data'])}条")
        
        if args.classify_proteins:
            print("对蛋白质进行分类...")
            crawler.classify_protein_data()
    
    elif args.mode == 'docking':
        print("启动对接模式...")
        start_time = time.time()
        run_full_docking_project()
        elapsed_time = time.time() - start_time
        dev_logger.add_entry("对接计算完成", f"- 耗时: {elapsed_time/3600:.2f}小时\n- 请查看进度仪表板了解详细信息")
        
    elif args.mode == 'analysis':
        print("启动分析模式...")
        dev_logger.log_mode_execution('analysis')
        analyzer = IncrementalAnalyzer()
        
        # 运行所有分析函数
        print("识别强相互作用...")
        analyzer.identify_strong_interactions()
        
        print("分析药物多靶点性质...")
        analyzer.analyze_drug_promiscuity()
        
        print("分析蛋白质可药性...")
        analyzer.analyze_protein_druggability()
        
        print("生成相互作用网络...")
        analyzer.generate_interaction_network()
        
        print("分析不同分类蛋白质的对接表现...")
        analyzer.analyze_tier_performance()
        
        dev_logger.add_entry("分析完成", "- 分析结果已保存到analysis目录")
        
    elif args.mode == 'recommend':
        print("启动推荐模式...")
        dev_logger.log_mode_execution('recommend')
        recommender = DrugRepositioningRecommender()
        
        print("生成药物重定位推荐...")
        recommender.generate_repositioning_recommendations(
            use_baseline=args.use_baseline,
            tier_threshold=args.tier_threshold
        )
        
        print("预测潜在副作用...")
        recommender.predict_side_effects()
    
    elif args.mode == 'baseline':
        print("启动基准模式...")
        dev_logger.log_mode_execution('baseline')
        baseline_system = BaselineReferenceSystem()
        baselines = baseline_system.establish_baselines()
        
        print("\n基准系统摘要:")
        summary = baseline_system.get_baseline_summary()
        print(summary)
        
        summary_file = f'{ANALYSIS_DIR}/baseline_summary.csv'
        summary.to_csv(summary_file, index=False)
        print(f"基准摘要已保存至 {summary_file}")
    
    elif args.mode == 'dashboard':
        print("生成进度仪表板...")
        dev_logger.log_mode_execution('dashboard')
        generate_progress_dashboard()
    
    if args.show_log:
        print(f"开发日志位置: {dev_logger.dev_log_path}")
        try:
            with open(dev_logger.dev_log_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"无法读取日志: {e}")
    
    print("执行完成！")

# 生成进度仪表板
def generate_progress_dashboard():
    """生成Markdown格式的进度仪表板"""
    tracker = TaskTracker()
    
    # 获取总体进度
    total_tasks = tracker.tasks['total_tasks']
    completed_tasks = tracker.tasks['completed_tasks']
    progress_percentage = tracker.get_progress()
    failed_tasks = tracker.tasks['failed_tasks']
    
    # 估计剩余时间
    # 获取最近的批次日志以估计每个任务的平均时间
    recent_batches = []
    batch_files = []
    for root, _, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith('_log.json'):
                batch_files.append(os.path.join(root, file))
    
    if batch_files:
        for file in batch_files[-10:]:  # 最近10个批次
            try:
                with open(file, 'r') as f:
                    batch_data = json.load(f)
                    recent_batches.append(batch_data)
            except:
                pass
    
    # 计算平均每任务时间
    if recent_batches:
        total_time = sum(batch['time_taken'] for batch in recent_batches)
        total_batch_tasks = sum(1 for _ in recent_batches)  # 每个批次50个任务
        avg_task_time = total_time / (total_batch_tasks * 50)
        remaining_tasks = total_tasks - completed_tasks
        estimated_time = remaining_tasks * avg_task_time / 3600  # 小时
        estimated_time_str = f"{estimated_time:.1f} 小时"
    else:
        estimated_time_str = "无法估计"
    
    # 获取药物和蛋白质覆盖率
    completed_file = f'{CHECKPOINTS_DIR}/completed_tasks.pkl'
    processed_drugs = set()
    processed_proteins = set()
    
    if os.path.exists(completed_file):
        with open(completed_file, 'rb') as f:
            completed_tasks_set = pickle.load(f)
        
        # 从任务ID中提取药物和蛋白质ID
        for task_id in completed_tasks_set:
            for file in os.listdir(RESULTS_DIR):
                if file.endswith('_result.json'):
                    parts = file.split('_')
                    if len(parts) >= 2:
                        drug_id = parts[0]
                        protein_id = parts[1]
                        processed_drugs.add(drug_id)
                        processed_proteins.add(protein_id)
    
    # 获取总药物和蛋白质数量
    total_drugs = 0
    total_proteins = 0
    
    if os.path.exists(f'{BASE_DIR}/all_drugs.csv'):
        drugs_df = pd.read_csv(f'{BASE_DIR}/all_drugs.csv')
        total_drugs = len(drugs_df)
    
    if os.path.exists(f'{BASE_DIR}/all_proteins.csv'):
        proteins_df = pd.read_csv(f'{BASE_DIR}/all_proteins.csv')
        total_proteins = len(proteins_df)
    
    # 计算覆盖率
    drug_coverage = 0 if total_drugs == 0 else len(processed_drugs) / total_drugs * 100
    protein_coverage = 0 if total_proteins == 0 else len(processed_proteins) / total_proteins * 100
    
    # 生成最近批次表格
    recent_batches_table = ""
    for batch in sorted(recent_batches, key=lambda x: x['batch_id'], reverse=True)[:5]:
        recent_batches_table += f"| {batch['batch_id']} | {time.strftime('%Y-%m-%d %H:%M', time.localtime(batch['completed_at']))} | 50 | {batch['time_taken']:.1f} | {batch['time_taken']/50:.1f} |\n"
    
    # 生成Markdown文本
    dashboard_text = f"""# 项目监控仪表板

## 项目总体进度

- 总任务数: {total_tasks}
- 已完成任务: {completed_tasks}
- 完成百分比: {progress_percentage:.2f}%
- 失败任务数: {failed_tasks}
- 预计剩余时间: {estimated_time_str}

## 最近批次统计

| 批次ID | 完成时间 | 任务数 | 用时(秒) | 平均每任务(秒) |
|-------|---------|-------|---------|--------------|
{recent_batches_table}

## 药物覆盖率

- 已处理药物数: {len(processed_drugs)}/{total_drugs}
- 药物覆盖率: {drug_coverage:.2f}%

## 蛋白质覆盖率

- 已处理蛋白质数: {len(processed_proteins)}/{total_proteins}
- 蛋白质覆盖率: {protein_coverage:.2f}%
"""
    
    # 保存到文件
    dashboard_file = f'{BASE_DIR}/progress_dashboard.md'
    with open(dashboard_file, 'w') as f:
        f.write(dashboard_text)
    
    print(f"进度仪表板已更新: {dashboard_file}")
    return dashboard_text

if __name__ == "__main__":
    main() 