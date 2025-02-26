# 数据源适配器基类和实现

import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import json
import gzip
import io
from abc import ABC, abstractmethod
from Bio import PDB, SeqIO
import re

# 全局设置
BASE_DIR = '/content/drive/MyDrive/full_docking_project'
DATA_DIR = f'{BASE_DIR}/data_sources'
os.makedirs(DATA_DIR, exist_ok=True)

class DataSourceAdapter(ABC):
    """数据源适配器基类"""
    
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.data_dir = f'{base_dir}/data_sources'
        os.makedirs(self.data_dir, exist_ok=True)
    
    @abstractmethod
    def fetch_data(self, **kwargs):
        """获取数据"""
        pass
    
    @abstractmethod
    def parse_data(self, raw_data, **kwargs):
        """解析数据"""
        pass
    
    @abstractmethod
    def transform_data(self, parsed_data, **kwargs):
        """转换数据为标准格式"""
        pass
    
    def save_data(self, data, filename):
        """保存数据到本地"""
        # 根据文件扩展名选择保存方式
        if filename.endswith('.csv'):
            data.to_csv(f'{self.data_dir}/{filename}', index=False)
        elif filename.endswith('.json'):
            with open(f'{self.data_dir}/{filename}', 'w') as f:
                json.dump(data, f)
        elif filename.endswith('.pkl'):
            data.to_pickle(f'{self.data_dir}/{filename}')
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        print(f"数据已保存到: {self.data_dir}/{filename}")
    
    def load_data(self, filename):
        """从本地加载数据"""
        file_path = f'{self.data_dir}/{filename}'
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        # 根据文件扩展名选择加载方式
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.pkl'):
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def make_request(self, url, headers=None, params=None, timeout=30):
        """发送HTTP请求"""
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"请求失败: {url}\n错误: {e}")
            return None

class PDBAdapter(DataSourceAdapter):
    """PDB数据源适配器"""
    
    def __init__(self, base_dir=BASE_DIR):
        super().__init__(base_dir)
        self.pdb_dir = f'{self.base_dir}/proteins/pdb'
        os.makedirs(self.pdb_dir, exist_ok=True)
    
    def fetch_data(self, uniprot_ids=None, limit=None):
        """获取PDB结构数据
        
        Args:
            uniprot_ids: UniProt ID列表，如果为None则使用所有人类蛋白质
            limit: 限制结构数量
        """
        output_file = f'{self.data_dir}/pdb_structures.csv'
        
        if os.path.exists(output_file) and uniprot_ids is None:
            print(f"使用现有PDB数据: {output_file}")
            return self.load_data('pdb_structures.csv')
        
        # 获取UniProt ID列表
        if uniprot_ids is None:
            # 使用UniProt适配器获取所有人类蛋白质
            uniprot_adapter = UniProtAdapter(self.base_dir)
            proteins_df = uniprot_adapter.fetch_data(limit=limit)
            uniprot_ids = proteins_df['uniprot_id'].tolist()
        
        # 准备存储PDB结构信息
        pdb_structures = []
        
        # 使用UniProt到PDB的映射服务
        base_url = "https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures/"
        
        # 分批处理UniProt ID，每次请求不超过10个
        batch_size = 10
        for i in range(0, len(uniprot_ids), batch_size):
            batch = uniprot_ids[i:i+batch_size]
            for uniprot_id in batch:
                url = f"{base_url}{uniprot_id}"
                response = self.make_request(url)
                
                if response and response.status_code == 200:
                    data = response.json()
                    
                    # 处理响应数据
                    if uniprot_id in data:
                        structures = data[uniprot_id]
                        for structure in structures:
                            pdb_id = structure['pdb_id'].lower()
                            # 获取结构元数据
                            for mapping in structure['mappings']:
                                chain_id = mapping['chain_id']
                                resolution = structure.get('resolution', None)
                                experiment_type = structure.get('experimental_method', '')
                                
                                # 添加到结构列表
                                pdb_structures.append({
                                    'pdb_id': pdb_id,
                                    'uniprot_id': uniprot_id,
                                    'chain_id': chain_id,
                                    'resolution': resolution,
                                    'experiment_type': experiment_type,
                                    'structure_type': 'experimental'
                                })
                                
                                # 下载PDB文件
                                self.download_pdb(pdb_id)
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(pdb_structures)
        
        # 限制数量
        if limit and len(df) > limit:
            df = df.head(limit)
        
        self.save_data(df, 'pdb_structures.csv')
        return df
    
    def parse_data(self, raw_data, **kwargs):
        """解析PDB数据"""
        # PDB数据已在fetch_data中解析为DataFrame
        return raw_data
    
    def transform_data(self, parsed_data, **kwargs):
        """转换PDB数据为标准格式"""
        # 已经是标准格式，无需转换
        return parsed_data
    
    def download_pdb(self, pdb_id):
        """下载PDB结构文件"""
        pdb_id = pdb_id.lower()
        output_file = f'{self.pdb_dir}/pdb_{pdb_id}.pdb'
        
        if os.path.exists(output_file):
            # 检查文件是否损坏
            if os.path.getsize(output_file) > 1000:  # 正常PDB文件至少几KB
                return output_file
            else:
                os.remove(output_file)
        
        # 尝试从RCSB PDB下载
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = self.make_request(url)
        
        if response and response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return output_file
        
        # 尝试从PDBe下载
        url = f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pdb_id}.ent"
        response = self.make_request(url)
        
        if response and response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return output_file
        
        print(f"无法下载PDB结构: {pdb_id}")
        return None

class PDBbindAdapter(DataSourceAdapter):
    """PDBbind数据源适配器"""
    
    def __init__(self, base_dir=BASE_DIR):
        super().__init__(base_dir)
        self.pdbbind_dir = f'{self.data_dir}/pdbbind'
        os.makedirs(self.pdbbind_dir, exist_ok=True)
    
    def fetch_data(self, dataset='refined', limit=None):
        """获取PDBbind数据
        
        Args:
            dataset: 使用的数据集，'general'，'refined'或'core'
            limit: 限制结构数量
        """
        output_file = f'{self.data_dir}/pdbbind_{dataset}_set.csv'
        
        if os.path.exists(output_file):
            df = self.load_data(f'pdbbind_{dataset}_set.csv')
            if limit:
                return df.head(limit)
            return df
        
        # 由于PDBbind需要注册访问，这里提供下载说明
        print(f"请按照以下步骤获取PDBbind {dataset}集数据:")
        print("1. 访问 http://pdbbind.org.cn/")
        print("2. 注册并登录")
        print("3. 下载 PDBbind v2020 数据集")
        print("4. 解压缩 index 文件夹")
        print(f"5. 找到 {dataset} 数据集的索引文件")
        print("6. 将索引文件放在 data_sources/pdbbind 目录下")
        
        # 尝试查找用户可能已下载的索引文件
        index_file = f'{self.pdbbind_dir}/INDEX_{dataset}_data.2020'
        if os.path.exists(index_file):
            print(f"找到PDBbind索引文件: {index_file}")
            return self.parse_data(index_file, dataset=dataset, limit=limit)
        
        raise RuntimeError("无法获取真实PDBbind数据。请确保已下载并正确放置PDBbind索引文件，且网络连接正常。")
    
    def parse_data(self, index_file, dataset='refined', limit=None):
        """解析PDBbind索引文件
        
        Args:
            index_file: 索引文件路径
            dataset: 数据集类型
            limit: 限制结构数量
        """
        pdbbind_data = []
        
        try:
            with open(index_file, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    if line.startswith('#'):  # 跳过注释行
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        pdb_id = parts[0].lower()
                        resolution = float(parts[1]) if parts[1] != 'None' else None
                        affinity_type = parts[2]  # Ki, Kd, IC50
                        affinity_value = float(parts[3])
                        
                        # 尝试获取UniProt ID
                        uniprot_id = self._get_uniprot_from_pdb(pdb_id)
                        
                        # 处理配体信息
                        ligand_id = None
                        ligand_name = None
                        if len(parts) > 4:
                            ligand_info = ' '.join(parts[4:])
                            # 解析配体信息
                            if '/' in ligand_info:
                                ligand_parts = ligand_info.split('/')
                                ligand_id = ligand_parts[0].strip()
                                ligand_name = ligand_parts[1].strip() if len(ligand_parts) > 1 else None
                        
                        pdbbind_data.append({
                            'pdb_id': pdb_id,
                            'uniprot_id': uniprot_id,
                            'resolution': resolution,
                            'affinity_type': affinity_type,
                            'affinity_value': affinity_value,
                            'ligand_id': ligand_id,
                            'ligand_name': ligand_name
                        })
            
            # 限制数量
            if limit and len(pdbbind_data) > limit:
                pdbbind_data = pdbbind_data[:limit]
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(pdbbind_data)
            self.save_data(df, f'pdbbind_{dataset}_set.csv')
            return df
            
        except Exception as e:
            print(f"解析PDBbind索引文件出错: {e}")
            return pd.DataFrame()
    
    def transform_data(self, parsed_data, **kwargs):
        """转换PDBbind数据为标准格式"""
        # 已经是标准格式，无需转换
        return parsed_data
    
    def _get_uniprot_from_pdb(self, pdb_id):
        """从PDB ID获取UniProt ID"""
        try:
            url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
            response = self.make_request(url)
            
            if response and response.status_code == 200:
                data = response.json()
                
                if pdb_id in data:
                    uniprot_mappings = data[pdb_id].get('UniProt', {})
                    if uniprot_mappings:
                        # 取第一个UniProt ID
                        return list(uniprot_mappings.keys())[0]
            
            return None
        except:
            return None 

class BindingDBAdapter(DataSourceAdapter):
    """BindingDB数据源适配器"""
    
    def __init__(self, base_dir=BASE_DIR):
        super().__init__(base_dir)
    
    def fetch_data(self, limit=None):
        """获取BindingDB数据
        
        Args:
            limit: 限制数据条数
        """
        output_file = f'{self.data_dir}/bindingdb_data.csv'
        
        if os.path.exists(output_file):
            df = self.load_data('bindingdb_data.csv')
            if limit:
                return df.head(limit)
            return df
        
        # BindingDB提供多种下载选项，这里使用REST API示例
        print("获取BindingDB数据...")
        print("注意: 完整的BindingDB数据集非常大，这里只获取人类蛋白质的部分数据")
        
        # 使用REST API获取人类蛋白质的结合数据
        try:
            # 获取前N条人类蛋白质数据
            fetch_limit = limit if limit else 100
            url = "https://www.bindingdb.org/rests/target/uniprot/P00533"  # 示例: EGFR
            response = self.make_request(url)
            
            if response and response.status_code == 200:
                data = response.json()
                
                # 解析数据
                binding_data = []
                if 'data' in data:
                    for item in data['data']:
                        binding_data.append({
                            'protein_id': item.get('uniprotID'),
                            'protein_name': item.get('name'),
                            'ligand_id': item.get('ligandID'),
                            'ligand_name': item.get('ligandName'),
                            'affinity_type': item.get('affinityType'),
                            'affinity_value': item.get('affinityValue'),
                            'binding_constant': item.get('Ki'),
                            'ic50': item.get('IC50'),
                            'target_organism': item.get('organism')
                        })
                
                # 转换为DataFrame并保存
                df = pd.DataFrame(binding_data)
                self.save_data(df, 'bindingdb_data.csv')
                
                if limit:
                    return df.head(limit)
                return df
            
        except Exception as e:
            print(f"获取BindingDB数据出错: {e}")
        
        raise RuntimeError("无法获取真实BindingDB数据。请检查网络连接或通过其他方式预先下载BindingDB数据。")
    
    def parse_data(self, raw_data, **kwargs):
        """解析BindingDB数据"""
        # 在fetch_data中已经解析为DataFrame
        return raw_data
    
    def transform_data(self, parsed_data, **kwargs):
        """转换BindingDB数据为标准格式"""
        # 已经是标准格式，无需转换
        return parsed_data 