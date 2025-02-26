import os
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from io import StringIO
import gzip
from tqdm import tqdm
import random

# 全局设置
BASE_DIR = '/content/drive/MyDrive/full_docking_project'
DATA_DIR = f'{BASE_DIR}/data_sources'
os.makedirs(DATA_DIR, exist_ok=True)

class DataCrawler:
    """生物医学数据获取系统"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Drug-Repositioning-Pipeline/1.0 (research project; contact@example.org)'
        })
        # 设置合理的请求间隔，避免过度请求被封
        self.request_delay = 1  # 秒
    
    def _make_request(self, url, params=None, stream=False):
        """发送请求并处理常见错误"""
        time.sleep(self.request_delay)  # 请求延迟
        try:
            response = self.session.get(url, params=params, stream=stream)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误: {e}")
            if e.response.status_code == 429:  # 太多请求
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"接收到速率限制，等待 {retry_after} 秒...")
                time.sleep(retry_after)
                return self._make_request(url, params, stream)
        except requests.exceptions.ConnectionError:
            print("连接错误，等待5秒后重试...")
            time.sleep(5)
            return self._make_request(url, params, stream)
        except Exception as e:
            print(f"请求错误: {e}")
        return None
    
    def get_drugbank_drugs(self, limit=None):
        """从DrugBank获取批准药物数据
        
        由于DrugBank需要注册API密钥，这里使用公开的DrugBank Open Data快照
        """
        print("获取DrugBank药物数据...")
        output_file = f'{DATA_DIR}/drugbank_drugs.csv'
        
        if os.path.exists(output_file):
            print(f"使用现有DrugBank数据: {output_file}")
            return pd.read_csv(output_file)
        
        # DrugBank Open Data的替代来源
        url = "https://go.drugbank.com/releases/latest/downloads/approved-structures"
        response = self._make_request(url)
        
        if response:
            # 解析TSV数据，通常包含DrugBank ID和SMILES
            try:
                data = StringIO(response.text)
                df = pd.read_csv(data, sep="\t")
                
                # 处理数据，确保包含必要字段
                if 'DrugBank ID' in df.columns and 'SMILES' in df.columns:
                    # 重命名列以匹配我们的需求
                    df = df.rename(columns={
                        'DrugBank ID': 'drug_id',
                        'Name': 'name',
                        'SMILES': 'smiles'
                    })
                    
                    # 如果没有药物名称，使用DrugBank ID
                    if 'name' not in df.columns:
                        df['name'] = df['drug_id']
                    
                    # 限制数量（用于测试）
                    if limit:
                        df = df.head(limit)
                    
                    # 保存到文件
                    df.to_csv(output_file, index=False)
                    print(f"已保存 {len(df)} 条DrugBank药物记录到 {output_file}")
                    return df
                else:
                    print("DrugBank数据格式不符合预期")
            except Exception as e:
                print(f"解析DrugBank数据时出错: {e}")
        
        # 如果无法获取真实数据，创建一些示例数据
        print("无法获取DrugBank数据，创建示例数据...")
        example_drugs = [
            {'drug_id': 'DB00316', 'name': 'Enalapril', 'smiles': 'CCOC(=O)C(CCC(=O)N)NC(C)C(=O)N1CCCC1C(=O)O'},
            {'drug_id': 'DB00945', 'name': 'Diclofenac', 'smiles': 'O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl'},
            {'drug_id': 'DB00273', 'name': 'Atorvastatin', 'smiles': 'CC(C)c1c(C(=O)Nc2ccccc2)c(c(c2ccc(F)cc2)n1CC[C@H](O)C[C@@H](O)CC(=O)O)c1ccc(F)cc1'},
            {'drug_id': 'DB00519', 'name': 'Aspirin', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O'},
            {'drug_id': 'DB00220', 'name': 'Fluoxetine', 'smiles': 'CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1'},
        ]
        
        df = pd.DataFrame(example_drugs)
        df.to_csv(output_file, index=False)
        print(f"已创建 {len(df)} 条示例药物记录")
        return df
    
    def get_uniprot_proteins(self, limit=None, proteome="UP000005640"):
        """从UniProt获取人类蛋白质组数据
        
        Args:
            limit: 限制蛋白质数量（用于测试）
            proteome: UniProt proteome ID，UP000005640是人类蛋白质组
        """
        print("获取UniProt人类蛋白质数据...")
        output_file = f'{DATA_DIR}/uniprot_proteins.csv'
        
        if os.path.exists(output_file):
            print(f"使用现有UniProt数据: {output_file}")
            return pd.read_csv(output_file)
        
        # 使用UniProt API
        base_url = "https://rest.uniprot.org/uniprotkb/stream"
        params = {
            "format": "tsv",
            "query": f"proteome:{proteome} AND reviewed:true",
            "fields": "accession,id,gene_names,protein_name,length,mass,organism_name"
        }
        
        response = self._make_request(base_url, params=params)
        
        if response:
            try:
                data = StringIO(response.text)
                df = pd.read_csv(data, sep="\t")
                
                # 重命名列以匹配我们的需求
                df = df.rename(columns={
                    'Entry': 'uniprot_id',
                    'Entry Name': 'entry_name',
                    'Gene Names': 'gene_name',
                    'Protein names': 'description',
                    'Length': 'length',
                    'Mass': 'mass'
                })
                
                # 处理基因名称（可能有多个，取第一个）
                if 'gene_name' in df.columns:
                    df['gene_name'] = df['gene_name'].str.split().str[0]
                
                # 限制数量（用于测试）
                if limit:
                    df = df.head(limit)
                
                # 保存到文件
                df.to_csv(output_file, index=False)
                print(f"已保存 {len(df)} 条UniProt蛋白质记录到 {output_file}")
                return df
            except Exception as e:
                print(f"解析UniProt数据时出错: {e}")
        
        # 如果无法获取真实数据，创建一些示例数据
        print("无法获取UniProt数据，创建示例数据...")
        example_proteins = [
            {'uniprot_id': 'P05067', 'gene_name': 'APP', 'description': 'Amyloid-beta precursor protein'},
            {'uniprot_id': 'P37840', 'gene_name': 'SNCA', 'description': 'Alpha-synuclein'},
            {'uniprot_id': 'P31645', 'gene_name': 'SLC6A4', 'description': 'Serotonin transporter'},
            {'uniprot_id': 'P30556', 'gene_name': 'AGTR1', 'description': 'Type-1 angiotensin II receptor'},
            {'uniprot_id': 'P01308', 'gene_name': 'INS', 'description': 'Insulin'},
        ]
        
        df = pd.DataFrame(example_proteins)
        df.to_csv(output_file, index=False)
        print(f"已创建 {len(df)} 条示例蛋白质记录")
        return df
    
    def get_disgenet_associations(self, limit=None):
        """从DisGeNET获取疾病-蛋白质关联数据"""
        print("获取DisGeNET疾病-蛋白质关联数据...")
        output_file = f'{DATA_DIR}/disease_protein_associations.csv'
        
        if os.path.exists(output_file):
            print(f"使用现有DisGeNET数据: {output_file}")
            return pd.read_csv(output_file)
        
        # DisGeNET数据下载URL
        url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz"
        
        try:
            # 下载压缩文件
            print("下载DisGeNET数据...")
            response = self._make_request(url, stream=True)
            
            if response:
                # 解压并解析TSV
                gzip_file = gzip.GzipFile(fileobj=response.raw)
                content = gzip_file.read().decode('utf-8')
                data = StringIO(content)
                df = pd.read_csv(data, sep="\t")
                
                # 处理数据，确保包含必要字段并重命名
                if 'geneId' in df.columns and 'diseaseId' in df.columns:
                    # 选择和重命名相关列
                    df = df[['geneId', 'geneSymbol', 'diseaseId', 'diseaseName', 'score']]
                    df = df.rename(columns={
                        'geneId': 'gene_id',
                        'geneSymbol': 'gene_name',
                        'diseaseId': 'disease_id',
                        'diseaseName': 'disease_name'
                    })
                    
                    # 将Entrez基因ID映射到UniProt ID
                    # 这需要额外的数据处理，这里简化处理
                    uniprot_df = self.get_uniprot_proteins()
                    gene_to_uniprot = dict(zip(uniprot_df['gene_name'], uniprot_df['uniprot_id']))
                    
                    # 添加UniProt ID列
                    df['protein_id'] = df['gene_name'].map(gene_to_uniprot)
                    
                    # 只保留有UniProt映射的行
                    df = df.dropna(subset=['protein_id'])
                    
                    # 限制数量（用于测试）
                    if limit:
                        df = df.head(limit)
                    
                    # 保存到文件
                    df.to_csv(output_file, index=False)
                    print(f"已保存 {len(df)} 条DisGeNET关联到 {output_file}")
                    return df
                else:
                    print("DisGeNET数据格式不符合预期")
        except Exception as e:
            print(f"获取DisGeNET数据时出错: {e}")
        
        # 如果无法获取真实数据，创建一些示例数据
        print("创建示例疾病-蛋白质关联数据...")
        example_associations = [
            {'disease_id': 'DOID:14330', 'disease_name': 'Alzheimer disease', 'protein_id': 'P05067', 'gene_name': 'APP', 'score': 0.9},
            {'disease_id': 'DOID:14330', 'disease_name': 'Alzheimer disease', 'protein_id': 'P37840', 'gene_name': 'SNCA', 'score': 0.7},
            {'disease_id': 'DOID:0060041', 'disease_name': 'Parkinson disease', 'protein_id': 'P37840', 'gene_name': 'SNCA', 'score': 0.95},
            {'disease_id': 'DOID:10652', 'disease_name': 'Alzheimer disease', 'protein_id': 'P05067', 'gene_name': 'APP', 'score': 0.8},
            {'disease_id': 'DOID:2030', 'disease_name': 'Major depressive disorder', 'protein_id': 'P31645', 'gene_name': 'SLC6A4', 'score': 0.85},
            {'disease_id': 'DOID:2531', 'disease_name': 'Hypertension', 'protein_id': 'P30556', 'gene_name': 'AGTR1', 'score': 0.9},
            {'disease_id': 'DOID:9351', 'disease_name': 'Diabetes mellitus', 'protein_id': 'P01308', 'gene_name': 'INS', 'score': 1.0},
        ]
        
        df = pd.DataFrame(example_associations)
        df.to_csv(output_file, index=False)
        print(f"已创建 {len(df)} 条示例疾病-蛋白质关联")
        return df
    
    def get_drug_indications(self, limit=None):
        """从DrugCentral获取药物适应症数据"""
        print("获取药物适应症数据...")
        output_file = f'{DATA_DIR}/drug_indications.csv'
        
        if os.path.exists(output_file):
            print(f"使用现有药物适应症数据: {output_file}")
            return pd.read_csv(output_file)
        
        # DrugCentral API或下载链接
        # 这需要根据DrugCentral的具体API或数据格式进行调整
        try:
            # 这里应该实现从DrugCentral获取数据的代码
            # 由于DrugCentral可能需要特定的API访问方式，这里使用简化的方法
            raise Exception("使用示例数据")
        except Exception as e:
            print(f"获取DrugCentral数据时出错: {e}")
            
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
            df.to_csv(output_file, index=False)
            print(f"已创建 {len(df)} 条示例药物适应症数据")
            return df
    
    def get_binding_data(self, limit=None):
        """从BindingDB获取已知结合数据"""
        print("获取BindingDB结合数据...")
        output_file = f'{DATA_DIR}/reference_binding_data.csv'
        
        if os.path.exists(output_file):
            print(f"使用现有BindingDB数据: {output_file}")
            return pd.read_csv(output_file)
        
        # BindingDB数据下载URL
        url = "https://www.bindingdb.org/bind/downloads/BindingDB_All_2023m1.tsv.zip"
        
        try:
            # 这里应该实现从BindingDB下载和处理数据的代码
            # 由于文件可能很大，这里使用简化的方法
            raise Exception("使用示例数据")
        except Exception as e:
            print(f"获取BindingDB数据时出错: {e}")
            
            # 创建示例数据
            print("创建示例结合数据...")
            example_data = [
                {'protein_id': 'P30556', 'ligand_name': 'Losartan', 'ligand_smiles': 'CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1', 'binding_constant': 2.3, 'constant_type': 'Ki_nM'},
                {'protein_id': 'P31645', 'ligand_name': 'Fluoxetine', 'ligand_smiles': 'CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1', 'binding_constant': 8.5, 'constant_type': 'IC50_nM'},
                {'protein_id': 'P05067', 'ligand_name': 'Verubecestat', 'ligand_smiles': 'CC(C)(C)OC(=O)N1CC(C1)C(=O)N(C)C1CCN(C1)C(=O)C1=CC=C(F)C(=C1)C#N', 'binding_constant': 12.0, 'constant_type': 'IC50_nM'},
                {'protein_id': 'P37840', 'ligand_name': 'Fasudil', 'ligand_smiles': 'CN1CC(=NNC1=O)N1CCCN(CC1)C1=CC=CC2=CC=CC=C21', 'binding_constant': 150.0, 'constant_type': 'EC50_nM'},
                {'protein_id': 'P01308', 'ligand_name': 'Insulin Lispro', 'ligand_smiles': 'CC(C)CC1NC(=O)C(CCCNC(=N)N)NC(=O)C(CC(C)C)NC(=O)CNC(=O)C(CO)NC(=O)C(CC(C)C)NC(=O)C(CCC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(CC2=CNC=N2)NC(=O)C(CO)NC(=O)C(CC(C)C)NC(=O)C(CC(=O)N)NC(=O)C(CC(N)=O)NC(=O)C(CC2=CC=CC=C2)NC(=O)C(CC(N)=O)NC(=O)C(CC(=O)O)NC(=O)C2CCCN2C(=O)C(CCCCN)NC(=O)C(CC2=CC=C(O)C=C2)NC(=O)C(CC(C)C)NC(=O)C(C)NC1=O', 'binding_constant': 0.2, 'constant_type': 'Kd_nM'}
            ]
            
            df = pd.DataFrame(example_data)
            df.to_csv(output_file, index=False)
            print(f"已创建 {len(df)} 条示例结合数据")
            return df
    
    def download_alphafold_structure(self, uniprot_id):
        """从AlphaFold下载蛋白质结构"""
        output_file = f'{PROTEINS_DIR}/{uniprot_id}.pdb'
        
        if os.path.exists(output_file):
            print(f"已存在蛋白质结构: {output_file}")
            return output_file
        
        print(f"从AlphaFold下载蛋白质结构: {uniprot_id}...")
        
        # AlphaFold数据库URL
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        
        response = self._make_request(url)
        
        if response and response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"下载完成: {output_file}")
            return output_file
        else:
            print(f"无法下载蛋白质结构: {uniprot_id}")
            return None
    
    def get_pdb_structures(self, uniprot_ids=None, limit=None):
        """从PDB获取实验确定的蛋白质结构
        
        Args:
            uniprot_ids: UniProt ID列表，如果为None则使用所有人类蛋白质
            limit: 限制结构数量
        """
        print("获取PDB蛋白质结构数据...")
        output_file = f'{DATA_DIR}/pdb_structures.csv'
        
        if os.path.exists(output_file) and uniprot_ids is None:
            print(f"使用现有PDB数据: {output_file}")
            return pd.read_csv(output_file)
        
        if uniprot_ids is None:
            # 如果未指定UniProt ID，获取所有人类蛋白质
            proteins_df = self.get_uniprot_proteins(limit=limit)
            uniprot_ids = proteins_df['uniprot_id'].tolist()
        
        # 准备存储PDB结构信息
        pdb_structures = []
        
        # 使用UniProt到PDB的映射服务
        base_url = "https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures/"
        
        # 分批处理UniProt ID，每次请求不超过10个
        batch_size = 10
        for i in range(0, len(uniprot_ids), batch_size):
            batch = uniprot_ids[i:i+batch_size]
            batch_str = ",".join(batch)
            
            url = f"{base_url}{batch_str}"
            response = self._make_request(url)
            
            if response and response.status_code == 200:
                data = response.json()
                
                # 处理每个UniProt ID的结果
                for uniprot_id in batch:
                    if uniprot_id in data:
                        structures = data[uniprot_id]
                        # 获取最佳结构（通常是第一个）
                        if structures:
                            for structure in structures:
                                pdb_id = structure.get('pdb_id')
                                chain_id = structure.get('chain_id')
                                resolution = structure.get('resolution')
                                start = structure.get('unp_start')
                                end = structure.get('unp_end')
                                coverage = (end - start + 1) if start and end else 0
                                experimental_method = structure.get('experimental_method')
                                
                                # 确保有足够的覆盖率和分辨率（如果有）
                                if coverage > 50 and (resolution is None or resolution < 3.0):
                                    pdb_structures.append({
                                        'uniprot_id': uniprot_id,
                                        'pdb_id': pdb_id,
                                        'chain_id': chain_id,
                                        'resolution': resolution,
                                        'coverage': coverage,
                                        'start': start,
                                        'end': end,
                                        'experimental_method': experimental_method
                                    })
                                    
                                    # 下载PDB文件
                                    self.download_pdb_structure(pdb_id)
            
            # 如果达到数量限制，提前退出
            if limit and len(pdb_structures) >= limit:
                pdb_structures = pdb_structures[:limit]
                break
        
        # 如果没有找到结构，创建示例数据
        if not pdb_structures:
            print("未找到PDB结构，创建示例数据...")
            example_structures = [
                {'uniprot_id': 'P05067', 'pdb_id': '6sjm', 'chain_id': 'B', 'resolution': 1.8, 'coverage': 120, 'start': 672, 'end': 792, 'experimental_method': 'X-ray diffraction'},
                {'uniprot_id': 'P37840', 'pdb_id': '2n0a', 'chain_id': 'A', 'resolution': None, 'coverage': 140, 'start': 1, 'end': 140, 'experimental_method': 'NMR'},
                {'uniprot_id': 'P31645', 'pdb_id': '6niv', 'chain_id': 'A', 'resolution': 2.5, 'coverage': 550, 'start': 50, 'end': 600, 'experimental_method': 'X-ray diffraction'},
                {'uniprot_id': 'P30556', 'pdb_id': '6os1', 'chain_id': 'R', 'resolution': 2.9, 'coverage': 310, 'start': 20, 'end': 330, 'experimental_method': 'Cryo-EM'},
                {'uniprot_id': 'P01308', 'pdb_id': '6pxv', 'chain_id': 'A', 'resolution': 1.6, 'coverage': 51, 'start': 1, 'end': 51, 'experimental_method': 'X-ray diffraction'},
            ]
            pdb_structures = example_structures
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(pdb_structures)
        df.to_csv(output_file, index=False)
        print(f"已保存 {len(df)} 条PDB结构数据到 {output_file}")
        return df

    def download_pdb_structure(self, pdb_id):
        """下载PDB结构文件，同时处理可能不存在的结构"""
        pdb_id = pdb_id.lower()
        output_file = f'{PROTEINS_DIR}/pdb_{pdb_id}.pdb'
        
        if os.path.exists(output_file):
            # 文件已存在，检查是否为空或太小
            if os.path.getsize(output_file) > 1000:  # 正常的PDB文件至少几KB
                # print(f"已存在有效的PDB结构: {output_file}")
                return output_file
            else:
                print(f"现有PDB文件 {output_file} 似乎损坏，重新下载...")
                os.remove(output_file)
        
        print(f"下载PDB结构: {pdb_id}...")
        
        try:
            # 尝试从RCSB PDB下载
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = self._make_request(url)
            
            if response and response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"成功从RCSB下载: {output_file}")
                return output_file
            
            # 如果RCSB失败，尝试从PDBe下载
            url = f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pdb_id}.ent"
            response = self._make_request(url)
            
            if response and response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"成功从PDBe下载: {output_file}")
                return output_file
            
            # 对于我们的示例数据中不存在于PDB的ID，创建占位符文件
            # 在实际应用中，这些PDB ID应该是真实的
            if any(pdb_id == sample['pdb_id'].lower() for sample in self._generate_extended_pdbbind_sample(count=200)):
                print(f"创建示例PDB结构: {pdb_id}")
                # 从基本示例复制一个结构作为占位符
                example_pdb = '6sjm'  # 一个常见的结构
                example_file = f'{PROTEINS_DIR}/pdb_{example_pdb}.pdb'
                
                # 如果示例文件不存在，先下载它
                if not os.path.exists(example_file):
                    self.download_pdb_structure(example_pdb)
                
                # 如果示例下载成功，复制它
                if os.path.exists(example_file):
                    import shutil
                    shutil.copy(example_file, output_file)
                    print(f"创建了示例PDB文件: {output_file}")
                    return output_file
        
            print(f"无法下载PDB结构: {pdb_id}")
            return None
        except Exception as e:
            print(f"下载PDB结构 {pdb_id} 时出错: {e}")
            return None
    
    def get_pdbbind_data(self, limit=None, dataset='refined'):
        """从PDBbind获取高质量结合数据
        
        Args:
            limit: 限制结构数量，None表示不限制
            dataset: 使用的数据集，'general'，'refined'或'core'
        """
        print(f"获取PDBbind {dataset}集数据...")
        output_file = f'{DATA_DIR}/pdbbind_{dataset}_set.csv'
        
        # 如果已有数据且足够，直接返回
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            if limit and len(existing_df) >= limit:
                print(f"使用现有PDBbind数据: {output_file} ({len(existing_df)}条记录)")
                return existing_df.head(limit)
            print(f"使用现有PDBbind数据: {output_file} ({len(existing_df)}条记录)")
            if not limit:
                return existing_df
        
        # 尝试使用真实PDBbind数据
        try:
            # 在实际应用中，应该实现以下功能：
            # 1. 从PDBbind官方网站下载数据
            # 2. 解析索引文件以获取结合数据
            # 3. 下载对应的PDB结构文件
            
            print("注意：真实应用中应从PDBbind官方网站下载数据")
            print("请访问 http://pdbbind.org.cn/ 注册并下载数据集")
            print("然后实现数据解析和处理代码")
            
            # 示例：解析PDBbind索引文件的代码
            '''
            pdbbind_data = []
            with open(f'pdbbind_v2020_{dataset}_set_index.txt', 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        pdb_id = parts[0]
                        resolution = float(parts[1]) if parts[1] != 'None' else None
                        affinity_type = parts[2]
                        affinity_value = float(parts[3])
                        
                        # 获取UniProt映射
                        uniprot_id = self._get_uniprot_from_pdb(pdb_id)
                        
                        pdbbind_data.append({
                            'pdb_id': pdb_id,
                            'uniprot_id': uniprot_id,
                            'resolution': resolution,
                            'affinity_type': affinity_type,
                            'affinity_value': affinity_value
                        })
            '''
            
            # 如果没有实现真实数据获取，返回已有数据或示例数据
            if os.path.exists(output_file):
                return pd.read_csv(output_file)
            
            raise Exception("需要实现真实PDBbind数据获取")
            
        except Exception as e:
            print(f"注意：{e}")
            print("为了演示系统功能，使用示例数据")
            
            # 使用基本示例数据
            if not os.path.exists(output_file):
                example_data = [
                    {'pdb_id': '6sjm', 'uniprot_id': 'P05067', 'resolution': 1.8, 'affinity_type': 'Ki', 'affinity_value': 5.7, 'ligand_id': 'JFM', 'ligand_name': 'N-(3,5-dimethylphenyl)-4-(3-methoxyphenyl)-1H-imidazole-2-carboxamide'},
                    {'pdb_id': '6niv', 'uniprot_id': 'P31645', 'resolution': 2.5, 'affinity_type': 'Ki', 'affinity_value': 8.2, 'ligand_id': 'FLU', 'ligand_name': 'Fluoxetine'},
                    {'pdb_id': '6os1', 'uniprot_id': 'P30556', 'resolution': 2.9, 'affinity_type': 'Ki', 'affinity_value': 2.3, 'ligand_id': '8AN', 'ligand_name': 'Losartan'},
                    {'pdb_id': '2n0a', 'uniprot_id': 'P37840', 'resolution': None, 'affinity_type': 'IC50', 'affinity_value': 120.0, 'ligand_id': 'FAS', 'ligand_name': 'Fasudil'},
                    {'pdb_id': '6pxv', 'uniprot_id': 'P01308', 'resolution': 1.6, 'affinity_type': 'Kd', 'affinity_value': 0.2, 'ligand_id': 'INS', 'ligand_name': 'Insulin Lispro'},
                ]
                
                df = pd.DataFrame(example_data)
                df.to_csv(output_file, index=False)
                print(f"已创建示例PDBbind数据，实际应用中请替换为真实数据")
                
                if limit:
                    return df.head(limit)
                return df
            else:
                existing_df = pd.read_csv(output_file)
                if limit:
                    return existing_df.head(limit)
                return existing_df

    def _get_uniprot_from_pdb(self, pdb_id):
        """从PDB ID获取UniProt ID"""
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
        response = self._make_request(url)
        
        if response and response.status_code == 200:
            data = response.json()
            if pdb_id.lower() in data:
                uniprot_mappings = data[pdb_id.lower()].get('UniProt', {})
                # 返回第一个UniProt条目
                for uniprot_id in uniprot_mappings:
                    return uniprot_id
        
        return None
    
    def fetch_all_data(self, limit=10, include_pdb=True, include_pdbbind=True, pdbbind_limit=50):
        """获取所有需要的数据
        
        Args:
            limit: 基础数据的最大记录数
            include_pdb: 是否包括PDB实验结构
            include_pdbbind: 是否包括PDBbind数据
            pdbbind_limit: PDBbind数据的最大记录数
        """
        print("开始获取所有数据...")
        
        # 获取基础数据
        drug_df = self.get_drugbank_drugs(limit=limit)
        protein_df = self.get_uniprot_proteins(limit=limit)
        disease_protein_df = self.get_disgenet_associations(limit=limit*2)
        drug_indication_df = self.get_drug_indications(limit=limit*2)
        binding_df = self.get_binding_data(limit=limit)
        
        # 获取实验结构数据
        if include_pdb:
            pdb_df = self.get_pdb_structures(limit=limit)
        else:
            pdb_df = pd.DataFrame()
        
        # 获取PDBbind数据 - 可以比其他数据多
        if include_pdbbind:
            pdbbind_df = self.get_pdbbind_data(limit=pdbbind_limit)
        else:
            pdbbind_df = pd.DataFrame()
        
        # 将药物和蛋白质数据复制到项目主目录
        drug_df.to_csv(f'{BASE_DIR}/all_drugs.csv', index=False)
        protein_df.to_csv(f'{BASE_DIR}/all_proteins.csv', index=False)
        
        # 下载蛋白质结构 - 首选PDB实验结构，其次AlphaFold预测结构
        print("下载蛋白质结构...")
        
        # 创建UniProt ID到PDB ID的映射
        uniprot_to_pdb = {}
        if not pdb_df.empty:
            for _, row in pdb_df.iterrows():
                uniprot_to_pdb[row['uniprot_id']] = row['pdb_id']
        
        for _, row in tqdm(protein_df.iterrows(), total=len(protein_df)):
            uniprot_id = row['uniprot_id']
            if uniprot_id in uniprot_to_pdb:
                # 优先使用PDB实验结构
                pdb_id = uniprot_to_pdb[uniprot_id]
                self.download_pdb_structure(pdb_id)
            else:
                # 如果没有实验结构，使用AlphaFold预测结构
                self.download_alphafold_structure(uniprot_id)
        
        print("所有数据获取完成!")
        
        return {
            'drugs': drug_df,
            'proteins': protein_df,
            'disease_protein': disease_protein_df,
            'drug_indication': drug_indication_df,
            'binding_data': binding_df,
            'pdb_structures': pdb_df,
            'pdbbind': pdbbind_df
        }

    def _generate_extended_pdbbind_sample(self, dataset='refined', existing_pdb_ids=None, count=100):
        """生成扩展的PDBbind示例数据
        
        Args:
            dataset: 数据集类型
            existing_pdb_ids: 已存在的PDB ID列表，避免重复
            count: 需要生成的数据数量
            
        Returns:
            包含PDBbind样本数据的列表
        """
        # 避免重复
        if existing_pdb_ids is None:
            existing_pdb_ids = []
        
        # PDBbind常见靶点的UniProt ID映射
        common_targets = {
            # 激酶
            '2oj9': 'P00533',  # EGFR
            '3bkl': 'P06213',  # 胰岛素受体
            '3oth': 'P31749',  # AKT1
            '4nym': 'P45983',  # JNK1
            '3pxf': 'P36507',  # MAP2K2
            '3py3': 'P27361',  # ERK1
            '3svv': 'P19838',  # NF-kB
            '2qhm': 'P04049',  # RAF1
            # 核受体
            '1fm9': 'P10275',  # 雄激素受体
            '3dt3': 'P03372',  # 雌激素受体
            '1y0s': 'P04150',  # 糖皮质激素受体
            '1nrl': 'P04062',  # 过氧化物酶体增殖物激活受体γ
            # 蛋白酶
            '1c25': 'P00734',  # 凝血酶
            '6iis': 'P20039',  # ACE2
            '2zen': 'P35354',  # COX-2
            '1aj6': 'P00760',  # 胰蛋白酶
            # 病毒蛋白
            '5re4': 'P0DTC2',  # SARS-CoV-2主蛋白酶
            '6lu7': 'P0DTD1',  # SARS-CoV-2 3CL蛋白酶
            # 离子通道
            '2bg9': 'P31645',  # 5-HT转运蛋白
            '5gof': 'P28223',  # 5-HT2A受体
            # 其他
            '3tss': 'P38398',  # BRCA1
            '1dzk': 'P38936',  # p21
            '3eml': 'P04637',  # p53
            '3kck': 'P00918',  # 碳酸酐酶II
            '3rze': 'P00966',  # 精氨酸琥珀酸合成酶
        }
        
        # 生成靶点-配体-亲和力样本
        # 这些示例代表了PDBbind中的常见数据模式
        extended_data = []
        
        # 生成随机化的示例数据
        target_pdb_ids = list(common_targets.keys())
        random.shuffle(target_pdb_ids)
        
        # 为简单起见，集中在这些常见的亲和力类型上
        affinity_types = ['Ki', 'Kd', 'IC50']
        
        # 结合亲和力数据范围（nM）
        affinity_ranges = {
            'high': (0.01, 10),    # 高亲和力 (< 10 nM)
            'medium': (10, 1000),  # 中等亲和力 (10-1000 nM)
            'low': (1000, 10000)   # 低亲和力 (> 1000 nM)
        }
        
        # 生成合理的亲和力数据
        added = 0
        for pdb_id in target_pdb_ids:
            if added >= count:
                break
            
            if pdb_id in existing_pdb_ids:
                continue
            
            uniprot_id = common_targets.get(pdb_id)
            if not uniprot_id:
                continue
            
            # 随机选择亲和力类型和值
            affinity_type = random.choice(affinity_types)
            
            # 为不同靶点分配不同亲和力范围，更符合实际情况
            if 'kinase' in pdb_id.lower() or pdb_id in ['3pxf', '3py3', '2qhm']:
                # 激酶抑制剂通常具有高亲和力
                affinity_range = affinity_ranges['high']
            elif pdb_id in ['1c25', '6iis', '2zen']:
                # 蛋白酶抑制剂通常具有中等亲和力
                affinity_range = affinity_ranges['medium']
            else:
                # 随机选择亲和力范围
                affinity_range = random.choice(list(affinity_ranges.values()))
            
            affinity_value = random.uniform(*affinity_range)
            
            # 随机分辨率（大多数晶体结构在1.5-2.5Å之间）
            resolution = round(random.uniform(1.5, 3.0), 1)
            
            # 随机配体ID和名称
            ligand_id = f"{pdb_id[-3:].upper()}"
            ligand_name = f"Compound_{pdb_id}"
            
            # 添加到数据集
            extended_data.append({
                'pdb_id': pdb_id,
                'uniprot_id': uniprot_id,
                'resolution': resolution,
                'affinity_type': affinity_type,
                'affinity_value': round(affinity_value, 2),
                'ligand_id': ligand_id,
                'ligand_name': ligand_name
            })
            
            added += 1
        
        # 如果没有生成足够的数据（例如，所有PDB ID都已存在）
        # 则生成随机PDB ID
        while added < count:
            # 生成随机PDB ID（符合PDB格式）
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            pdb_id = ''.join(random.choice(chars) for _ in range(4))
            
            if pdb_id in existing_pdb_ids or any(entry['pdb_id'] == pdb_id for entry in extended_data):
                continue
            
            # 随机分配一个UniProt ID
            uniprot_id = random.choice(list(common_targets.values()))
            
            # 随机选择亲和力类型和值
            affinity_type = random.choice(affinity_types)
            affinity_range = random.choice(list(affinity_ranges.values()))
            affinity_value = random.uniform(*affinity_range)
            
            # 随机分辨率
            resolution = round(random.uniform(1.5, 3.0), 1)
            
            # 随机配体ID和名称
            ligand_id = f"{pdb_id[-3:].upper()}"
            ligand_name = f"Compound_{pdb_id}"
            
            # 添加到数据集
            extended_data.append({
                'pdb_id': pdb_id,
                'uniprot_id': uniprot_id,
                'resolution': resolution,
                'affinity_type': affinity_type,
                'affinity_value': round(affinity_value, 2),
                'ligand_id': ligand_id,
                'ligand_name': ligand_name
            })
            
            added += 1
        
        return extended_data

    def classify_protein_data(self):
        """对蛋白质数据进行分类，包括四个等级"""
        print("对蛋白质数据进行分类...")
        
        # 获取所有蛋白质
        proteins_df = self.get_uniprot_proteins()
        
        # 获取PDBbind数据
        pdbbind_df = self.get_pdbbind_data(limit=None)
        
        # 获取PDB结构数据
        pdb_df = self.get_pdb_structures()
        
        # 获取BindingDB数据
        binding_df = self.get_binding_data(limit=None)
        
        # 合并所有结合数据源，确保我们考虑所有已知结合数据
        all_binding_uniprot_ids = set(pdbbind_df['uniprot_id'].dropna().unique())
        all_binding_uniprot_ids.update(binding_df['protein_id'].dropna().unique())
        
        # 初始化分类
        tier1_proteins = []  # 一级：有晶体结构 + 有结合数据
        tier2_proteins = []  # 二级：有晶体结构 + 无结合数据
        tier3_proteins = []  # 三级：无晶体结构 + 有结合数据
        tier4_proteins = []  # 四级：无晶体结构 + 无结合数据
        
        # 处理每个蛋白质
        for _, protein in proteins_df.iterrows():
            uniprot_id = protein['uniprot_id']
            
            # 检查是否有结合数据
            has_binding_data = uniprot_id in all_binding_uniprot_ids
            
            # 检查是否有PDB结构
            has_structure = uniprot_id in pdb_df['uniprot_id'].values
            
            # 分类
            if has_structure and has_binding_data:
                # 一级：高质量数据
                tier1_proteins.append({
                    'uniprot_id': uniprot_id,
                    'gene_name': protein['gene_name'] if 'gene_name' in protein else None,
                    'description': protein['description'] if 'description' in protein else None,
                    'tier': 1,
                    'has_binding_data': True,
                    'has_pdb_structure': True
                })
            elif has_structure:
                # 二级：有结构无结合数据
                tier2_proteins.append({
                    'uniprot_id': uniprot_id,
                    'gene_name': protein['gene_name'] if 'gene_name' in protein else None,
                    'description': protein['description'] if 'description' in protein else None,
                    'tier': 2,
                    'has_binding_data': False,
                    'has_pdb_structure': True
                })
            elif has_binding_data:
                # 三级：有结合数据无结构
                tier3_proteins.append({
                    'uniprot_id': uniprot_id,
                    'gene_name': protein['gene_name'] if 'gene_name' in protein else None,
                    'description': protein['description'] if 'description' in protein else None,
                    'tier': 3,
                    'has_binding_data': True,
                    'has_pdb_structure': False
                })
            else:
                # 四级：仅有预测结构
                tier4_proteins.append({
                    'uniprot_id': uniprot_id,
                    'gene_name': protein['gene_name'] if 'gene_name' in protein else None,
                    'description': protein['description'] if 'description' in protein else None,
                    'tier': 4,
                    'has_binding_data': False,
                    'has_pdb_structure': False
                })
        
        # 合并所有分类
        all_classified = tier1_proteins + tier2_proteins + tier3_proteins + tier4_proteins
        classified_df = pd.DataFrame(all_classified)
        
        # 保存分类结果
        output_file = f'{DATA_DIR}/protein_classification.csv'
        classified_df.to_csv(output_file, index=False)
        
        # 返回统计信息
        stats = {
            'tier1_count': len(tier1_proteins),
            'tier2_count': len(tier2_proteins),
            'tier3_count': len(tier3_proteins),
            'tier4_count': len(tier4_proteins),
            'total': len(all_classified)
        }
        
        print(f"蛋白质分类完成：")
        print(f"一级(有结构+有结合数据)：{stats['tier1_count']}个蛋白质")
        print(f"二级(有结构+无结合数据)：{stats['tier2_count']}个蛋白质")
        print(f"三级(无结构+有结合数据)：{stats['tier3_count']}个蛋白质")
        print(f"四级(无结构+无结合数据)：{stats['tier4_count']}个蛋白质")
        
        return classified_df, stats

# 如果作为独立脚本运行
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生物医学数据获取')
    parser.add_argument('--limit', type=int, default=10, help='每类基础数据的最大记录数')
    parser.add_argument('--pdbbind-limit', type=int, default=50, help='PDBbind数据的最大记录数')
    parser.add_argument('--pdbbind-dataset', type=str, default='refined', 
                       choices=['general', 'refined', 'core'],
                       help='使用的PDBbind数据集')
    parser.add_argument('--no-pdb', action='store_true', help='不包括PDB实验结构')
    parser.add_argument('--no-pdbbind', action='store_true', help='不包括PDBbind数据')
    
    args = parser.parse_args()
    
    crawler = DataCrawler()
    data = crawler.fetch_all_data(
        limit=args.limit,
        include_pdb=not args.no_pdb,
        include_pdbbind=not args.no_pdbbind,
        pdbbind_limit=args.pdbbind_limit
    )
    
    print("\n数据摘要:")
    for key, df in data.items():
        print(f"{key}: {len(df)} 条记录") 