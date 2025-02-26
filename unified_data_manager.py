from data_adapters import *

class UnifiedDataManager:
    """统一数据管理系统"""
    
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        
        # 初始化所有适配器
        self.adapters = {
            'pdb': PDBAdapter(base_dir),
            'pdbbind': PDBbindAdapter(base_dir),
            'bindingdb': BindingDBAdapter(base_dir),
            'uniprot': UniProtAdapter(base_dir),
            'alphafold': AlphaFoldAdapter(base_dir),
            'drugbank': DrugBankAdapter(base_dir),
            'disgenet': DisGeNETAdapter(base_dir),
            'chembl': ChEMBLAdapter(base_dir),
            'zinc': ZINCAdapter(base_dir),
            'opentarget': OpenTargetsAdapter(base_dir),
            'cmap': CMapAdapter(base_dir)
        }
    
    def get_data(self, source, **kwargs):
        """获取指定数据源的数据"""
        if source not in self.adapters:
            raise ValueError(f"未知的数据源: {source}")
        
        return self.adapters[source].fetch_data(**kwargs)
    
    def get_all_protein_data(self, limit=50):
        """获取所有蛋白质相关数据"""
        # 获取基础蛋白质数据
        uniprot_data = self.get_data('uniprot', limit=limit)
        
        # 获取蛋白质结构数据
        pdb_data = self.get_data('pdb', limit=limit)
        alphafold_data = self.get_data('alphafold', uniprot_ids=uniprot_data['uniprot_id'].tolist())
        
        # 获取蛋白质结合数据
        pdbbind_data = self.get_data('pdbbind', limit=limit)
        bindingdb_data = self.get_data('bindingdb', limit=limit)
        
        return {
            'uniprot': uniprot_data,
            'pdb': pdb_data,
            'alphafold': alphafold_data,
            'pdbbind': pdbbind_data,
            'bindingdb': bindingdb_data
        }
    
    def get_all_drug_data(self, limit=50):
        """获取所有药物相关数据"""
        # 获取药物数据
        drugbank_data = self.get_data('drugbank', limit=limit)
        chembl_data = self.get_data('chembl', limit=limit)
        zinc_data = self.get_data('zinc', limit=limit)
        
        return {
            'drugbank': drugbank_data,
            'chembl': chembl_data,
            'zinc': zinc_data
        }
    
    def get_all_disease_data(self, limit=50):
        """获取所有疾病相关数据"""
        # 获取疾病数据
        disgenet_data = self.get_data('disgenet', limit=limit)
        opentarget_data = self.get_data('opentarget', limit=limit)
        
        return {
            'disgenet': disgenet_data,
            'opentarget': opentarget_data
        }
    
    def classify_proteins(self):
        """对蛋白质数据进行四级分类"""
        # 获取所有蛋白质
        proteins_df = self.get_data('uniprot')
        
        # 获取结构数据
        pdb_df = self.get_data('pdb')
        
        # 获取结合数据
        pdbbind_df = self.get_data('pdbbind')
        binding_df = self.get_data('bindingdb')
        
        # 合并所有结合数据源
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
            tier_data = {
                'uniprot_id': uniprot_id,
                'gene_name': protein['gene_name'] if 'gene_name' in protein else None,
                'description': protein['description'] if 'description' in protein else None,
                'has_binding_data': has_binding_data,
                'has_pdb_structure': has_structure
            }
            
            if has_structure and has_binding_data:
                tier_data['tier'] = 1
                tier1_proteins.append(tier_data)
            elif has_structure:
                tier_data['tier'] = 2
                tier2_proteins.append(tier_data)
            elif has_binding_data:
                tier_data['tier'] = 3
                tier3_proteins.append(tier_data)
            else:
                tier_data['tier'] = 4
                tier4_proteins.append(tier_data)
        
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