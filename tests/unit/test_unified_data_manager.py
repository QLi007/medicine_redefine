import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.data.manager.unified_data_manager import UnifiedDataManager

@pytest.fixture
def mock_data():
    """创建模拟数据"""
    uniprot_data = pd.DataFrame({
        'uniprot_id': ['P12345', 'P67890'],
        'gene_name': ['GENE1', 'GENE2'],
        'protein_name': ['Protein 1', 'Protein 2'],
        'length': [100, 200]
    })
    
    pdb_data = pd.DataFrame({
        'pdb_id': ['1abc', '2def'],
        'uniprot_id': ['P12345', 'P67890'],
        'resolution': [1.8, 2.0],
        'structure_type': ['experimental', 'experimental']
    })
    
    pdbbind_data = pd.DataFrame({
        'pdb_id': ['1abc'],
        'uniprot_id': ['P12345'],
        'affinity_value': [7.5],
        'affinity_type': ['Ki']
    })
    
    bindingdb_data = pd.DataFrame({
        'protein_id': ['P12345'],
        'ligand_id': ['DB00001'],
        'affinity_value': [6.8],
        'affinity_type': ['Kd']
    })
    
    alphafold_data = pd.DataFrame({
        'uniprot_id': ['P12345', 'P67890'],
        'structure_file': ['AF-P12345.pdb', 'AF-P67890.pdb'],
        'confidence': [90, 85]
    })
    
    return {
        'uniprot': uniprot_data,
        'pdb': pdb_data,
        'pdbbind': pdbbind_data,
        'bindingdb': bindingdb_data,
        'alphafold': alphafold_data
    }

@pytest.fixture
def mock_adapters(mock_data):
    """模拟所有数据适配器"""
    adapters = {}
    for source, data in mock_data.items():
        mock_adapter = MagicMock()
        mock_adapter.fetch_data.return_value = data
        adapters[source] = mock_adapter
    return adapters

@pytest.fixture
def data_manager(tmp_path, mock_adapters):
    """创建UnifiedDataManager实例"""
    with patch('src.data.manager.unified_data_manager.PDBAdapter') as mock_pdb, \
         patch('src.data.manager.unified_data_manager.PDBbindAdapter') as mock_pdbbind, \
         patch('src.data.manager.unified_data_manager.BindingDBAdapter') as mock_bindingdb, \
         patch('src.data.manager.unified_data_manager.UniProtAdapter') as mock_uniprot, \
         patch('src.data.manager.unified_data_manager.AlphaFoldAdapter') as mock_alphafold:
        
        # 设置mock适配器
        mock_pdb.return_value = mock_adapters['pdb']
        mock_pdbbind.return_value = mock_adapters['pdbbind']
        mock_bindingdb.return_value = mock_adapters['bindingdb']
        mock_uniprot.return_value = mock_adapters['uniprot']
        mock_alphafold.return_value = mock_adapters['alphafold']
        
        manager = UnifiedDataManager(base_dir=str(tmp_path))
        return manager

def test_initialization(data_manager):
    """测试初始化"""
    assert data_manager is not None
    assert len(data_manager.adapters) == 5
    assert all(adapter in data_manager.adapters for adapter in ['pdb', 'pdbbind', 'bindingdb', 'uniprot', 'alphafold'])

def test_get_data(data_manager, mock_data):
    """测试获取特定数据源的数据"""
    # 测试获取PDB数据
    pdb_data = data_manager.get_data('pdb')
    assert len(pdb_data) == len(mock_data['pdb'])
    assert all(col in pdb_data.columns for col in ['pdb_id', 'uniprot_id', 'resolution'])

def test_get_all_protein_data(data_manager, mock_data):
    """测试获取所有蛋白质相关数据"""
    all_data = data_manager.get_all_protein_data()
    
    # 验证返回的数据结构
    assert isinstance(all_data, dict)
    assert all(source in all_data for source in ['uniprot', 'pdb', 'alphafold', 'pdbbind', 'bindingdb'])
    
    # 验证数据内容
    assert len(all_data['uniprot']) == len(mock_data['uniprot'])
    assert len(all_data['pdb']) == len(mock_data['pdb'])
    assert len(all_data['alphafold']) == len(mock_data['alphafold'])

def test_classify_proteins(data_manager, mock_data):
    """测试蛋白质分类功能"""
    classified_df, stats = data_manager.classify_proteins()
    
    # 验证分类结果
    assert isinstance(classified_df, pd.DataFrame)
    assert isinstance(stats, dict)
    assert all(key in stats for key in ['tier1_count', 'tier2_count', 'tier3_count', 'tier4_count', 'total'])
    assert stats['total'] == len(mock_data['uniprot'])

def test_invalid_data_source(data_manager):
    """测试无效数据源处理"""
    with pytest.raises(ValueError, match="未知的数据源"):
        data_manager.get_data('invalid_source')

def test_error_handling(data_manager, mock_adapters):
    """测试错误处理"""
    # 模拟数据获取失败
    mock_adapters['pdb'].fetch_data.side_effect = Exception("API Error")
    
    # 验证是否正确处理错误
    with pytest.raises(Exception, match="API Error"):
        data_manager.get_data('pdb')

def test_data_integration(data_manager, mock_data):
    """测试数据整合功能"""
    # 获取整合后的数据
    protein_data = data_manager.get_all_protein_data()
    
    # 验证数据整合
    assert len(protein_data['uniprot']) == len(mock_data['uniprot'])
    assert len(protein_data['pdb']) == len(mock_data['pdb'])
    assert len(protein_data['alphafold']) == len(mock_data['alphafold'])
    
    # 验证关联性
    pdb_proteins = set(protein_data['pdb']['uniprot_id'])
    uniprot_proteins = set(protein_data['uniprot']['uniprot_id'])
    assert pdb_proteins.issubset(uniprot_proteins) 