import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.data.adapters.data_adapters import DrugBankAdapter

@pytest.fixture
def mock_response():
    """模拟API响应"""
    class MockResponse:
        def __init__(self, data, status_code=200):
            self._data = data
            self.status_code = status_code
            
        def json(self):
            return self._data
            
    return MockResponse

@pytest.fixture
def drugbank_adapter(tmp_path):
    """创建测试用的DrugBank适配器"""
    return DrugBankAdapter(base_dir=str(tmp_path))

@pytest.fixture
def sample_drugs_list():
    """样本药物列表数据"""
    return {
        'drugs': [
            {'drugbank_id': 'DB00001'},
            {'drugbank_id': 'DB00002'}
        ]
    }

@pytest.fixture
def sample_drug_data():
    """样本药物详细数据"""
    return {
        'drugbank_id': 'DB00001',
        'name': 'Test Drug',
        'description': 'Test description',
        'cas_number': '123-45-6',
        'type': 'small molecule',
        'groups': ['approved', 'investigational'],
        'molecular_formula': 'C10H15N5O10P2',
        'molecular_weight': 507.18,
        'smiles': 'CC1=NC=NC2=C1NC=N2',
        'inchi': 'InChI=1S/C8H10N4O2/1-7-8(13)10-4-9-6(7)12(2)5-11(3)14',
        'inchikey': 'RYYVLZVUVIJVGH-UHFFFAOYSA-N',
        'state': 'solid',
        'indication': 'Treatment of...',
        'pharmacodynamics': 'Acts by...',
        'mechanism_of_action': 'Inhibits...',
        'toxicity': 'Low toxicity',
        'metabolism': 'Liver metabolism',
        'absorption': 'Oral absorption',
        'half_life': '3-4 hours',
        'protein_binding': '80%',
        'route_of_elimination': 'Renal',
        'volume_of_distribution': '0.5 L/kg',
        'clearance': '30 mL/min',
        'targets': [
            {
                'uniprot_id': 'P12345',
                'name': 'Target Protein',
                'organism': 'Humans',
                'actions': ['inhibitor', 'antagonist'],
                'known_action': 'yes'
            }
        ]
    }

def test_initialization(drugbank_adapter):
    """测试初始化"""
    assert drugbank_adapter.api_base_url == "https://go.drugbank.com/api/v1"
    assert drugbank_adapter.data_dir.exists()

@patch('src.data.adapters.data_adapters.DrugBankAdapter.make_request')
def test_fetch_data(mock_make_request, drugbank_adapter, mock_response, 
                   sample_drugs_list, sample_drug_data):
    """测试数据获取"""
    # 模拟API响应
    mock_make_request.side_effect = [
        mock_response(sample_drugs_list),
        mock_response(sample_drug_data)
    ]
    
    # 获取数据
    result = drugbank_adapter.fetch_data(limit=1)
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert 'drugbank_id' in result.columns
    assert 'name' in result.columns
    assert 'targets' in result.columns
    
    # 验证数据内容
    assert result.iloc[0]['drugbank_id'] == 'DB00001'
    assert result.iloc[0]['name'] == 'Test Drug'
    assert len(result.iloc[0]['targets']) == 1

def test_parse_data(drugbank_adapter):
    """测试数据解析"""
    # 准备测试数据
    raw_data = pd.DataFrame({
        'drugbank_id': ['DB00001', 'DB00002', None],
        'name': ['Drug A', None, 'Drug C'],
        'targets': [
            [{'uniprot_id': 'P12345', 'name': 'Target A', 'actions': 'inhibitor', 'known_action': 'yes'}],
            [{'uniprot_id': 'P67890', 'name': 'Target B', 'actions': 'activator', 'known_action': 'yes'}],
            []
        ]
    })
    
    # 解析数据
    parsed_data = drugbank_adapter.parse_data(raw_data)
    
    # 验证结果
    assert len(parsed_data) == 1  # 应该只保留有效数据
    assert 'uniprot_id' in parsed_data.columns
    assert 'name_drug' in parsed_data.columns
    assert 'name_target' in parsed_data.columns

def test_transform_data(drugbank_adapter):
    """测试数据转换"""
    # 准备测试数据
    parsed_data = pd.DataFrame({
        'drugbank_id': ['DB00001'],
        'name_drug': ['Test Drug'],
        'smiles': ['CC1=NC=NC2=C1NC=N2'],
        'inchi': ['InChI=1S/...'],
        'molecular_weight': [507.18],
        'description': ['Test description'],
        'mechanism_of_action': ['Inhibits...'],
        'indication': ['Treatment of...'],
        'uniprot_id': ['P12345'],
        'name_target': ['Target Protein'],
        'actions': ['inhibitor']
    })
    
    # 转换数据
    transformed_data = drugbank_adapter.transform_data(parsed_data)
    
    # 验证结果
    assert len(transformed_data) == 1
    assert all(col in transformed_data.columns for col in [
        'drug_id', 'drug_name', 'smiles', 'inchi', 'molecular_weight',
        'description', 'mechanism_of_action', 'indication',
        'target_uniprot_id', 'target_name', 'target_action', 'data_source'
    ])
    assert transformed_data['data_source'].unique() == ['DrugBank']

def test_cache_mechanism(drugbank_adapter, tmp_path):
    """测试缓存机制"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'drugbank_id': ['DB00001'],
        'name': ['Test Drug'],
        'description': ['Test description']
    })
    
    # 保存到缓存
    cache_file = drugbank_adapter.data_dir / 'drugbank_drugs.csv'
    test_data.to_csv(cache_file, index=False)
    
    # 验证从缓存加载
    loaded_data = drugbank_adapter.fetch_data()
    assert len(loaded_data) == len(test_data)
    assert 'drugbank_id' in loaded_data.columns

def test_error_handling(drugbank_adapter):
    """测试错误处理"""
    # 准备包含无效数据的DataFrame
    invalid_data = pd.DataFrame({
        'drugbank_id': [None, None],
        'name': [None, 'Invalid Drug'],
        'targets': [[], []]
    })
    
    # 验证解析处理
    parsed_data = drugbank_adapter.parse_data(invalid_data)
    assert len(parsed_data) == 0  # 应该过滤掉所有无效数据 