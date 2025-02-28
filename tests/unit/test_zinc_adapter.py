import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.data.adapters.data_adapters import ZINCAdapter

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
def zinc_adapter(tmp_path):
    """创建测试用的ZINC适配器"""
    return ZINCAdapter(base_dir=str(tmp_path))

@pytest.fixture
def sample_compounds_list():
    """样本化合物列表数据"""
    return {
        'compounds': [
            {'zinc_id': 'ZINC000123456789'},
            {'zinc_id': 'ZINC000987654321'}
        ]
    }

@pytest.fixture
def sample_compound_data():
    """样本化合物详细数据"""
    return {
        'zinc_id': 'ZINC000123456789',
        'name': 'Test Compound',
        'smiles': 'CC1=CC=CC=C1',
        'inchi': 'InChI=1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3',
        'inchikey': 'YXFVVABEGXRONW-UHFFFAOYSA-N',
        'molecular_formula': 'C7H8',
        'molecular_weight': 92.14,
        'logp': 2.1,
        'rotatable_bonds': 0,
        'polar_surface_area': 0.0,
        'hbd': 0,
        'hba': 0,
        'charge': 0,
        'substance_type': 'small molecule',
        'purchasability': 'in-stock',
        'similar_compounds': [
            'ZINC000123456790',
            'ZINC000123456791'
        ],
        '3d_url': 'http://zinc.docking.org/3d/ZINC000123456789.mol2'
    }

def test_initialization(zinc_adapter):
    """测试初始化"""
    assert zinc_adapter.api_base_url == "http://zinc.docking.org/api/v2"
    assert zinc_adapter.data_dir.exists()
    assert (zinc_adapter.data_dir / 'structures').exists()

@patch('src.data.adapters.data_adapters.ZINCAdapter.make_request')
def test_fetch_data(mock_make_request, zinc_adapter, mock_response, 
                   sample_compounds_list, sample_compound_data):
    """测试数据获取"""
    # 模拟API响应
    mock_make_request.side_effect = [
        mock_response(sample_compounds_list),
        mock_response(sample_compound_data),
        mock_response(sample_compound_data)  # 为第二个化合物返回相同的数据
    ]
    
    # 模拟3D结构下载
    with patch.object(zinc_adapter, '_download_3d_structure', return_value='path/to/structure.mol2'):
        # 获取数据
        result = zinc_adapter.fetch_data(limit=1)
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert 'zinc_id' in result.columns
    assert 'smiles' in result.columns
    assert 'molecular_weight' in result.columns
    
    # 验证数据内容
    assert result.iloc[0]['zinc_id'] == 'ZINC000123456789'
    assert result.iloc[0]['name'] == 'Test Compound'
    assert result.iloc[0]['molecular_weight'] == 92.14

def test_parse_data(zinc_adapter):
    """测试数据解析"""
    # 准备测试数据
    raw_data = pd.DataFrame({
        'zinc_id': ['ZINC000123456789', 'ZINC000987654321', None],
        'smiles': ['CC1=CC=CC=C1', None, 'CC1=CC=CC=C1'],
        'molecular_weight': [92.14, 100.0, 92.14],
        'logp': [2.1, 1.5, 2.1],
        'hbd': [0, 2, 0],
        'hba': [0, 3, 0]
    })
    
    # 解析数据
    parsed_data = zinc_adapter.parse_data(raw_data)
    
    # 验证结果
    assert len(parsed_data) == 1  # 应该只保留有效数据
    assert 'drug_likeness' in parsed_data.columns
    assert parsed_data.iloc[0]['drug_likeness'] == 'high'  # 符合Lipinski规则

def test_transform_data(zinc_adapter):
    """测试数据转换"""
    # 准备测试数据
    parsed_data = pd.DataFrame({
        'zinc_id': ['ZINC000123456789'],
        'name': ['Test Compound'],
        'smiles': ['CC1=CC=CC=C1'],
        'inchi': ['InChI=1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3'],
        'molecular_formula': ['C7H8'],
        'molecular_weight': [92.14],
        'logp': [2.1],
        'drug_likeness': ['high'],
        'structure_file': ['path/to/structure.mol2'],
        'purchasability': ['in-stock']
    })
    
    # 转换数据
    transformed_data = zinc_adapter.transform_data(parsed_data)
    
    # 验证结果
    assert len(transformed_data) == 1
    assert all(col in transformed_data.columns for col in [
        'compound_id', 'name', 'smiles', 'inchi',
        'molecular_formula', 'molecular_weight', 'logp',
        'drug_likeness', 'structure_file', 'purchasability',
        'data_source'
    ])
    assert transformed_data['data_source'].unique() == ['ZINC']

def test_cache_mechanism(zinc_adapter, tmp_path):
    """测试缓存机制"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'zinc_id': ['ZINC000123456789'],
        'name': ['Test Compound'],
        'smiles': ['CC1=CC=CC=C1']
    })
    
    # 保存到缓存
    cache_file = zinc_adapter.data_dir / 'zinc_drug-like_compounds.csv'
    test_data.to_csv(cache_file, index=False)
    
    # 验证从缓存加载
    loaded_data = zinc_adapter.fetch_data()
    assert len(loaded_data) == len(test_data)
    assert 'zinc_id' in loaded_data.columns

@patch('src.data.adapters.data_adapters.ZINCAdapter.make_request')
def test_3d_structure_download(mock_make_request, zinc_adapter, mock_response):
    """测试3D结构下载"""
    # 模拟结构文件内容
    structure_content = b"MOLECULE\nTest Compound\n..."
    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 200
    mock_response_obj.content = structure_content
    mock_make_request.return_value = mock_response_obj
    
    # 下载结构
    zinc_id = 'ZINC000123456789'
    url = 'http://zinc.docking.org/3d/ZINC000123456789.mol2'
    result = zinc_adapter._download_3d_structure(zinc_id, url)
    
    # 验证结果
    assert result is not None
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0

def test_error_handling(zinc_adapter):
    """测试错误处理"""
    # 准备包含无效数据的DataFrame
    invalid_data = pd.DataFrame({
        'zinc_id': [None, None],
        'smiles': [None, 'CC1=CC=CC=C1']
    })
    
    # 验证解析处理
    parsed_data = zinc_adapter.parse_data(invalid_data)
    assert len(parsed_data) == 0  # 应该过滤掉所有无效数据

def test_drug_likeness_calculation(zinc_adapter):
    """测试药物相似性计算"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'zinc_id': ['Z1', 'Z2', 'Z3'],
        'smiles': ['C1', 'C2', 'C3'],
        'molecular_weight': [400, 600, 450],
        'logp': [2.0, 5.5, 4.0],
        'hbd': [2, 6, 3],
        'hba': [5, 12, 8]
    })
    
    # 解析数据
    parsed_data = zinc_adapter.parse_data(test_data)
    
    # 验证药物相似性评分
    drug_likeness = parsed_data['drug_likeness'].tolist()
    assert drug_likeness == ['high', 'low', 'high']  # 修正预期结果 