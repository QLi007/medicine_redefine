import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.data.adapters.data_adapters import DisGeNETAdapter

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
def disgenet_adapter(tmp_path):
    """创建测试用的DisGeNET适配器"""
    return DisGeNETAdapter(base_dir=str(tmp_path))

@pytest.fixture
def sample_disease_associations():
    """样本疾病关联数据"""
    return [
        {
            'disease_id': 'C0023467',
            'disease_name': 'Alzheimer Disease',
            'disease_type': 'disease',
            'disease_class': 'Nervous System Disease',
            'gene_id': '348',
            'gene_symbol': 'APOE',
            'score': 0.85,
            'evidence_count': 10,
            'source': 'CURATED',
            'pmid_count': 8,
            'first_reported': '1990',
            'last_reported': '2023'
        },
        {
            'disease_id': 'C0023467',
            'disease_name': 'Alzheimer Disease',
            'disease_type': 'disease',
            'disease_class': 'Nervous System Disease',
            'gene_id': '351',
            'gene_symbol': 'APP',
            'score': 0.75,
            'evidence_count': 7,
            'source': 'CURATED',
            'pmid_count': 5,
            'first_reported': '1991',
            'last_reported': '2023'
        }
    ]

@pytest.fixture
def sample_gene_associations():
    """样本基因关联数据"""
    return [
        {
            'disease_id': 'C0023467',
            'disease_name': 'Alzheimer Disease',
            'disease_type': 'disease',
            'disease_class': 'Nervous System Disease',
            'gene_id': '348',
            'gene_symbol': 'APOE',
            'score': 0.85,
            'evidence_count': 10,
            'source': 'CURATED',
            'pmid_count': 8,
            'first_reported': '1990',
            'last_reported': '2023'
        },
        {
            'disease_id': 'C0010417',
            'disease_name': 'Coronary Disease',
            'disease_type': 'disease',
            'disease_class': 'Cardiovascular Disease',
            'gene_id': '348',
            'gene_symbol': 'APOE',
            'score': 0.65,
            'evidence_count': 5,
            'source': 'CURATED',
            'pmid_count': 3,
            'first_reported': '1992',
            'last_reported': '2023'
        }
    ]

def test_initialization(disgenet_adapter):
    """测试初始化"""
    assert disgenet_adapter.api_base_url == "https://www.disgenet.org/api"
    assert disgenet_adapter.data_dir.exists()

@patch('src.data.adapters.data_adapters.DisGeNETAdapter.make_request')
def test_fetch_data_by_disease(mock_make_request, disgenet_adapter, mock_response, 
                             sample_disease_associations):
    """测试按疾病ID获取数据"""
    # 模拟API响应
    mock_make_request.return_value = mock_response(sample_disease_associations)
    
    # 获取数据
    result = disgenet_adapter.fetch_data(disease_ids=['C0023467'])
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'disease_id' in result.columns
    assert 'gene_id' in result.columns
    assert 'score' in result.columns
    
    # 验证数据内容
    assert result.iloc[0]['disease_id'] == 'C0023467'
    assert result.iloc[0]['gene_symbol'] == 'APOE'
    assert result.iloc[0]['score'] == 0.85

@patch('src.data.adapters.data_adapters.DisGeNETAdapter.make_request')
def test_fetch_data_by_gene(mock_make_request, disgenet_adapter, mock_response, 
                          sample_gene_associations):
    """测试按基因ID获取数据"""
    # 模拟API响应
    mock_make_request.return_value = mock_response(sample_gene_associations)
    
    # 获取数据
    result = disgenet_adapter.fetch_data(gene_ids=['348'])
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'disease_id' in result.columns
    assert 'gene_id' in result.columns
    assert 'score' in result.columns
    
    # 验证数据内容
    assert result.iloc[0]['gene_id'] == '348'
    assert result.iloc[0]['gene_symbol'] == 'APOE'
    assert len(result['disease_id'].unique()) == 2

def test_parse_data(disgenet_adapter):
    """测试数据解析"""
    # 准备测试数据
    raw_data = pd.DataFrame({
        'disease_id': ['C0023467', 'C0010417', None],
        'gene_id': ['348', None, '351'],
        'score': [0.85, 0.65, None],
        'pmid_count': [8, 3, 1]
    })
    
    # 解析数据
    parsed_data = disgenet_adapter.parse_data(raw_data)
    
    # 验证结果
    assert len(parsed_data) == 1  # 应该只保留有效数据
    assert 'evidence_level' in parsed_data.columns
    assert parsed_data.iloc[0]['evidence_level'] == 'high'  # score >= 0.7 and pmid_count >= 5

def test_transform_data(disgenet_adapter):
    """测试数据转换"""
    # 准备测试数据
    parsed_data = pd.DataFrame({
        'disease_id': ['C0023467'],
        'disease_name': ['Alzheimer Disease'],
        'disease_type': ['disease'],
        'gene_id': ['348'],
        'gene_symbol': ['APOE'],
        'score': [0.85],
        'evidence_level': ['high'],
        'evidence_count': [10],
        'pmid_count': [8]
    })
    
    # 转换数据
    transformed_data = disgenet_adapter.transform_data(parsed_data)
    
    # 验证结果
    assert len(transformed_data) == 1
    assert all(col in transformed_data.columns for col in [
        'disease_id', 'disease_name', 'disease_type',
        'gene_id', 'gene_symbol', 'association_score',
        'evidence_level', 'evidence_count', 'publication_count',
        'data_source'
    ])
    assert transformed_data['data_source'].unique() == ['DisGeNET']

def test_cache_mechanism(disgenet_adapter, tmp_path):
    """测试缓存机制"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'disease_id': ['C0023467'],
        'disease_name': ['Alzheimer Disease'],
        'gene_id': ['348'],
        'gene_symbol': ['APOE']
    })
    
    # 保存到缓存
    cache_file = disgenet_adapter.data_dir / 'disease_gene_associations.csv'
    test_data.to_csv(cache_file, index=False)
    
    # 验证从缓存加载
    loaded_data = disgenet_adapter.fetch_data()
    assert len(loaded_data) == len(test_data)
    assert 'disease_id' in loaded_data.columns

def test_error_handling(disgenet_adapter):
    """测试错误处理"""
    # 准备包含无效数据的DataFrame
    invalid_data = pd.DataFrame({
        'disease_id': [None, None],
        'gene_id': [None, '348'],
        'score': [None, 0.5]
    })
    
    # 验证解析处理
    parsed_data = disgenet_adapter.parse_data(invalid_data)
    assert len(parsed_data) == 0  # 应该过滤掉所有无效数据

def test_evidence_level_calculation(disgenet_adapter):
    """测试证据等级计算"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'disease_id': ['D1', 'D2', 'D3'],
        'gene_id': ['G1', 'G2', 'G3'],
        'score': [0.8, 0.5, 0.3],
        'pmid_count': [6, 3, 1]
    })
    
    # 解析数据
    parsed_data = disgenet_adapter.parse_data(test_data)
    
    # 验证证据等级
    evidence_levels = parsed_data['evidence_level'].tolist()
    assert evidence_levels == ['high', 'medium', 'low'] 