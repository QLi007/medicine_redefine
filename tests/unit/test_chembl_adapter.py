import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.data.adapters.data_adapters import ChEMBLAdapter

@pytest.fixture
def mock_response():
    """模拟API响应"""
    class MockResponse:
        def __init__(self, data):
            self._data = data
            
        def json(self):
            return self._data
            
    return MockResponse

@pytest.fixture
def chembl_adapter(tmp_path):
    """创建测试用的ChEMBL适配器"""
    return ChEMBLAdapter(base_dir=str(tmp_path))

@pytest.fixture
def sample_targets_data():
    """样本靶点数据"""
    return {
        'targets': [
            {
                'target_chembl_id': 'CHEMBL1234',
                'target_components': [
                    {'accession': 'P12345'}
                ]
            },
            {
                'target_chembl_id': 'CHEMBL5678',
                'target_components': [
                    {'accession': 'P67890'}
                ]
            }
        ]
    }

@pytest.fixture
def sample_activities_data():
    """样本活性数据"""
    return {
        'activities': [
            {
                'molecule_chembl_id': 'CHEMBL100',
                'standard_type': 'IC50',
                'standard_value': 50.0,
                'standard_units': 'nM',
                'pchembl_value': 7.3,
                'activity_comment': 'Active',
                'assay_type': 'B',
                'assay_description': 'Binding assay'
            },
            {
                'molecule_chembl_id': 'CHEMBL200',
                'standard_type': 'Ki',
                'standard_value': 2.5,
                'standard_units': 'uM',
                'pchembl_value': 6.6,
                'activity_comment': 'Active',
                'assay_type': 'B',
                'assay_description': 'Inhibition assay'
            }
        ]
    }

def test_initialization(chembl_adapter):
    """测试初始化"""
    assert chembl_adapter.api_base_url == "https://www.ebi.ac.uk/chembl/api/data"
    assert chembl_adapter.data_dir.exists()

@patch('src.data.adapters.data_adapters.ChEMBLAdapter.make_request')
def test_fetch_data(mock_make_request, chembl_adapter, mock_response, 
                   sample_targets_data, sample_activities_data):
    """测试数据获取"""
    # 模拟API响应
    mock_make_request.side_effect = [
        mock_response({'targets': [{'target_chembl_id': 'CHEMBL1234'}]}),
        mock_response(sample_activities_data)
    ]
    
    # 获取数据
    result = chembl_adapter.fetch_data(uniprot_ids=['P12345'])
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_activities_data['activities'])
    assert 'molecule_chembl_id' in result.columns
    assert 'standard_type' in result.columns
    assert 'standard_value' in result.columns

def test_parse_data(chembl_adapter):
    """测试数据解析"""
    # 准备测试数据
    raw_data = pd.DataFrame({
        'standard_type': ['IC50', 'Ki', 'EC50', 'Other'],
        'standard_value': [100, 50, 75, 200],
        'standard_units': ['nM', 'uM', 'pM', 'mM']
    })
    
    # 解析数据
    parsed_data = chembl_adapter.parse_data(raw_data)
    
    # 验证结果
    assert len(parsed_data) == 3  # 应该过滤掉'Other'类型
    assert 'standard_value_nm' in parsed_data.columns
    # 验证单位转换
    assert parsed_data.iloc[1]['standard_value_nm'] == 50000  # uM -> nM

def test_transform_data(chembl_adapter):
    """测试数据转换"""
    # 准备测试数据
    parsed_data = pd.DataFrame({
        'uniprot_id': ['P12345', 'P12345'],
        'molecule_chembl_id': ['CHEMBL100', 'CHEMBL200'],
        'standard_type': ['IC50', 'Ki'],
        'standard_value_nm': [100, 50000],
        'pchembl_value': [7.3, 6.6],
        'assay_type': ['B', 'B'],
        'assay_description': ['Binding', 'Inhibition']
    })
    
    # 转换数据
    transformed_data = chembl_adapter.transform_data(parsed_data)
    
    # 验证结果
    assert len(transformed_data) == 2
    assert all(col in transformed_data.columns for col in [
        'uniprot_id', 'compound_id', 'activity_type', 'activity_value',
        'activity_unit', 'activity_score', 'data_source'
    ])
    assert transformed_data['data_source'].unique() == ['ChEMBL']
    assert transformed_data['activity_unit'].unique() == ['nM']

def test_cache_mechanism(chembl_adapter, tmp_path):
    """测试缓存机制"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL100'],
        'standard_type': ['IC50'],
        'standard_value': [100]
    })
    
    # 保存到缓存
    cache_file = chembl_adapter.data_dir / 'chembl_activities.csv'
    test_data.to_csv(cache_file, index=False)
    
    # 验证从缓存加载
    loaded_data = chembl_adapter.fetch_data()
    assert len(loaded_data) == len(test_data)
    assert 'molecule_chembl_id' in loaded_data.columns

def test_error_handling(chembl_adapter):
    """测试错误处理"""
    # 准备包含无效数据的DataFrame
    invalid_data = pd.DataFrame({
        'standard_type': ['Invalid'],
        'standard_value': [None],
        'standard_units': ['Invalid']
    })
    
    # 验证解析处理
    parsed_data = chembl_adapter.parse_data(invalid_data)
    assert len(parsed_data) == 0  # 应该过滤掉所有无效数据 