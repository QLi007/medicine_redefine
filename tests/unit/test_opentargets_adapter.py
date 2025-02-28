import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.data.adapters.data_adapters import OpenTargetsAdapter

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
def opentargets_adapter(tmp_path):
    """创建测试用的OpenTargets适配器"""
    return OpenTargetsAdapter(base_dir=str(tmp_path))

@pytest.fixture
def sample_target_data():
    """样本靶点关联数据"""
    return {
        'data': {
            'target': {
                'id': 'ENSG00000141510',
                'approvedSymbol': 'TP53',
                'associatedDiseases': {
                    'rows': [
                        {
                            'disease': {
                                'id': 'EFO_0000311',
                                'name': 'Cancer',
                                'therapeuticAreas': [
                                    {'id': 'EFO_0000616', 'name': 'Neoplasm'}
                                ]
                            },
                            'score': 0.85,
                            'datatypeScores': [
                                {'id': 'clinical_trial', 'score': 0.7},
                                {'id': 'genetic_association', 'score': 0.8}
                            ],
                            'evidenceCount': 10
                        },
                        {
                            'disease': {
                                'id': 'EFO_0000270',
                                'name': 'Asthma',
                                'therapeuticAreas': [
                                    {'id': 'EFO_0000540', 'name': 'Respiratory Disease'}
                                ]
                            },
                            'score': 0.45,
                            'datatypeScores': [
                                {'id': 'literature', 'score': 0.5}
                            ],
                            'evidenceCount': 3
                        }
                    ]
                }
            }
        }
    }

@pytest.fixture
def sample_disease_data():
    """样本疾病关联数据"""
    return {
        'data': {
            'disease': {
                'id': 'EFO_0000311',
                'name': 'Cancer',
                'therapeuticAreas': [
                    {'id': 'EFO_0000616', 'name': 'Neoplasm'}
                ],
                'associatedTargets': {
                    'rows': [
                        {
                            'target': {
                                'id': 'ENSG00000141510',
                                'approvedSymbol': 'TP53'
                            },
                            'score': 0.85,
                            'datatypeScores': [
                                {'id': 'clinical_trial', 'score': 0.7},
                                {'id': 'genetic_association', 'score': 0.8}
                            ],
                            'evidenceCount': 10
                        },
                        {
                            'target': {
                                'id': 'ENSG00000171862',
                                'approvedSymbol': 'PTEN'
                            },
                            'score': 0.75,
                            'datatypeScores': [
                                {'id': 'clinical_trial', 'score': 0.6},
                                {'id': 'genetic_association', 'score': 0.7}
                            ],
                            'evidenceCount': 8
                        }
                    ]
                }
            }
        }
    }

def test_initialization(opentargets_adapter):
    """测试初始化"""
    assert opentargets_adapter.api_base_url == "https://api.platform.opentargets.org/api/v4"
    assert opentargets_adapter.data_dir.exists()

@patch('src.data.adapters.data_adapters.OpenTargetsAdapter.make_request')
def test_fetch_data_by_target(mock_make_request, opentargets_adapter, mock_response, 
                            sample_target_data):
    """测试按靶点ID获取数据"""
    # 模拟API响应
    mock_make_request.return_value = mock_response(sample_target_data)
    
    # 获取数据
    result = opentargets_adapter.fetch_data(target_ids=['ENSG00000141510'])
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'target_id' in result.columns
    assert 'disease_id' in result.columns
    assert 'association_score' in result.columns
    
    # 验证数据内容
    assert result.iloc[0]['target_id'] == 'ENSG00000141510'
    assert result.iloc[0]['target_symbol'] == 'TP53'
    assert result.iloc[0]['association_score'] == 0.85

@patch('src.data.adapters.data_adapters.OpenTargetsAdapter.make_request')
def test_fetch_data_by_disease(mock_make_request, opentargets_adapter, mock_response, 
                             sample_disease_data):
    """测试按疾病ID获取数据"""
    # 模拟API响应
    mock_make_request.return_value = mock_response(sample_disease_data)
    
    # 获取数据
    result = opentargets_adapter.fetch_data(disease_ids=['EFO_0000311'])
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'disease_id' in result.columns
    assert 'target_id' in result.columns
    assert 'association_score' in result.columns
    
    # 验证数据内容
    assert result.iloc[0]['disease_id'] == 'EFO_0000311'
    assert result.iloc[0]['disease_name'] == 'Cancer'
    assert len(result['target_id'].unique()) == 2

def test_parse_data(opentargets_adapter):
    """测试数据解析"""
    # 准备测试数据
    raw_data = pd.DataFrame({
        'target_id': ['ENSG00000141510', 'ENSG00000171862', None],
        'disease_id': ['EFO_0000311', None, 'EFO_0000270'],
        'association_score': [0.85, 0.75, None],
        'evidence_count': [10, 8, 3],
        'datatype_scores': [
            {'clinical_trial': 0.7, 'genetic_association': 0.8},
            {'clinical_trial': 0.6},  # 缺少genetic_association
            {'literature': 0.5}
        ]
    })
    
    # 解析数据
    parsed_data = opentargets_adapter.parse_data(raw_data)
    
    # 验证结果
    assert len(parsed_data) == 1  # 应该只保留完全有效的数据
    assert 'evidence_level' in parsed_data.columns
    assert parsed_data.iloc[0]['evidence_level'] == 'high'  # 高分数、高证据数量、有临床和遗传证据

def test_transform_data(opentargets_adapter):
    """测试数据转换"""
    # 准备测试数据
    parsed_data = pd.DataFrame({
        'target_id': ['ENSG00000141510'],
        'target_symbol': ['TP53'],
        'disease_id': ['EFO_0000311'],
        'disease_name': ['Cancer'],
        'therapeutic_areas': [['Neoplasm']],
        'association_score': [0.85],
        'evidence_level': ['high'],
        'evidence_count': [10]
    })
    
    # 转换数据
    transformed_data = opentargets_adapter.transform_data(parsed_data)
    
    # 验证结果
    assert len(transformed_data) == 1
    assert all(col in transformed_data.columns for col in [
        'target_id', 'target_symbol', 'disease_id', 'disease_name',
        'therapeutic_areas', 'association_score', 'evidence_level',
        'evidence_count', 'data_source'
    ])
    assert transformed_data['data_source'].unique() == ['OpenTargets']

def test_cache_mechanism(opentargets_adapter, tmp_path):
    """测试缓存机制"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'target_id': ['ENSG00000141510'],
        'target_symbol': ['TP53'],
        'disease_id': ['EFO_0000311'],
        'disease_name': ['Cancer']
    })
    
    # 保存到缓存
    cache_file = opentargets_adapter.data_dir / 'target_disease_associations.csv'
    test_data.to_csv(cache_file, index=False)
    
    # 验证从缓存加载
    loaded_data = opentargets_adapter.fetch_data()
    assert len(loaded_data) == len(test_data)
    assert 'target_id' in loaded_data.columns

def test_error_handling(opentargets_adapter):
    """测试错误处理"""
    # 准备包含无效数据的DataFrame
    invalid_data = pd.DataFrame({
        'target_id': [None, None],
        'disease_id': [None, 'EFO_0000311'],
        'association_score': [None, 0.5]
    })
    
    # 验证解析处理
    parsed_data = opentargets_adapter.parse_data(invalid_data)
    assert len(parsed_data) == 0  # 应该过滤掉所有无效数据

def test_evidence_level_calculation(opentargets_adapter):
    """测试证据等级计算"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'target_id': ['T1', 'T2', 'T3'],
        'disease_id': ['D1', 'D2', 'D3'],
        'association_score': [0.85, 0.45, 0.35],
        'evidence_count': [10, 3, 1],
        'datatype_scores': [
            {'clinical_trial': 0.7, 'genetic_association': 0.8},
            {'literature': 0.5},
            {'literature': 0.3}
        ]
    })
    
    # 解析数据
    parsed_data = opentargets_adapter.parse_data(test_data)
    
    # 验证证据等级
    evidence_levels = parsed_data['evidence_level'].tolist()
    assert evidence_levels == ['high', 'medium', 'low'] 