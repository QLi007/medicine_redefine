import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.data.adapters.data_adapters import CMapAdapter

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
def cmap_adapter(tmp_path):
    """创建测试用的CMap适配器"""
    return CMapAdapter(base_dir=str(tmp_path))

@pytest.fixture
def sample_profiles_data():
    """样本表达谱列表数据"""
    return {
        'profiles': [
            {'profile_id': 'CPC001_A375_24H'},
            {'profile_id': 'CPC002_MCF7_6H'}
        ]
    }

@pytest.fixture
def sample_profile_data():
    """样本表达谱详细数据"""
    return {
        'profile_id': 'CPC001_A375_24H',
        'perturbagen': {
            'name': 'Vorinostat',
            'type': 'small_molecule'
        },
        'cell_line': {
            'name': 'A375',
            'cell_type': 'melanoma'
        },
        'dose': {
            'value': 10,
            'unit': 'uM'
        },
        'time': {
            'value': 24,
            'unit': 'h'
        },
        'gene_signatures': [
            {
                'gene_symbol': 'CDKN1A',
                'zscore': 3.5,
                'pvalue': 0.001,
                'qvalue': 0.005
            },
            {
                'gene_symbol': 'HSPA1A',
                'zscore': 2.8,
                'pvalue': 0.002,
                'qvalue': 0.008
            },
            {
                'gene_symbol': 'BCL2',
                'zscore': -2.5,
                'pvalue': 0.003,
                'qvalue': 0.01
            }
        ],
        'platform': 'L1000',
        'batch': 'BATCH001',
        'quality_metrics': {
            'qc_pass': True,
            'batch_effect': 0.05,
            'reproducibility': 0.95
        }
    }

def test_initialization(cmap_adapter):
    """测试初始化"""
    assert cmap_adapter.api_base_url == "https://api.clue.io/api/v1"
    assert cmap_adapter.data_dir.exists()

@patch('src.data.adapters.data_adapters.CMapAdapter.make_request')
def test_fetch_data(mock_make_request, cmap_adapter, mock_response, 
                   sample_profiles_data, sample_profile_data):
    """测试数据获取"""
    # 模拟API响应
    mock_make_request.side_effect = [
        mock_response(sample_profiles_data),
        mock_response(sample_profile_data),
        mock_response(sample_profile_data)  # 为第二个表达谱返回相同的数据
    ]
    
    # 获取数据
    result = cmap_adapter.fetch_data(limit=1)
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert 'profile_id' in result.columns
    assert 'perturbagen' in result.columns
    assert 'gene_signatures' in result.columns
    
    # 验证数据内容
    assert result.iloc[0]['profile_id'] == 'CPC001_A375_24H'
    assert result.iloc[0]['perturbagen'] == 'Vorinostat'
    assert len(result.iloc[0]['gene_signatures']) == 3

def test_parse_data(cmap_adapter):
    """测试数据解析"""
    # 准备测试数据
    raw_data = pd.DataFrame({
        'profile_id': ['CPC001', 'CPC002', None],
        'perturbagen': ['Drug1', None, 'Drug3'],
        'gene_signatures': [
            [
                {'gene_symbol': 'CDKN1A', 'zscore': 3.5},
                {'gene_symbol': 'BCL2', 'zscore': -2.5}
            ],
            [
                {'gene_symbol': 'HSPA1A', 'zscore': 2.8}
            ],
            []
        ],
        'metadata': [
            {'quality_metrics': {'qc_pass': True, 'batch_effect': 0.05, 'reproducibility': 0.95}},
            {'quality_metrics': {'qc_pass': False, 'batch_effect': 0.2, 'reproducibility': 0.8}},
            {'quality_metrics': {}}
        ]
    })
    
    # 解析数据
    parsed_data = cmap_adapter.parse_data(raw_data)
    
    # 验证结果
    assert len(parsed_data) == 1  # 应该只保留有效数据
    assert 'expression_pattern' in parsed_data.columns
    assert 'quality_score' in parsed_data.columns
    assert parsed_data.iloc[0]['expression_pattern'] == 'mixed'  # 上调和下调基因比例接近
    assert parsed_data.iloc[0]['quality_score'] == 1.0  # 高质量分数

def test_transform_data(cmap_adapter):
    """测试数据转换"""
    # 准备测试数据
    parsed_data = pd.DataFrame({
        'profile_id': ['CPC001'],
        'perturbagen': ['Vorinostat'],
        'perturbagen_type': ['small_molecule'],
        'cell_line': ['A375'],
        'cell_type': ['melanoma'],
        'dose': [10],
        'dose_unit': ['uM'],
        'time': [24],
        'time_unit': ['h'],
        'expression_pattern': ['mixed'],
        'quality_score': [1.0],
        'gene_signatures': [[
            {'gene_symbol': 'CDKN1A', 'zscore': 3.5},
            {'gene_symbol': 'BCL2', 'zscore': -2.5}
        ]]
    })
    
    # 转换数据
    transformed_data = cmap_adapter.transform_data(parsed_data)
    
    # 验证结果
    assert len(transformed_data) == 1
    assert all(col in transformed_data.columns for col in [
        'profile_id', 'perturbagen', 'perturbagen_type',
        'cell_line', 'cell_type', 'dose', 'dose_unit',
        'time', 'time_unit', 'expression_pattern',
        'quality_score', 'gene_signatures', 'data_source'
    ])
    assert transformed_data['data_source'].unique() == ['CMap']

def test_cache_mechanism(cmap_adapter, tmp_path):
    """测试缓存机制"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'profile_id': ['CPC001'],
        'perturbagen': ['Vorinostat'],
        'cell_line': ['A375']
    })
    
    # 保存到缓存
    cache_file = cmap_adapter.data_dir / 'expression_profiles.csv'
    test_data.to_csv(cache_file, index=False)
    
    # 验证从缓存加载
    loaded_data = cmap_adapter.fetch_data()
    assert len(loaded_data) == len(test_data)
    assert 'profile_id' in loaded_data.columns

def test_error_handling(cmap_adapter):
    """测试错误处理"""
    # 准备包含无效数据的DataFrame
    invalid_data = pd.DataFrame({
        'profile_id': [None, None],
        'perturbagen': [None, 'Drug1'],
        'gene_signatures': [None, []]
    })
    
    # 验证解析处理
    parsed_data = cmap_adapter.parse_data(invalid_data)
    assert len(parsed_data) == 0  # 应该过滤掉所有无效数据

def test_expression_pattern_classification(cmap_adapter):
    """测试表达模式分类"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'profile_id': ['P1', 'P2', 'P3', 'P4'],
        'perturbagen': ['D1', 'D2', 'D3', 'D4'],
        'gene_signatures': [
            # 主要上调
            [
                {'gene_symbol': 'G1', 'zscore': 2.5},
                {'gene_symbol': 'G2', 'zscore': 3.0},
                {'gene_symbol': 'G3', 'zscore': -1.0}
            ],
            # 主要下调
            [
                {'gene_symbol': 'G1', 'zscore': -2.5},
                {'gene_symbol': 'G2', 'zscore': -3.0},
                {'gene_symbol': 'G3', 'zscore': 1.0}
            ],
            # 混合模式
            [
                {'gene_symbol': 'G1', 'zscore': 2.5},
                {'gene_symbol': 'G2', 'zscore': -2.5}
            ],
            # 空列表
            []
        ],
        'metadata': [
            {'quality_metrics': {'qc_pass': True, 'batch_effect': 0.05, 'reproducibility': 0.95}},
            {'quality_metrics': {'qc_pass': True, 'batch_effect': 0.05, 'reproducibility': 0.95}},
            {'quality_metrics': {'qc_pass': True, 'batch_effect': 0.05, 'reproducibility': 0.95}},
            {'quality_metrics': {'qc_pass': True, 'batch_effect': 0.05, 'reproducibility': 0.95}}
        ]
    })
    
    # 解析数据
    parsed_data = cmap_adapter.parse_data(test_data)
    
    # 验证表达模式
    patterns = parsed_data['expression_pattern'].tolist()
    assert patterns == ['up_regulated', 'down_regulated', 'mixed', 'unknown']

def test_quality_score_calculation(cmap_adapter):
    """测试质量评分计算"""
    # 准备测试数据
    test_data = pd.DataFrame({
        'profile_id': ['P1', 'P2', 'P3'],
        'perturbagen': ['D1', 'D2', 'D3'],
        'gene_signatures': [[], [], []],
        'metadata': [
            # 高质量
            {
                'quality_metrics': {
                    'qc_pass': True,
                    'batch_effect': 0.05,
                    'reproducibility': 0.95
                }
            },
            # 中等质量
            {
                'quality_metrics': {
                    'qc_pass': True,
                    'batch_effect': 0.2,
                    'reproducibility': 0.8
                }
            },
            # 低质量
            {
                'quality_metrics': {
                    'qc_pass': False,
                    'batch_effect': 0.4,
                    'reproducibility': 0.6
                }
            }
        ]
    })
    
    # 解析数据
    parsed_data = cmap_adapter.parse_data(test_data)
    
    # 验证质量分数
    scores = parsed_data['quality_score'].tolist()
    assert scores[0] == 1.0  # 高质量
    assert 0.5 < scores[1] < 1.0  # 中等质量
    assert scores[2] < 0.5  # 低质量 