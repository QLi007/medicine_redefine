import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.data.adapters.data_adapters import (
    PDBAdapter,
    PDBbindAdapter,
    BindingDBAdapter,
    UniProtAdapter,
    AlphaFoldAdapter,
    ChEMBLAdapter,
    DrugBankAdapter,
    DisGeNETAdapter,
    ZINCAdapter,
    OpenTargetsAdapter,
    CMapAdapter
)

@pytest.fixture
def base_dir(tmp_path):
    """创建测试用的基础目录"""
    return str(tmp_path)

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
def all_adapters(base_dir):
    """创建所有适配器的实例"""
    return {
        'pdb': PDBAdapter(base_dir),
        'pdbbind': PDBbindAdapter(base_dir),
        'bindingdb': BindingDBAdapter(base_dir),
        'uniprot': UniProtAdapter(base_dir),
        'alphafold': AlphaFoldAdapter(base_dir),
        'chembl': ChEMBLAdapter(base_dir),
        'drugbank': DrugBankAdapter(base_dir),
        'disgenet': DisGeNETAdapter(base_dir),
        'zinc': ZINCAdapter(base_dir),
        'opentargets': OpenTargetsAdapter(base_dir),
        'cmap': CMapAdapter(base_dir)
    }

@pytest.fixture
def sample_data():
    """创建示例数据"""
    return {
        'uniprot': pd.DataFrame({
            'uniprot_id': ['P12345', 'P67890'],
            'entry_name': ['TEST1_HUMAN', 'TEST2_HUMAN'],
            'protein_name': ['Test protein 1', 'Test protein 2'],
            'gene_name': ['TEST1', 'TEST2'],
            'length': [100, 200],
            'sequence': ['MTEST', 'MTEST2']
        }),
        'pdb': pd.DataFrame({
            'pdb_id': ['1abc', '2def'],
            'uniprot_id': ['P12345', 'P67890'],
            'chain_id': ['A', 'B'],
            'resolution': [1.5, 2.0],
            'experiment_type': ['X-ray', 'X-ray'],
            'data_source': ['PDB', 'PDB']
        }),
        'drugbank': pd.DataFrame({
            'drugbank_id': ['DB00001', 'DB00002'],
            'name': ['Drug1', 'Drug2'],
            'smiles': ['CC1=CC=CC=C1', 'CC2=CC=CC=C2'],
            'description': ['Test drug 1', 'Test drug 2'],
            'data_source': ['DrugBank', 'DrugBank']
        }),
        'chembl': pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2'],
            'standard_value': [1.0, 2.0],
            'standard_type': ['IC50', 'Ki'],
            'standard_units': ['nM', 'nM'],
            'data_source': ['ChEMBL', 'ChEMBL']
        }),
        'disgenet': pd.DataFrame({
            'disease_id': ['C0001', 'C0002'],
            'disease_name': ['Disease1', 'Disease2'],
            'gene_id': ['1234', '5678'],
            'score': [0.8, 0.7],
            'data_source': ['DisGeNET', 'DisGeNET']
        }),
        'opentargets': pd.DataFrame({
            'target_id': ['ENSG1', 'ENSG2'],
            'disease_id': ['EFO1', 'EFO2'],
            'score': [0.9, 0.8],
            'evidence_count': [10, 8],
            'data_source': ['OpenTargets', 'OpenTargets']
        }),
        'cmap': pd.DataFrame({
            'profile_id': ['CPC001', 'CPC002'],
            'perturbagen': ['Drug1', 'Drug2'],
            'gene_signatures': [
                [{'gene_symbol': 'G1', 'zscore': 2.5}],
                [{'gene_symbol': 'G2', 'zscore': -2.5}]
            ],
            'metadata': [
                {'quality_metrics': {'qc_pass': True}},
                {'quality_metrics': {'qc_pass': True}}
            ],
            'data_source': ['CMap', 'CMap']
        })
    }

@patch('src.data.adapters.data_adapters.DataSourceAdapter.make_request')
def test_protein_data_integration(mock_make_request, all_adapters, mock_response, sample_data):
    """测试蛋白质数据整合"""
    # 模拟API响应
    mock_make_request.side_effect = [
        mock_response({'results': sample_data['uniprot'].to_dict('records')}),
        mock_response({
            'P12345': [
                {
                    'pdb_id': '1abc',
                    'mappings': [{'chain_id': 'A'}],
                    'resolution': 1.5,
                    'experimental_method': 'X-ray'
                }
            ]
        }),
        mock_response({
            'P67890': [
                {
                    'pdb_id': '2def',
                    'mappings': [{'chain_id': 'B'}],
                    'resolution': 2.0,
                    'experimental_method': 'X-ray'
                }
            ]
        })
    ]
    
    # 获取示例蛋白质数据
    uniprot_data = all_adapters['uniprot'].fetch_data(limit=2)
    assert len(uniprot_data) > 0
    
    uniprot_ids = uniprot_data['uniprot_id'].tolist()
    
    # 获取结构数据
    pdb_data = all_adapters['pdb'].fetch_data(uniprot_ids=uniprot_ids)
    assert len(pdb_data) > 0
    assert all(id in pdb_data['uniprot_id'].tolist() for id in uniprot_ids)
    assert 'data_source' in pdb_data.columns
    assert all(source == 'PDB' for source in pdb_data['data_source'])

@patch('src.data.adapters.data_adapters.DataSourceAdapter.make_request')
def test_drug_data_integration(mock_make_request, all_adapters, mock_response, sample_data):
    """测试药物数据整合"""
    # 模拟DrugBank数据响应
    mock_make_request.side_effect = [
        mock_response({'drugs': sample_data['drugbank'].to_dict('records')}),
        mock_response({
            'targets': [
                {
                    'target_chembl_id': 'CHEMBL1',
                    'target_components': [{'accession': 'P12345'}]
                }
            ]
        }),
        mock_response({
            'activities': [
                {
                    'molecule_chembl_id': 'CHEMBL1',
                    'standard_type': 'IC50',
                    'standard_value': 1.0,
                    'standard_units': 'nM',
                    'pchembl_value': 7.0,
                    'assay_type': 'B',
                    'assay_description': 'Test assay'
                }
            ]
        })
    ]
    
    # 获取示例药物数据
    drugbank_data = all_adapters['drugbank'].fetch_data(limit=2)
    assert len(drugbank_data) > 0
    assert 'data_source' in drugbank_data.columns
    assert all(source == 'DrugBank' for source in drugbank_data['data_source'])
    
    # 获取活性数据
    chembl_data = all_adapters['chembl'].fetch_data(limit=2)
    assert len(chembl_data) > 0
    assert 'data_source' in chembl_data.columns
    assert all(source == 'ChEMBL' for source in chembl_data['data_source'])

@patch('src.data.adapters.data_adapters.DataSourceAdapter.make_request')
def test_disease_data_integration(mock_make_request, all_adapters, mock_response, sample_data):
    """测试疾病数据整合"""
    # 模拟DisGeNET数据响应
    mock_make_request.side_effect = [
        mock_response(sample_data['disgenet'].to_dict('records')),
        mock_response({
            'data': {
                'target': {
                    'id': 'ENSG1',
                    'approvedSymbol': 'TEST1',
                    'associatedDiseases': {
                        'rows': [
                            {
                                'disease': {
                                    'id': 'EFO1',
                                    'name': 'Disease1',
                                    'therapeuticAreas': [{'name': 'Area1'}]
                                },
                                'score': 0.9,
                                'datatypeScores': [
                                    {'id': 'genetic_association', 'score': 0.8}
                                ],
                                'evidenceCount': 10
                            }
                        ]
                    }
                }
            }
        })
    ]
    
    # 获取疾病-基因关联
    disgenet_data = all_adapters['disgenet'].fetch_data(limit=2)
    assert len(disgenet_data) > 0
    assert 'data_source' in disgenet_data.columns
    assert all(source == 'DisGeNET' for source in disgenet_data['data_source'])
    
    # 获取靶点-疾病关联
    opentargets_data = all_adapters['opentargets'].fetch_data(target_ids=['ENSG1'])
    assert len(opentargets_data) > 0
    assert 'data_source' in opentargets_data.columns
    assert all(source == 'OpenTargets' for source in opentargets_data['data_source'])

@patch('src.data.adapters.data_adapters.DataSourceAdapter.make_request')
def test_expression_data_integration(mock_make_request, all_adapters, mock_response, sample_data):
    """测试表达数据整合"""
    # 模拟CMap数据响应
    mock_make_request.return_value = mock_response({
        'data': {
            'profiles': [
                {
                    'profile_id': 'CPC001',
                    'perturbagen': {
                        'name': 'Drug1',
                        'type': 'small_molecule'
                    },
                    'cell_line': {
                        'name': 'A375',
                        'cell_type': 'melanoma'
                    },
                    'gene_signatures': [
                        {
                            'gene_symbol': 'CDKN1A',
                            'zscore': 3.5,
                            'pvalue': 0.001
                        },
                        {
                            'gene_symbol': 'BCL2',
                            'zscore': -2.5,
                            'pvalue': 0.003
                        }
                    ],
                    'quality_metrics': {
                        'qc_pass': True,
                        'batch_effect': 0.05,
                        'reproducibility': 0.95
                    }
                }
            ]
        }
    })
    
    # 获取表达谱数据
    cmap_data = all_adapters['cmap'].fetch_data(limit=2)
    assert len(cmap_data) > 0
    
    # 验证数据格式
    assert 'profile_id' in cmap_data.columns
    assert 'perturbagen' in cmap_data.columns
    assert 'cell_line' in cmap_data.columns
    assert 'gene_signatures' in cmap_data.columns
    assert 'expression_pattern' in cmap_data.columns
    assert 'quality_score' in cmap_data.columns
    assert 'data_source' in cmap_data.columns
    assert all(source == 'CMap' for source in cmap_data['data_source'])

def test_data_transformation_consistency(all_adapters, sample_data):
    """测试数据转换一致性"""
    # 更新示例数据以匹配实际格式
    data_mapping = {
        'uniprot': pd.DataFrame({
            'uniprot_id': ['P12345', 'P67890'],
            'entry_name': ['TEST1_HUMAN', 'TEST2_HUMAN'],
            'protein_name': ['Test protein 1', 'Test protein 2'],
            'gene_name': ['TEST1', 'TEST2'],
            'length': [100, 200],
            'sequence': ['MTEST', 'MTEST2']
        }),
        'pdb': pd.DataFrame({
            'pdb_id': ['1abc', '2def'],
            'uniprot_id': ['P12345', 'P67890'],
            'chain_id': ['A', 'B'],
            'resolution': [1.5, 2.0],
            'experiment_type': ['X-ray', 'X-ray']
        }),
        'chembl': pd.DataFrame({
            'chembl_id': ['CHEMBL1', 'CHEMBL2'],
            'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2'],
            'standard_value': [1.0, 2.0],
            'standard_type': ['IC50', 'Ki'],
            'standard_units': ['nM', 'nM'],
            'pchembl_value': [7.0, 6.5],
            'assay_type': ['B', 'B'],
            'assay_description': ['Test assay 1', 'Test assay 2']
        }),
        'drugbank': pd.DataFrame({
            'drugbank_id': ['DB00001', 'DB00002'],
            'name': ['Drug1', 'Drug2'],
            'smiles': ['CC1=CC=CC=C1', 'CC2=CC=CC=C2'],
            'description': ['Test drug 1', 'Test drug 2'],
            'mechanism_of_action': ['MOA1', 'MOA2'],
            'indication': ['IND1', 'IND2']
        }),
        'disgenet': pd.DataFrame({
            'disease_id': ['C0001', 'C0002'],
            'disease_name': ['Disease1', 'Disease2'],
            'gene_id': ['1234', '5678'],
            'gene_symbol': ['GENE1', 'GENE2'],
            'score': [0.8, 0.7],
            'evidence_count': [10, 8],
            'pmid_count': [5, 4]
        }),
        'opentargets': pd.DataFrame({
            'target_id': ['ENSG1', 'ENSG2'],
            'target_symbol': ['GENE1', 'GENE2'],
            'disease_id': ['EFO1', 'EFO2'],
            'disease_name': ['Disease1', 'Disease2'],
            'score': [0.9, 0.8],
            'evidence_count': [10, 8],
            'datatype_scores': [
                {'genetic_association': 0.8},
                {'genetic_association': 0.7}
            ]
        }),
        'cmap': pd.DataFrame({
            'profile_id': ['CPC001', 'CPC002'],
            'perturbagen': ['Drug1', 'Drug2'],
            'cell_line': ['A375', 'MCF7'],
            'gene_signatures': [
                [{'gene_symbol': 'G1', 'zscore': 2.5}],
                [{'gene_symbol': 'G2', 'zscore': -2.5}]
            ],
            'quality_metrics': [
                {'qc_pass': True, 'batch_effect': 0.05},
                {'qc_pass': True, 'batch_effect': 0.06}
            ]
        })
    }
    
    for name, adapter in all_adapters.items():
        if name in data_mapping:
            print(f"测试 {name} 适配器的数据转换...")
            
            # 解析数据
            parsed_data = adapter.parse_data(data_mapping[name])
            assert len(parsed_data) > 0, f"{name} 适配器解析数据失败"
            
            # 转换数据
            transformed_data = adapter.transform_data(parsed_data)
            assert len(transformed_data) > 0, f"{name} 适配器转换数据失败"
            
            # 验证数据源标记
            assert 'data_source' in transformed_data.columns, f"{name} 适配器缺少数据源标记"
            expected_source = name.replace('adapter', '').upper()
            assert all(source == expected_source for source in transformed_data['data_source']), \
                f"{name} 适配器数据源标记不正确"

def test_error_handling_consistency(all_adapters):
    """测试错误处理一致性"""
    # 为每个适配器准备特定的无效数据
    invalid_data_mapping = {
        'uniprot': pd.DataFrame({
            'uniprot_id': [None, ''],
            'entry_name': [None, ''],
            'protein_name': [None, ''],
            'gene_name': [None, ''],
            'length': [None, -1],
            'sequence': [None, '']
        }),
        'pdb': pd.DataFrame({
            'pdb_id': [None, ''],
            'uniprot_id': [None, ''],
            'chain_id': [None, ''],
            'resolution': [None, -1.0],
            'experiment_type': [None, '']
        }),
        'chembl': pd.DataFrame({
            'chembl_id': [None, ''],
            'molecule_chembl_id': [None, ''],
            'standard_value': [None, -1.0],
            'standard_type': [None, ''],
            'standard_units': [None, ''],
            'pchembl_value': [None, -1.0]
        }),
        'drugbank': pd.DataFrame({
            'drugbank_id': [None, ''],
            'name': [None, ''],
            'smiles': [None, ''],
            'description': [None, '']
        }),
        'disgenet': pd.DataFrame({
            'disease_id': [None, ''],
            'disease_name': [None, ''],
            'gene_id': [None, ''],
            'score': [None, -1.0]
        }),
        'opentargets': pd.DataFrame({
            'target_id': [None, ''],
            'disease_id': [None, ''],
            'score': [None, -1.0],
            'evidence_count': [None, -1]
        }),
        'cmap': pd.DataFrame({
            'profile_id': [None, ''],
            'perturbagen': [None, ''],
            'gene_signatures': [None, []],
            'quality_metrics': [None, {}]
        })
    }
    
    for name, adapter in all_adapters.items():
        if name in invalid_data_mapping:
            print(f"测试 {name} 适配器的错误处理...")
            
            # 测试无效数据处理
            invalid_data = invalid_data_mapping[name]
            parsed_data = adapter.parse_data(invalid_data)
            assert len(parsed_data) == 0, f"{name} 适配器未能正确过滤无效数据"
            
            # 测试转换空数据
            transformed_data = adapter.transform_data(parsed_data)
            assert len(transformed_data) == 0, f"{name} 适配器未能正确处理空数据"
            
            # 测试None输入
            assert len(adapter.parse_data(None)) == 0, f"{name} 适配器未能正确处理None输入"
            assert len(adapter.transform_data(None)) == 0, f"{name} 适配器未能正确处理None输入"
            
            # 测试空DataFrame输入
            empty_df = pd.DataFrame()
            assert len(adapter.parse_data(empty_df)) == 0, f"{name} 适配器未能正确处理空DataFrame"
            assert len(adapter.transform_data(empty_df)) == 0, f"{name} 适配器未能正确处理空DataFrame"

def test_cache_mechanism_consistency(all_adapters, base_dir, sample_data):
    """测试缓存机制一致性"""
    data_mapping = {
        'uniprot': ('uniprot_proteins.csv', sample_data['uniprot']),
        'drugbank': ('drugbank_drugs.csv', sample_data['drugbank']),
        'chembl': ('chembl_activities.csv', sample_data['chembl']),
        'disgenet': ('disease_gene_associations.csv', sample_data['disgenet']),
        'cmap': ('expression_profiles.csv', sample_data['cmap'])
    }
    
    for name, adapter in all_adapters.items():
        if name in data_mapping:
            # 验证缓存目录创建
            assert adapter.data_dir.exists()
            
            # 准备缓存数据
            cache_file = adapter.data_dir / data_mapping[name][0]
            data_mapping[name][1].to_csv(cache_file, index=False)
            
            # 获取数据（应该从缓存加载）
            data = adapter.fetch_data(limit=2)
            assert len(data) > 0 