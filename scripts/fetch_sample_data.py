"""
获取各个数据库的示例数据用于分析数据结构
"""

import os
import sys
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.adapters.data_adapters import (
    UniProtAdapter,
    PDBAdapter,
    PDBbindAdapter,
    BindingDBAdapter,
    AlphaFoldAdapter,
    ChEMBLAdapter,
    DrugBankAdapter,
    DisGeNETAdapter,
    ZINCAdapter,
    OpenTargetsAdapter,
    CMapAdapter
)

def fetch_sample_data(sample_size=10):
    """从各个数据库获取示例数据"""
    
    # 创建示例数据目录
    sample_dir = project_root / 'data' / 'samples'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据适配器实例
    adapters = {
        'uniprot': UniProtAdapter(project_root),
        'pdb': PDBAdapter(project_root),
        'pdbbind': PDBbindAdapter(project_root),
        'bindingdb': BindingDBAdapter(project_root),
        'alphafold': AlphaFoldAdapter(project_root),
        'chembl': ChEMBLAdapter(project_root),
        'drugbank': DrugBankAdapter(project_root),
        'disgenet': DisGeNETAdapter(project_root),
        'zinc': ZINCAdapter(project_root),
        'opentargets': OpenTargetsAdapter(project_root),
        'cmap': CMapAdapter(project_root)
    }
    
    # 获取并保存示例数据
    for name, adapter in adapters.items():
        print(f"\n获取{name}数据示例...")
        try:
            # 获取数据
            data = adapter.fetch_data(limit=sample_size)
            
            if data is not None and not data.empty:
                # 确保数据类型正确
                for col in data.columns:
                    if data[col].dtype == 'object':
                        data[col] = data[col].fillna('')  # 将NaN替换为空字符串
                
                # 保存原始数据
                raw_file = sample_dir / f'{name}_raw.csv'
                data.to_csv(raw_file, index=False)
                print(f"原始数据已保存到: {raw_file}")
                print(f"数据形状: {data.shape}")
                print("列名:", list(data.columns))
                print("\n数据预览:")
                print(data.head(2))
                
                try:
                    # 解析数据
                    parsed_data = adapter.parse_data(data)
                    if parsed_data is not None and not parsed_data.empty:
                        # 确保数据类型正确
                        for col in parsed_data.columns:
                            if parsed_data[col].dtype == 'object':
                                parsed_data[col] = parsed_data[col].fillna('')
                        
                        parsed_file = sample_dir / f'{name}_parsed.csv'
                        parsed_data.to_csv(parsed_file, index=False)
                        print(f"\n解析后数据已保存到: {parsed_file}")
                        print(f"解析后数据形状: {parsed_data.shape}")
                        print("解析后列名:", list(parsed_data.columns))
                        
                        try:
                            # 转换数据
                            transformed_data = adapter.transform_data(parsed_data)
                            if transformed_data is not None and not transformed_data.empty:
                                # 确保数据类型正确
                                for col in transformed_data.columns:
                                    if transformed_data[col].dtype == 'object':
                                        transformed_data[col] = transformed_data[col].fillna('')
                                
                                transformed_file = sample_dir / f'{name}_transformed.csv'
                                transformed_data.to_csv(transformed_file, index=False)
                                print(f"\n转换后数据已保存到: {transformed_file}")
                                print(f"转换后数据形状: {transformed_data.shape}")
                                print("转换后列名:", list(transformed_data.columns))
                        except Exception as e:
                            print(f"转换{name}数据时出错: {str(e)}")
                except Exception as e:
                    print(f"解析{name}数据时出错: {str(e)}")
            else:
                print(f"未能获取到{name}数据")
                
        except Exception as e:
            print(f"获取{name}数据时出错: {str(e)}")
            continue
            
        print("\n" + "="*80)

if __name__ == "__main__":
    fetch_sample_data() 