import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from src.docking.core.docking_module import DockingModule, DataTier
import json

@pytest.fixture
def docking_module(tmp_path):
    """创建测试用的DockingModule实例"""
    module = DockingModule()
    # 使用临时目录
    module.results_dir = tmp_path / 'docking_results'
    module.baseline_dir = tmp_path / 'baseline_data'
    module.results_dir.mkdir(parents=True)
    module.baseline_dir.mkdir(parents=True)
    return module

@pytest.fixture
def sample_protein():
    """样本蛋白质数据"""
    return {
        "uniprot_id": "P12345",
        "structure_type": "experimental",
        "resolution": 1.8
    }

@pytest.fixture
def sample_ligand():
    """样本配体数据"""
    return {
        "drug_id": "DB00001",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
    }

def test_data_tier_classification(docking_module, sample_protein, sample_ligand):
    """测试数据分层功能"""
    # 测试TIER1：有结合数据+实验结构
    docking_module.add_baseline_data(
        sample_protein["uniprot_id"],
        sample_ligand["drug_id"],
        {"binding_score": 7.5},
        {"resolution": 1.8}
    )
    assert docking_module.get_data_tier(sample_protein, sample_ligand) == DataTier.TIER1

    # 测试TIER2：有结合数据+预测结构
    sample_protein["structure_type"] = "predicted"
    assert docking_module.get_data_tier(sample_protein, sample_ligand) == DataTier.TIER2

    # 测试TIER3：实验结构+无结合数据
    sample_protein["structure_type"] = "experimental"
    new_ligand = {"drug_id": "DB00002", "smiles": "CC(=O)O"}
    assert docking_module.get_data_tier(sample_protein, new_ligand) == DataTier.TIER3

    # 测试TIER4：预测结构+无结合数据
    sample_protein["structure_type"] = "predicted"
    assert docking_module.get_data_tier(sample_protein, new_ligand) == DataTier.TIER4

def test_baseline_data_management(docking_module, sample_protein, sample_ligand):
    """测试baseline数据管理"""
    # 测试添加baseline数据
    exp_data = {"binding_score": 7.5, "ki": 100}
    struct_quality = {"resolution": 1.8}
    
    docking_module.add_baseline_data(
        sample_protein["uniprot_id"],
        sample_ligand["drug_id"],
        exp_data,
        struct_quality
    )
    
    # 验证数据已保存
    key = f"{sample_protein['uniprot_id']}_{sample_ligand['drug_id']}"
    assert key in docking_module.baseline_data
    assert docking_module.baseline_data[key]["experimental_data"] == exp_data
    assert docking_module.baseline_data[key]["structure_quality"] == struct_quality

def test_relative_score_calculation(docking_module, sample_protein):
    """测试相对评分计算"""
    # 准备baseline数据
    baseline_scores = [7.5, 8.0, 8.5, 7.0, 7.8]
    
    # 测试正常情况
    result = docking_module.calculate_relative_score(8.0, sample_protein, baseline_scores)
    assert "relative_score" in result
    assert "z_score" in result
    assert "baseline_stats" in result
    assert "confidence" in result
    
    # 测试无baseline数据情况
    result = docking_module.calculate_relative_score(8.0, sample_protein, [])
    assert result["relative_score"] is None
    assert result["confidence"] == "low"

def test_confidence_assessment(docking_module, sample_protein):
    """测试可信度评估"""
    # 测试数据量少的情况
    assert docking_module._assess_confidence(sample_protein, 2, 1.0) == "low"
    
    # 测试中等数据量
    assert docking_module._assess_confidence(sample_protein, 5, 1.0) == "medium"
    
    # 测试大数据量，低标准差
    assert docking_module._assess_confidence(sample_protein, 15, 1.5) == "high"
    
    # 测试大数据量，高标准差
    assert docking_module._assess_confidence(sample_protein, 15, 2.5) == "medium"

def test_dock_and_evaluate(docking_module, sample_protein, sample_ligand):
    """测试完整的对接和评估流程"""
    # 添加一些baseline数据
    docking_module.add_baseline_data(
        sample_protein["uniprot_id"],
        "DB00002",
        {"binding_score": 7.5},
        {"resolution": 1.8}
    )
    docking_module.add_baseline_data(
        sample_protein["uniprot_id"],
        "DB00003",
        {"binding_score": 8.0},
        {"resolution": 1.8}
    )
    
    # 执行对接和评估
    result = docking_module.dock_and_evaluate(sample_protein, sample_ligand)
    
    # 验证结果
    assert "docking_result" in result
    assert "relative_score" in result
    assert "data_tier" in result
    assert "timestamp" in result
    
    # 验证结果文件已保存
    result_file = docking_module.results_dir / f"{sample_protein['uniprot_id']}_{sample_ligand['drug_id']}.json"
    assert result_file.exists()

def test_result_persistence(docking_module, sample_protein, sample_ligand):
    """测试结果持久化"""
    # 生成测试结果
    test_result = {
        "docking_score": 8.0,
        "relative_score": 0.75,
        "confidence": "high"
    }
    
    # 保存结果
    docking_module._save_result(
        sample_protein["uniprot_id"],
        sample_ligand["drug_id"],
        test_result
    )
    
    # 验证文件存在且内容正确
    result_file = docking_module.results_dir / f"{sample_protein['uniprot_id']}_{sample_ligand['drug_id']}.json"
    assert result_file.exists()
    
    with open(result_file, 'r') as f:
        saved_result = json.load(f)
        assert saved_result == test_result 