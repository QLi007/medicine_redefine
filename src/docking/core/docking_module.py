import hashlib
import json
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 全局设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DOCKING_RESULTS_DIR = PROJECT_ROOT / 'data' / 'docking_results'
BASELINE_DATA_DIR = PROJECT_ROOT / 'data' / 'baseline_data'

class DataTier(Enum):
    """数据质量分层"""
    TIER1 = "tier1"  # 有结合数据 + 实验3D结构
    TIER2 = "tier2"  # 有结合数据 + 预测3D结构
    TIER3 = "tier3"  # 有实验3D结构 + 无结合数据
    TIER4 = "tier4"  # 预测3D结构 + 无结合数据

class DockingModule:
    """
    DockingModule实现分子对接和评分系统，支持：
    1. 基于已知结合数据的baseline建立
    2. 相对评分计算
    3. 多层次数据质量评估
    4. 结果可靠性评估
    """
    def __init__(self):
        self.results_dir = DOCKING_RESULTS_DIR
        self.baseline_dir = BASELINE_DATA_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化baseline数据
        self.baseline_data = {}
        self._load_baseline_data()

    def _load_baseline_data(self):
        """加载baseline数据"""
        baseline_file = self.baseline_dir / "baseline_data.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline_data = json.load(f)

    def save_baseline_data(self):
        """保存baseline数据"""
        baseline_file = self.baseline_dir / "baseline_data.json"
        with open(baseline_file, 'w') as f:
            json.dump(self.baseline_data, f, indent=2)

    def add_baseline_data(self, protein_id: str, ligand_id: str, 
                         experimental_data: Dict, structure_quality: Dict):
        """
        添加baseline数据
        
        参数:
            protein_id: 蛋白质ID
            ligand_id: 配体ID
            experimental_data: 实验数据，包含Kd/Ki/IC50等
            structure_quality: 结构质量信息，包含分辨率等
        """
        key = f"{protein_id}_{ligand_id}"
        self.baseline_data[key] = {
            "experimental_data": experimental_data,
            "structure_quality": structure_quality,
            "timestamp": str(datetime.now())
        }
        self.save_baseline_data()

    def get_data_tier(self, protein: Dict, ligand: Dict) -> DataTier:
        """
        确定数据质量层级
        
        参数:
            protein: 蛋白质信息字典
            ligand: 配体信息字典
        返回:
            DataTier: 数据质量层级
        """
        has_exp_binding = self._has_experimental_binding(protein, ligand)
        has_exp_structure = self._has_experimental_structure(protein)
        
        if has_exp_binding and has_exp_structure:
            return DataTier.TIER1
        elif has_exp_binding:
            return DataTier.TIER2
        elif has_exp_structure:
            return DataTier.TIER3
        else:
            return DataTier.TIER4

    def calculate_relative_score(self, docking_score: float, 
                               protein: Dict, baseline_scores: List[float]) -> Dict:
        """
        计算相对评分
        
        参数:
            docking_score: 当前对接得分
            protein: 蛋白质信息
            baseline_scores: 该蛋白质的baseline得分列表
        返回:
            Dict: 相对评分结果
        """
        if not baseline_scores:
            return {
                "relative_score": None,
                "confidence": "low",
                "reason": "No baseline data available"
            }
            
        baseline_mean = np.mean(baseline_scores)
        baseline_std = np.std(baseline_scores)
        
        # 计算Z-score
        z_score = (docking_score - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        # 计算相对分数（转换到0-1区间）
        relative_score = 1 / (1 + np.exp(-z_score))
        
        # 评估可信度
        confidence = self._assess_confidence(protein, len(baseline_scores), baseline_std)
        
        return {
            "relative_score": relative_score,
            "z_score": z_score,
            "baseline_stats": {
                "mean": baseline_mean,
                "std": baseline_std,
                "sample_size": len(baseline_scores)
            },
            "confidence": confidence
        }

    def dock_and_evaluate(self, protein: Dict, ligand: Dict, 
                         mode: str = "fast") -> Dict:
        """
        执行对接并评估结果
        
        参数:
            protein: 蛋白质信息
            ligand: 配体信息
            mode: 对接模式 ("fast" 或 "deep")
        返回:
            Dict: 评估结果
        """
        # 确定数据层级
        data_tier = self.get_data_tier(protein, ligand)
        
        # 执行对接
        docking_result = self.fast_dock(protein, ligand) if mode == "fast" \
                        else self.deep_dock(protein, ligand)
        
        # 获取baseline数据
        baseline_scores = self._get_baseline_scores(protein["uniprot_id"])
        
        # 计算相对评分
        relative_score = self.calculate_relative_score(
            docking_result["docking_score"], 
            protein,
            baseline_scores
        )
        
        # 整合结果
        result = {
            "docking_result": docking_result,
            "relative_score": relative_score,
            "data_tier": data_tier.value,
            "timestamp": str(datetime.now())
        }
        
        # 保存结果
        self._save_result(protein["uniprot_id"], ligand["drug_id"], result)
        
        return result

    def _assess_confidence(self, protein: Dict, 
                          baseline_count: int, baseline_std: float) -> str:
        """评估结果可信度"""
        if baseline_count < 3:
            return "low"
        elif baseline_count < 10:
            return "medium"
        else:
            return "high" if baseline_std < 2.0 else "medium"

    def _get_baseline_scores(self, protein_id: str) -> List[float]:
        """获取特定蛋白质的baseline得分"""
        scores = []
        for key, data in self.baseline_data.items():
            if key.startswith(f"{protein_id}_"):
                scores.append(data["experimental_data"].get("binding_score"))
        return [s for s in scores if s is not None]

    def _save_result(self, protein_id: str, ligand_id: str, result: Dict):
        """保存对接结果"""
        result_file = self.results_dir / f"{protein_id}_{ligand_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    def _has_experimental_binding(self, protein: Dict, ligand: Dict) -> bool:
        """检查是否有实验结合数据"""
        key = f"{protein.get('uniprot_id')}_{ligand.get('drug_id')}"
        return key in self.baseline_data

    def _has_experimental_structure(self, protein: Dict) -> bool:
        """检查是否有实验结构"""
        return protein.get('structure_type') == 'experimental'

    def fast_dock(self, protein, drug):
        """
        快速对接：提供初步粗筛。
        参数:
          protein (dict): 至少包含 'uniprot_id'
          drug (dict): 至少包含 'drug_id'
        返回:
          dict: 对接结果字典, 包括对接模式、得分和相关描述。
        """
        docking_score = self._simulate_docking(protein, drug, mode="fast")
        docking_result = {
            "mode": "fast",
            "docking_score": docking_score,
            "details": "fast docking simulation"
        }
        return docking_result

    def deep_dock(self, protein, drug):
        """
        深度对接：提供更精细的对接计算结果。
        参数同 fast_dock。
        返回:
          dict: 对接结果字典。
        """
        docking_score = self._simulate_docking(protein, drug, mode="deep")
        docking_result = {
            "mode": "deep",
            "docking_score": docking_score,
            "details": "deep docking simulation"
        }
        return docking_result

    def dynamics_simulation(self, protein, drug):
        """
        动力学模拟：对关键候选组合进行分子动力学模拟，评估结合稳定性。
        参数同 fast_dock。
        返回:
          dict: 动力学模拟结果字典, 包含动力学得分和描述信息。
        """
        dynamics_score = self._simulate_dynamics(protein, drug)
        dynamics_result = {
            "dynamics_score": dynamics_score,
            "details": "dynamics simulation result"
        }
        return dynamics_result

    def evaluate_with_baseline(self, docking_result, baseline_data):
        """
        基于实验基准数据对 docking 结果进行多维综合评分。
        参数:
          docking_result (dict): 对接得到的评分信息
          baseline_data (dict): 基准数据, 至少包含 "baseline_score" 键，对应实验数据计算的得分
        返回:
          dict: 综合评价结果，包括最终评分、各子指标得分及评价方式。
        """
        exp_score = baseline_data.get("baseline_score", 0)
        docking_score = docking_result.get("docking_score", 0)
        # 这里使用加权求和，权重根据实验反馈可进行调整
        final_score = 0.6 * docking_score + 0.4 * exp_score
        evaluation = {
            "final_score": final_score,
            "docking_score": docking_score,
            "experimental_score": exp_score,
            "method": "baseline evaluation"
        }
        return evaluation

    def evaluate_with_binding_energy(self, docking_result):
        """
        当无实验基准数据时，直接使用 docking 得分（结合能）进行评价。
        参数:
          docking_result (dict): 对接结果信息
        返回:
          dict: 综合评价结果。
        """
        docking_score = docking_result.get("docking_score", 0)
        # 此处可加入其它公式转换，当前示例直接使用 docking_score
        final_score = docking_score
        evaluation = {
            "final_score": final_score,
            "docking_score": docking_score,
            "method": "binding energy only"
        }
        return evaluation

    def _simulate_docking(self, protein, drug, mode="fast"):
        """
        模拟 docking 得分计算。实际使用时，此函数应调用真实的 docking 工具。
        此处通过 protein['uniprot_id'] 和 drug['drug_id'] 生成一个伪随机但可重复的得分。
        参数:
          mode: "fast" 或者 "deep"，用于区分不同的对接策略
        返回:
          float: 模拟的 docking 得分
        """
        key = f"{protein.get('uniprot_id')}_{drug.get('drug_id')}_{mode}"
        score = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1000 / 100.0
        return score

    def _simulate_dynamics(self, protein, drug):
        """
        模拟分子动力学得分计算。同样为伪随机方法，实际应用中需使用分子动力学模拟工具。
        返回:
          float: 模拟的动力学得分
        """
        key = f"{protein.get('uniprot_id')}_{drug.get('drug_id')}_dynamics"
        score = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 1000 / 100.0
        return score 