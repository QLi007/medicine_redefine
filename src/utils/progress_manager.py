import os
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
from typing import Any, Dict, Optional

class ProgressManager:
    """进度管理器：处理Colab会话中断和进度保存"""
    
    def __init__(self, project_path: str, auto_save_interval: int = 300):
        """
        初始化进度管理器
        
        参数:
            project_path: 项目根目录
            auto_save_interval: 自动保存间隔（秒），默认5分钟
        """
        self.project_path = Path(project_path)
        self.checkpoint_dir = self.project_path / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.last_save_time = time.time()
        self.current_state = {}
        
        # 加载最新检查点
        self._load_latest_checkpoint()
    
    def save_checkpoint(self, state: Dict[str, Any], name: str = None) -> str:
        """
        保存检查点
        
        参数:
            state: 需要保存的状态字典
            name: 检查点名称（可选）
        
        返回:
            str: 检查点文件路径
        """
        # 生成检查点名称
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = name or f'checkpoint_{timestamp}'
        
        # 保存检查点
        checkpoint_path = self.checkpoint_dir / f'{name}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        # 更新元数据
        self._update_metadata(name, state)
        
        print(f"检查点已保存: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, name: str = None) -> Optional[Dict[str, Any]]:
        """
        加载检查点
        
        参数:
            name: 检查点名称（可选，默认加载最新的）
        
        返回:
            Dict: 检查点状态
        """
        if name is None:
            # 获取最新检查点
            checkpoints = list(self.checkpoint_dir.glob('*.pkl'))
            if not checkpoints:
                return None
            checkpoint_path = max(checkpoints, key=os.path.getctime)
        else:
            checkpoint_path = self.checkpoint_dir / f'{name}.pkl'
            if not checkpoint_path.exists():
                print(f"检查点不存在: {name}")
                return None
        
        # 加载检查点
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        print(f"已加载检查点: {checkpoint_path}")
        return state
    
    def auto_save(self, state: Dict[str, Any]):
        """
        如果达到自动保存间隔，则自动保存检查点
        
        参数:
            state: 当前状态
        """
        current_time = time.time()
        if current_time - self.last_save_time >= self.auto_save_interval:
            self.save_checkpoint(state, 'autosave')
            self.last_save_time = current_time
    
    def _load_latest_checkpoint(self):
        """加载最新的检查点"""
        state = self.load_checkpoint()
        if state:
            self.current_state = state
    
    def _update_metadata(self, name: str, state: Dict[str, Any]):
        """更新检查点元数据"""
        metadata_path = self.checkpoint_dir / 'metadata.json'
        
        # 读取现有元数据
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'checkpoints': []}
        
        # 添加新检查点信息
        checkpoint_info = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'keys': list(state.keys())
        }
        metadata['checkpoints'].append(checkpoint_info)
        
        # 保存更新后的元数据
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def list_checkpoints(self) -> pd.DataFrame:
        """
        列出所有检查点
        
        返回:
            DataFrame: 检查点信息
        """
        metadata_path = self.checkpoint_dir / 'metadata.json'
        if not metadata_path.exists():
            return pd.DataFrame()
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return pd.DataFrame(metadata['checkpoints'])
    
    def clean_old_checkpoints(self, keep_days: int = 7):
        """
        清理旧检查点
        
        参数:
            keep_days: 保留最近几天的检查点
        """
        current_time = time.time()
        for checkpoint_file in self.checkpoint_dir.glob('*.pkl'):
            if checkpoint_file.name == 'autosave.pkl':
                continue
                
            file_time = os.path.getctime(checkpoint_file)
            if (current_time - file_time) > (keep_days * 24 * 3600):
                os.remove(checkpoint_file)
                print(f"已删除旧检查点: {checkpoint_file}")
                
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.current_state
    
    def update_state(self, updates: Dict[str, Any]):
        """更新当前状态"""
        self.current_state.update(updates)
        # 强制进行自动保存，不考虑时间间隔
        self.save_checkpoint(self.current_state, 'autosave')
        self.last_save_time = time.time() 