import pytest
import os
import time
from pathlib import Path
import shutil
from src.utils.progress_manager import ProgressManager

@pytest.fixture
def temp_project_dir(tmp_path):
    """创建临时项目目录"""
    return str(tmp_path / "test_project")

@pytest.fixture
def progress_manager(temp_project_dir):
    """创建ProgressManager实例"""
    return ProgressManager(temp_project_dir, auto_save_interval=1)  # 设置1秒自动保存用于测试

def test_initialization(progress_manager, temp_project_dir):
    """测试初始化"""
    assert progress_manager.project_path == Path(temp_project_dir)
    assert progress_manager.checkpoint_dir.exists()
    assert progress_manager.auto_save_interval == 1

def test_save_and_load_checkpoint(progress_manager):
    """测试保存和加载检查点"""
    # 准备测试数据
    test_state = {
        'step': 1,
        'data': [1, 2, 3],
        'status': 'processing'
    }
    
    # 保存检查点
    checkpoint_path = progress_manager.save_checkpoint(test_state, 'test_checkpoint')
    assert os.path.exists(checkpoint_path)
    
    # 加载检查点
    loaded_state = progress_manager.load_checkpoint('test_checkpoint')
    assert loaded_state == test_state

def test_auto_save(progress_manager):
    """测试自动保存功能"""
    test_state = {'step': 1}
    
    # 更新状态
    progress_manager.update_state(test_state)
    
    # 等待自动保存间隔
    time.sleep(1.5)
    
    # 验证是否自动保存
    autosave_path = progress_manager.checkpoint_dir / 'autosave.pkl'
    assert autosave_path.exists()
    
    # 加载自动保存的状态
    loaded_state = progress_manager.load_checkpoint('autosave')
    assert loaded_state == test_state

def test_list_checkpoints(progress_manager):
    """测试列出检查点"""
    # 保存多个检查点
    states = [
        {'step': 1},
        {'step': 2},
        {'step': 3}
    ]
    
    for i, state in enumerate(states):
        progress_manager.save_checkpoint(state, f'checkpoint_{i}')
    
    # 获取检查点列表
    checkpoints_df = progress_manager.list_checkpoints()
    assert len(checkpoints_df) >= 3
    assert 'name' in checkpoints_df.columns
    assert 'timestamp' in checkpoints_df.columns

def test_clean_old_checkpoints(progress_manager):
    """测试清理旧检查点"""
    # 保存检查点
    test_state = {'step': 1}
    progress_manager.save_checkpoint(test_state, 'test_checkpoint')
    
    # 清理检查点
    progress_manager.clean_old_checkpoints(keep_days=0)  # 立即清理
    
    # 验证是否只保留自动保存的检查点
    checkpoints = list(progress_manager.checkpoint_dir.glob('*.pkl'))
    assert len(checkpoints) <= 1
    if len(checkpoints) == 1:
        assert checkpoints[0].name == 'autosave.pkl'

def test_state_management(progress_manager):
    """测试状态管理"""
    # 初始状态应该为空
    assert progress_manager.get_state() == {}
    
    # 更新状态
    test_state = {'step': 1}
    progress_manager.update_state(test_state)
    assert progress_manager.get_state()['step'] == 1
    
    # 再次更新状态
    progress_manager.update_state({'status': 'running'})
    current_state = progress_manager.get_state()
    assert current_state['step'] == 1
    assert current_state['status'] == 'running' 