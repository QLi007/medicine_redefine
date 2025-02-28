import pytest
from pathlib import Path
import json
import tempfile
from unittest.mock import MagicMock, patch
from src.data.storage.google_drive_manager import GoogleDriveManager

@pytest.fixture
def mock_drive():
    """模拟Google Drive对象"""
    with patch('src.data.storage.google_drive_manager.GoogleDrive') as mock:
        drive = mock.return_value
        # 模拟文件列表方法
        list_file = MagicMock()
        list_file.GetList.return_value = []
        drive.ListFile.return_value = list_file
        yield drive

@pytest.fixture
def mock_auth():
    """模拟Google认证"""
    with patch('src.data.storage.google_drive_manager.GoogleAuth') as mock:
        yield mock.return_value

@pytest.fixture
def drive_manager(tmp_path, mock_drive, mock_auth):
    """创建测试用的Drive管理器"""
    # 创建临时凭证文件
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    
    return GoogleDriveManager(
        credentials_path=str(credentials),
        cache_dir=str(tmp_path / "cache")
    )

def test_initialization(drive_manager):
    """测试初始化"""
    assert drive_manager.cache_dir.exists()
    assert drive_manager.folder_ids == {}

def test_ensure_folder(drive_manager, mock_drive):
    """测试文件夹创建和获取"""
    # 模拟文件夹不存在的情况
    folder = MagicMock()
    folder['id'] = 'test_folder_id'
    mock_drive.CreateFile.return_value = folder
    
    # 测试创建新文件夹
    folder_id = drive_manager.ensure_folder("test/folder")
    assert folder_id == 'test_folder_id'
    assert drive_manager.folder_ids["test/folder"] == 'test_folder_id'
    
    # 测试获取已存在的文件夹
    folder_id = drive_manager.ensure_folder("test/folder")
    assert folder_id == 'test_folder_id'
    mock_drive.CreateFile.assert_called_once()

def test_upload_file(drive_manager, mock_drive, tmp_path):
    """测试文件上传"""
    # 创建测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    # 模拟上传
    file = MagicMock()
    file['id'] = 'test_file_id'
    mock_drive.CreateFile.return_value = file
    
    result = drive_manager.upload_file(
        str(test_file),
        "test/folder/test.txt"
    )
    
    assert result['id'] == 'test_file_id'
    assert (drive_manager.cache_dir / "test/folder/test.txt").exists()

def test_download_file(drive_manager, mock_drive, tmp_path):
    """测试文件下载"""
    # 模拟文件存在
    file = MagicMock()
    mock_drive.ListFile().GetList.return_value = [file]
    
    # 测试下载到指定位置
    local_path = str(tmp_path / "downloaded.txt")
    result = drive_manager.download_file(
        "test/folder/test.txt",
        local_path
    )
    
    assert result == local_path
    assert Path(local_path).parent.exists()
    
    # 测试使用缓存
    cached_result = drive_manager.download_file(
        "test/folder/test.txt",
        local_path,
        use_cache=True
    )
    assert cached_result == local_path
    file.GetContentFile.assert_called_once()

def test_list_files(drive_manager, mock_drive):
    """测试文件列表获取"""
    # 模拟文件列表
    files = [
        {'id': 'file1', 'title': 'test1.txt'},
        {'id': 'file2', 'title': 'test2.txt'}
    ]
    mock_drive.ListFile().GetList.return_value = files
    
    result = drive_manager.list_files("test/folder")
    assert len(result) == 2
    assert result[0]['id'] == 'file1'
    assert result[1]['title'] == 'test2.txt'

def test_delete_file(drive_manager, mock_drive, tmp_path):
    """测试文件删除"""
    # 创建缓存文件
    cache_file = drive_manager.cache_dir / "test/folder/test.txt"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("test content")
    
    # 模拟文件存在
    file = MagicMock()
    mock_drive.ListFile().GetList.return_value = [file]
    
    drive_manager.delete_file("test/folder/test.txt")
    
    file.Delete.assert_called_once()
    assert not cache_file.exists()

def test_clear_cache(drive_manager, tmp_path):
    """测试缓存清理"""
    # 创建一些缓存文件
    cache_files = [
        "test/folder1/file1.txt",
        "test/folder2/file2.txt"
    ]
    
    for file_path in cache_files:
        cache_file = drive_manager.cache_dir / file_path
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("test content")
    
    drive_manager.clear_cache()
    
    # 验证所有缓存文件都被删除
    for file_path in cache_files:
        assert not (drive_manager.cache_dir / file_path).exists() 