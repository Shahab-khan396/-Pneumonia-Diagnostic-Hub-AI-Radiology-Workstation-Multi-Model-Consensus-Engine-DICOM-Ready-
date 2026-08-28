import pytest
from core.sample_manager import list_sample_catalog, get_sample_info, ensure_samples_generated
from config import SAMPLES_CATALOG


def test_ensure_samples_generated():
    """Verify that sample radiographs are automatically synthesized on disk."""
    samples_dir = ensure_samples_generated()
    assert samples_dir.exists()
    
    for sid, meta in SAMPLES_CATALOG.items():
        sample_file = samples_dir / meta["filename"]
        assert sample_file.exists()
        assert sample_file.stat().st_size > 1000  # Non-empty image file


def test_list_sample_catalog():
    """Verify sample catalog returns all 3 radiograph types with metadata."""
    catalog = list_sample_catalog()
    assert len(catalog) == 3
    sample_ids = [s["id"] for s in catalog]
    assert "sample_normal" in sample_ids
    assert "sample_bacterial" in sample_ids
    assert "sample_viral" in sample_ids


def test_get_sample_info_valid_and_invalid():
    """Verify get_sample_info retrieves correct file path and handles unknown IDs."""
    info = get_sample_info("sample_bacterial")
    assert info is not None
    assert info["category"] == "PNEUMONIA"
    assert info["path"].exists() or info["file_path"].exists()
    
    bad_info = get_sample_info("unknown_nonexistent_sample")
    assert bad_info is None
