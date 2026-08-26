import sys
from pathlib import Path
import numpy as np
import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

# Ensure Flask Application is in sys.path
flask_app_dir = Path(__file__).resolve().parent.parent / "Flask Application"
if str(flask_app_dir) not in sys.path:
    sys.path.insert(0, str(flask_app_dir))

from core.dicom_parser import is_dicom_file, parse_dicom_file


@pytest.fixture
def synthetic_dicom_file(tmp_path):
    """Generate a valid synthetic clinical DICOM radiograph."""
    dcm_path = tmp_path / "test_scan.dcm"
    
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Clinical Tags
    ds.PatientID = "PT-8839-EXP"
    ds.PatientAge = "52Y"
    ds.PatientSex = "F"
    ds.StudyDate = "20260826"
    ds.Modality = "DX"
    ds.BodyPartExamined = "CHEST"
    ds.Manufacturer = "Siemens Healthineers"
    ds.KVP = "120"
    ds.ExposureTime = "25"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 128
    ds.Columns = 128
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    
    # Raw 16-bit lung parenchyma pixel data
    raw_pixels = np.random.randint(200, 2500, (128, 128), dtype=np.uint16)
    ds.PixelData = raw_pixels.tobytes()
    
    ds.save_as(str(dcm_path), write_like_original=False)
    return dcm_path


def test_is_dicom_file_detection(synthetic_dicom_file, tmp_path):
    """Verify DICOM file detection by extension and header."""
    assert is_dicom_file(synthetic_dicom_file) is True
    
    # Non-DICOM file
    non_dcm = tmp_path / "plain_text.txt"
    non_dcm.write_text("not a dicom file")
    assert is_dicom_file(non_dcm) is False


def test_parse_dicom_file_extraction(synthetic_dicom_file, tmp_path):
    """Verify DICOM metadata extraction and 8-bit image conversion."""
    out_jpg = tmp_path / "converted_dcm.jpg"
    img_bgr, meta, saved_path = parse_dicom_file(synthetic_dicom_file, output_jpg_path=out_jpg)
    
    assert meta["is_dicom"] is True
    assert meta["patient_id"] == "PT-8839-EXP"
    assert meta["patient_age"] == "52Y"
    assert meta["patient_sex"] == "F"
    assert meta["modality"] == "DX"
    assert meta["manufacturer"] == "Siemens Healthineers"
    
    assert img_bgr.shape == (128, 128, 3)
    assert img_bgr.dtype == np.uint8
    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000
