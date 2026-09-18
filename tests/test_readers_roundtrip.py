import numpy as np
import pytest

from echoprep.cine import Cine
from echoprep.readers import read_cine
from echoprep.standardize import export_cine


@pytest.mark.parametrize("ext", ["mp4", "avi", "npz"])
def test_video_export_roundtrip(tmp_path, synthetic_cine, ext):
    frames, _ = synthetic_cine
    cine = Cine(frames, 30.0, "synthetic")
    p = export_cine(cine, tmp_path / f"c.{ext}", record={"x": 1})
    assert p.with_suffix(".json").exists()
    if ext == "npz":
        d = np.load(p)
        assert d["frames"].shape == frames.shape and float(d["fps"]) == 30.0
        return
    back = read_cine(p)
    assert back.n_frames == cine.n_frames
    assert back.height == cine.height and back.width == cine.width
    assert abs(back.fps - 30.0) < 0.01
    tol = 6 if ext == "mp4" else 16  # h264 crf 12 vs default-quality MJPEG on noisy speckle
    assert np.abs(back.frames.astype(int) - frames.astype(int)).mean() < tol


def test_dicom_multiframe_roundtrip(tmp_path, synthetic_cine):
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    frames, _ = synthetic_cine
    gray = frames[:, :, :, 0]
    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.UltrasoundMultiFrameImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "US"
    ds.Rows, ds.Columns, ds.NumberOfFrames = gray.shape[1], gray.shape[2], gray.shape[0]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.FrameTime = "33.3333"
    ds.PixelData = gray.tobytes()
    p = tmp_path / "cine.dcm"
    ds.save_as(p, enforce_file_format=True)
    back = read_cine(p, reader="dicom")
    assert back.n_frames == gray.shape[0] and back.height == gray.shape[1]
    assert abs(back.fps - 30.0) < 0.01
    assert np.array_equal(back.frames[..., 0], gray)


def test_nifti_roundtrip(tmp_path, synthetic_cine):
    import nibabel as nib

    frames, _ = synthetic_cine
    gray = frames[:, :, :, 0]                       # (T, H, W)
    arr = np.transpose(gray, (2, 1, 0)).astype(np.float32)  # CAMUS layout (X, Y, T)
    img = nib.Nifti1Image(arr, np.eye(4))
    p = tmp_path / "patient0001_4CH_half_sequence.nii.gz"
    nib.save(img, p)
    (tmp_path / "Info_4CH.cfg").write_text("ED: 1\nES: 20\nNbFrame: 40\nFrameRate: 48.4\n")
    back = read_cine(p, reader="nifti")
    assert back.n_frames == gray.shape[0] and back.height == gray.shape[1] and back.width == gray.shape[2]
    assert abs(back.fps - 48.4) < 1e-6
    assert np.array_equal(back.frames[..., 0], gray)
