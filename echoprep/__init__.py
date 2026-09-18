"""echoprep: standardized preprocessing for echocardiography cines from heterogeneous sources.

Pipeline: raw file (MP4/AVI/DICOM/NIfTI/MHD/image sequence)
          -> Cine (T,H,W,3 uint8 RGB + metadata)
          -> imaging-region (sector) detection
          -> crop / mask / pad
          -> canonical output + JSON sidecar.
"""
from echoprep.cine import Cine
from echoprep.readers import read_cine
from echoprep.sector.motion import SectorResult, detect_sector
from echoprep.standardize import export_cine, standardize_cine

__all__ = ["Cine", "SectorResult", "detect_sector", "export_cine", "read_cine", "standardize_cine"]
__version__ = "0.2.0"
