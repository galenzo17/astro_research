"""Type definitions and data models for the asteroid detection pipeline."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time


class Survey(Enum):
    """Supported astronomical surveys."""
    
    ZTF = "ztf"  # Zwicky Transient Facility
    ATLAS = "atlas"  # Asteroid Terrestrial-impact Last Alert System
    PANSTARRS = "panstarrs"  # Panoramic Survey Telescope and Rapid Response System
    CATALINA = "catalina"  # Catalina Sky Survey
    

@dataclass
class ImageMetadata:
    """Metadata for astronomical images."""
    
    file_path: Path
    observation_time: Time
    exposure_time: float  # seconds
    filter_band: str
    ra_center: float  # degrees
    dec_center: float  # degrees
    field_of_view: Tuple[float, float]  # degrees
    pixel_scale: float  # arcsec/pixel
    survey: Optional[Survey] = None
    telescope: Optional[str] = None
    instrument: Optional[str] = None
    airmass: Optional[float] = None
    seeing: Optional[float] = None  # arcsec
    

@dataclass
class Detection:
    """Single source detection in an image."""
    
    x: float  # pixel coordinates
    y: float  # pixel coordinates
    ra: float  # degrees
    dec: float  # degrees
    magnitude: float
    magnitude_err: float
    flux: float
    flux_err: float
    fwhm: float  # pixels
    ellipticity: float
    timestamp: Time
    image_id: str
    

@dataclass
class MovingObject:
    """Tracked moving object across multiple frames."""
    
    detections: List[Detection]
    velocity: Tuple[float, float]  # arcsec/hour in RA, Dec
    trajectory_rms: float  # arcsec
    magnitude_mean: float
    magnitude_std: float
    classification_score: float  # 0-1, likelihood of being an asteroid
    mpc_designation: Optional[str] = None
    orbital_elements: Optional[dict] = None
    

@dataclass
class ProcessingConfig:
    """Configuration for the processing pipeline."""
    
    min_detections: int = 3  # Minimum detections for valid track
    max_velocity: float = 10.0  # Maximum velocity in arcsec/sec
    detection_threshold: float = 5.0  # Sigma threshold for detection
    alignment_tolerance: float = 2.0  # Pixels for star matching
    tracking_radius: float = 10.0  # Search radius for linking detections
    min_snr: float = 5.0  # Minimum signal-to-noise ratio
    use_ml_validation: bool = False
    parallel_workers: int = 4