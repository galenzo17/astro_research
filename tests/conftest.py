"""Pytest configuration and fixtures for astro_research tests."""

import pytest
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from astro_research.core.types import Detection, ImageMetadata, ProcessingConfig, Survey


@pytest.fixture
def test_config():
    """Provide a test configuration."""
    return ProcessingConfig(
        min_detections=3,
        max_velocity=10.0,
        detection_threshold=3.0,
        alignment_tolerance=2.0,
        tracking_radius=5.0,
        min_snr=3.0,
        use_ml_validation=False,
        parallel_workers=1,
    )


@pytest.fixture
def synthetic_image():
    """Generate a synthetic astronomical image with stars."""
    np.random.seed(42)  # For reproducible tests
    
    # Create base image
    image = np.random.poisson(100, size=(512, 512)).astype(np.float64)
    
    # Add synthetic stars
    star_positions = [
        (100, 100), (200, 150), (300, 200), (400, 300), (150, 400)
    ]
    
    for x, y in star_positions:
        # Create a 2D Gaussian star
        xx, yy = np.meshgrid(
            np.arange(max(0, x-10), min(512, x+10)),
            np.arange(max(0, y-10), min(512, y+10))
        )
        
        if len(xx) > 0 and len(yy) > 0:
            gaussian = 1000 * np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 3**2))
            
            y_start, y_end = max(0, y-10), min(512, y+10)
            x_start, x_end = max(0, x-10), min(512, x+10)
            
            image[y_start:y_end, x_start:x_end] += gaussian
    
    return image


@pytest.fixture
def synthetic_image_sequence():
    """Generate a sequence of synthetic images with a moving object."""
    np.random.seed(42)
    images = []
    
    # Moving object starts at (250, 250) and moves 2 pixels per frame
    moving_obj_positions = [(250 + i*2, 250 + i*1.5) for i in range(5)]
    
    for frame_idx in range(5):
        # Base image with noise
        image = np.random.poisson(100, size=(512, 512)).astype(np.float64)
        
        # Add fixed stars
        star_positions = [(100, 100), (400, 150), (350, 400)]
        for x, y in star_positions:
            xx, yy = np.meshgrid(
                np.arange(max(0, int(x)-8), min(512, int(x)+8)),
                np.arange(max(0, int(y)-8), min(512, int(y)+8))
            )
            if len(xx) > 0 and len(yy) > 0:
                gaussian = 800 * np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 2.5**2))
                y_start, y_end = max(0, int(y)-8), min(512, int(y)+8)
                x_start, x_end = max(0, int(x)-8), min(512, int(x)+8)
                image[y_start:y_end, x_start:x_end] += gaussian
        
        # Add moving object
        mov_x, mov_y = moving_obj_positions[frame_idx]
        xx, yy = np.meshgrid(
            np.arange(max(0, int(mov_x)-6), min(512, int(mov_x)+6)),
            np.arange(max(0, int(mov_y)-6), min(512, int(mov_y)+6))
        )
        if len(xx) > 0 and len(yy) > 0:
            gaussian = 600 * np.exp(-((xx-mov_x)**2 + (yy-mov_y)**2) / (2 * 2**2))
            y_start, y_end = max(0, int(mov_y)-6), min(512, int(mov_y)+6)
            x_start, x_end = max(0, int(mov_x)-6), min(512, int(mov_x)+6)
            image[y_start:y_end, x_start:x_end] += gaussian
        
        images.append(image)
    
    return images


@pytest.fixture
def sample_detections():
    """Provide sample detections for testing."""
    detections = []
    
    for i in range(5):
        det = Detection(
            x=100.0 + i * 2.0,
            y=100.0 + i * 1.5,
            ra=150.0 + i * 0.001,
            dec=30.0 + i * 0.0005,
            magnitude=18.5 + np.random.normal(0, 0.1),
            magnitude_err=0.1,
            flux=1000.0 - i * 50,
            flux_err=50.0,
            fwhm=2.5,
            ellipticity=0.1,
            timestamp=Time('2024-01-01T00:00:00') + i * 3600,  # 1 hour intervals
            image_id=f"test_img_{i}",
        )
        detections.append(det)
    
    return detections


@pytest.fixture
def sample_metadata():
    """Provide sample image metadata for testing."""
    metadata_list = []
    
    for i in range(5):
        metadata = ImageMetadata(
            file_path=Path(f"test_image_{i}.fits"),
            observation_time=Time('2024-01-01T00:00:00') + i * 3600,
            exposure_time=300.0,
            filter_band="r",
            ra_center=150.0,
            dec_center=30.0,
            field_of_view=(2048.0, 2048.0),
            pixel_scale=0.396,
            survey=Survey.ZTF,
            telescope="P48",
            instrument="ZTF",
            airmass=1.2,
            seeing=2.1,
        )
        metadata_list.append(metadata)
    
    return metadata_list


@pytest.fixture
def temp_fits_file(tmp_path, synthetic_image):
    """Create a temporary FITS file for testing."""
    fits_path = tmp_path / "test_image.fits"
    
    # Create basic FITS header
    header = fits.Header()
    header['OBJECT'] = 'Test Field'
    header['EXPTIME'] = 300.0
    header['FILTER'] = 'r'
    header['DATE-OBS'] = '2024-01-01T00:00:00'
    header['CRVAL1'] = 150.0  # RA
    header['CRVAL2'] = 30.0   # Dec
    header['CRPIX1'] = 256.0
    header['CRPIX2'] = 256.0
    header['CDELT1'] = -0.00011  # degrees/pixel
    header['CDELT2'] = 0.00011
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    
    # Create FITS file
    hdu = fits.PrimaryHDU(data=synthetic_image, header=header)
    hdu.writeto(fits_path)
    
    return fits_path


@pytest.fixture
def temp_fits_sequence(tmp_path, synthetic_image_sequence, sample_metadata):
    """Create a sequence of temporary FITS files for testing."""
    fits_paths = []
    
    for i, (image, metadata) in enumerate(zip(synthetic_image_sequence, sample_metadata)):
        fits_path = tmp_path / f"test_image_{i}.fits"
        
        header = fits.Header()
        header['OBJECT'] = 'Test Field'
        header['EXPTIME'] = metadata.exposure_time
        header['FILTER'] = metadata.filter_band
        header['DATE-OBS'] = metadata.observation_time.iso
        header['CRVAL1'] = metadata.ra_center
        header['CRVAL2'] = metadata.dec_center
        header['CRPIX1'] = 256.0
        header['CRPIX2'] = 256.0
        header['CDELT1'] = -metadata.pixel_scale / 3600  # arcsec to degrees
        header['CDELT2'] = metadata.pixel_scale / 3600
        header['CTYPE1'] = 'RA---TAN'
        header['CTYPE2'] = 'DEC--TAN'
        
        hdu = fits.PrimaryHDU(data=image, header=header)
        hdu.writeto(fits_path)
        
        fits_paths.append(fits_path)
    
    return fits_paths


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"


# Skip tests that require external data if not available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "external_data: marks tests that require external data"
    )