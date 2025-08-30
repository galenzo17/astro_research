"""Unit tests for core functionality."""

import pytest
from pathlib import Path

from astro_research.core.logger import setup_logger, get_logger
from astro_research.core.exceptions import PipelineError, DataError, DetectionError
from astro_research.core.types import ProcessingConfig, Detection, ImageMetadata


class TestLogger:
    """Test logging functionality."""
    
    def test_setup_logger(self):
        """Test logger setup."""
        logger = setup_logger("test_logger", level="DEBUG")
        
        assert logger.name == "test_logger"
        assert logger.level == 10  # DEBUG level
    
    def test_get_logger(self):
        """Test getting existing logger."""
        logger1 = setup_logger("shared_logger")
        logger2 = get_logger("shared_logger")
        
        assert logger1 is logger2
    
    def test_file_logging(self, tmp_path):
        """Test logging to file."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("file_logger", log_file=log_file)
        
        logger.info("Test message")
        
        assert log_file.exists()
        assert "Test message" in log_file.read_text()


class TestExceptions:
    """Test custom exceptions."""
    
    def test_pipeline_error(self):
        """Test PipelineError with details."""
        details = {"error_code": 404, "module": "download"}
        error = PipelineError("Test error", details=details)
        
        assert str(error) == "Test error"
        assert error.details == details
    
    def test_data_error_inheritance(self):
        """Test DataError inherits from PipelineError."""
        error = DataError("Data problem")
        
        assert isinstance(error, PipelineError)
        assert str(error) == "Data problem"
    
    def test_detection_error(self):
        """Test DetectionError."""
        error = DetectionError("Detection failed")
        
        assert isinstance(error, PipelineError)
        assert str(error) == "Detection failed"


class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        
        assert config.min_detections == 3
        assert config.max_velocity == 10.0
        assert config.detection_threshold == 5.0
        assert config.alignment_tolerance == 2.0
        assert config.tracking_radius == 10.0
        assert config.min_snr == 5.0
        assert config.use_ml_validation is False
        assert config.parallel_workers == 4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            min_detections=5,
            max_velocity=20.0,
            detection_threshold=3.0,
            use_ml_validation=True,
        )
        
        assert config.min_detections == 5
        assert config.max_velocity == 20.0
        assert config.detection_threshold == 3.0
        assert config.use_ml_validation is True


class TestDetection:
    """Test Detection dataclass."""
    
    def test_detection_creation(self):
        """Test creating Detection object."""
        from astropy.time import Time
        
        detection = Detection(
            x=100.0,
            y=200.0,
            ra=150.5,
            dec=30.2,
            magnitude=18.5,
            magnitude_err=0.1,
            flux=1000.0,
            flux_err=50.0,
            fwhm=2.5,
            ellipticity=0.2,
            timestamp=Time('2024-01-01T00:00:00'),
            image_id="test_img",
        )
        
        assert detection.x == 100.0
        assert detection.y == 200.0
        assert detection.ra == 150.5
        assert detection.dec == 30.2
        assert detection.magnitude == 18.5
        assert detection.image_id == "test_img"
    
    def test_detection_without_timestamp(self):
        """Test Detection without timestamp."""
        detection = Detection(
            x=100.0,
            y=200.0,
            ra=150.5,
            dec=30.2,
            magnitude=18.5,
            magnitude_err=0.1,
            flux=1000.0,
            flux_err=50.0,
            fwhm=2.5,
            ellipticity=0.2,
            timestamp=None,
            image_id="test_img",
        )
        
        assert detection.timestamp is None


class TestImageMetadata:
    """Test ImageMetadata dataclass."""
    
    def test_metadata_creation(self, sample_metadata):
        """Test creating ImageMetadata."""
        metadata = sample_metadata[0]
        
        assert isinstance(metadata.file_path, Path)
        assert metadata.exposure_time == 300.0
        assert metadata.filter_band == "r"
        assert metadata.ra_center == 150.0
        assert metadata.dec_center == 30.0
        assert metadata.telescope == "P48"