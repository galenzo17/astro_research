"""Unit tests for detection module."""

import pytest
import numpy as np

from astro_research.detection.extractor import SourceExtractor
from astro_research.detection.detector import ObjectDetector
from astro_research.core.types import Detection


class TestSourceExtractor:
    """Test SourceExtractor functionality."""
    
    def test_extractor_initialization(self):
        """Test SourceExtractor initialization."""
        extractor = SourceExtractor(
            threshold=3.0,
            min_area=10,
            clean=False,
        )
        
        assert extractor.threshold == 3.0
        assert extractor.min_area == 10
        assert extractor.clean is False
    
    def test_extract_sources_synthetic(self, synthetic_image):
        """Test source extraction on synthetic image."""
        extractor = SourceExtractor(threshold=5.0, min_area=5)
        
        detections = extractor.extract_sources(synthetic_image)
        
        # Should detect the 5 synthetic stars we added
        assert len(detections) >= 4  # Allow for some variation in detection
        
        # Check that detections are Detection objects
        assert all(isinstance(det, Detection) for det in detections)
        
        # Check that positions are reasonable
        for det in detections:
            assert 0 <= det.x <= 512
            assert 0 <= det.y <= 512
            assert det.flux > 0
            assert det.magnitude < 50  # Reasonable magnitude
    
    def test_extract_sources_with_mask(self, synthetic_image):
        """Test source extraction with mask."""
        extractor = SourceExtractor(threshold=3.0)
        
        # Create a mask that covers half the image
        mask = np.zeros_like(synthetic_image, dtype=bool)
        mask[:, :256] = True
        
        detections = extractor.extract_sources(synthetic_image, mask=mask)
        
        # Should only find sources in unmasked region
        for det in detections:
            assert det.x >= 256  # Only in right half
    
    def test_filter_detections(self, sample_detections):
        """Test detection filtering."""
        extractor = SourceExtractor()
        
        # Modify one detection to have bad properties
        bad_detection = sample_detections[0]
        bad_detection.ellipticity = 0.8  # Very elongated
        
        filtered = extractor.filter_detections(
            sample_detections,
            min_snr=5.0,
            max_ellipticity=0.5,
        )
        
        # Should filter out the bad detection
        assert len(filtered) == len(sample_detections) - 1
        assert bad_detection not in filtered
    
    def test_photometry(self, synthetic_image):
        """Test aperture photometry."""
        extractor = SourceExtractor()
        
        # Test positions near synthetic stars
        positions = [(100.0, 100.0), (200.0, 150.0)]
        
        results = extractor.photometry(
            synthetic_image,
            positions,
            radius=5.0,
        )
        
        assert len(results) == 2
        
        for result in results:
            assert 'flux' in result
            assert 'magnitude' in result
            assert 'background' in result
            assert result['flux'] > 0


class TestObjectDetector:
    """Test ObjectDetector functionality."""
    
    def test_detector_initialization(self, test_config):
        """Test ObjectDetector initialization."""
        detector = ObjectDetector(config=test_config)
        
        assert detector.config == test_config
        assert detector.extractor is not None
    
    def test_detect_sources_batch(self, synthetic_image_sequence, sample_metadata):
        """Test batch source detection."""
        detector = ObjectDetector()
        
        detection_lists = detector.detect_sources_batch(
            synthetic_image_sequence,
            sample_metadata,
        )
        
        assert len(detection_lists) == len(synthetic_image_sequence)
        
        # Each image should have some detections
        for detection_list in detection_lists:
            assert len(detection_list) > 0
            assert all(isinstance(det, Detection) for det in detection_list)
    
    def test_detection_statistics(self, synthetic_image_sequence, sample_metadata):
        """Test detection statistics computation."""
        detector = ObjectDetector()
        
        detection_lists = detector.detect_sources_batch(
            synthetic_image_sequence,
            sample_metadata,
        )
        
        stats = detector.compute_detection_statistics(detection_lists)
        
        assert 'total_detections' in stats
        assert 'mean_per_image' in stats
        assert 'detections_per_image' in stats
        assert stats['total_detections'] > 0
        assert len(stats['detections_per_image']) == len(detection_lists)
    
    def test_save_detections(self, sample_detections, tmp_path):
        """Test saving detections to file."""
        detector = ObjectDetector()
        
        output_path = tmp_path / "test_detections.csv"
        
        detector.save_detections(
            [sample_detections],
            output_path,
            format="csv",
        )
        
        assert output_path.exists()
        
        # Check file content
        content = output_path.read_text()
        assert "x,y,ra,dec" in content  # CSV header
        assert str(sample_detections[0].x) in content
    
    def test_cross_match_catalogs(self, sample_detections):
        """Test catalog cross-matching."""
        detector = ObjectDetector()
        
        # Create synthetic catalog near detection positions
        catalog_ra = np.array([det.ra + 0.0001 for det in sample_detections[:3]])
        catalog_dec = np.array([det.dec + 0.0001 for det in sample_detections[:3]])
        
        det_indices, cat_indices = detector.cross_match_catalogs(
            sample_detections,
            catalog_ra,
            catalog_dec,
            match_radius=2.0,  # arcseconds
        )
        
        # Should find matches for first 3 detections
        assert len(det_indices) >= 3
        assert len(cat_indices) >= 3
        assert len(det_indices) == len(cat_indices)