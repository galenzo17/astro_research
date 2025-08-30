"""Integration tests for the full asteroid detection pipeline."""

import pytest
from pathlib import Path

from astro_research.preprocessing.processor import ImageProcessor
from astro_research.detection.detector import ObjectDetector
from astro_research.tracking.tracker import AsteroidTracker
from astro_research.visualization.plotter import AsteroidPlotter
from astro_research.visualization.reporter import DetectionReporter


@pytest.mark.integration
class TestFullPipeline:
    """Test the complete asteroid detection pipeline."""
    
    def test_end_to_end_pipeline(self, temp_fits_sequence, sample_metadata, test_config, tmp_path):
        """Test the full pipeline from FITS files to final results."""
        
        # Step 1: Image preprocessing
        processor = ImageProcessor()
        images, metadata_list, processing_info = processor.process_image_sequence(
            temp_fits_sequence,
            output_dir=tmp_path / "processed",
            align_images=True,
        )
        
        assert len(images) == len(temp_fits_sequence)
        assert len(metadata_list) == len(temp_fits_sequence)
        assert processing_info["num_images"] == len(temp_fits_sequence)
        assert processing_info["aligned"] is True
        
        # Step 2: Source detection
        detector = ObjectDetector(config=test_config)
        detection_lists = detector.detect_sources_batch(images, metadata_list)
        
        assert len(detection_lists) == len(images)
        total_detections = sum(len(dets) for dets in detection_lists)
        assert total_detections > 0
        
        # Step 3: Asteroid tracking
        tracker = AsteroidTracker(config=test_config)
        moving_objects = tracker.track_asteroids(detection_lists, metadata_list)
        
        # Should detect at least the synthetic moving object we created
        assert len(moving_objects) >= 1
        
        # Verify moving object properties
        for obj in moving_objects:
            assert len(obj.detections) >= test_config.min_detections
            assert obj.velocity[0] != 0 or obj.velocity[1] != 0  # Has motion
            assert 0 <= obj.classification_score <= 1
        
        # Step 4: Save results
        output_path = tmp_path / "asteroid_tracks.mpc"
        tracker.save_tracks(moving_objects, output_path, format="mpc")
        assert output_path.exists()
        
        # Step 5: Generate report
        reporter = DetectionReporter()
        report_text = reporter.generate_summary_report(
            detection_lists, metadata_list, moving_objects, processing_info
        )
        
        assert "ASTEROID DETECTION PIPELINE REPORT" in report_text
        assert f"Number of images processed: {len(images)}" in report_text
        assert f"Number of candidates: {len(moving_objects)}" in report_text
        
        # Step 6: Create visualizations
        plotter = AsteroidPlotter()
        
        # Sky detections plot
        sky_fig = plotter.plot_detections_sky(
            detection_lists, 
            output_path=tmp_path / "sky_plot.html",
            show_plot=False
        )
        assert (tmp_path / "sky_plot.html").exists()
        
        # Track plot
        track_fig = plotter.plot_asteroid_tracks(
            moving_objects,
            output_path=tmp_path / "tracks.html",
            show_plot=False
        )
        assert (tmp_path / "tracks.html").exists()
    
    def test_pipeline_with_no_moving_objects(self, temp_fits_file, tmp_path, test_config):
        """Test pipeline behavior when no moving objects are detected."""
        
        # Process single image (no motion possible)
        processor = ImageProcessor()
        images, metadata_list, _ = processor.process_image_sequence([temp_fits_file])
        
        detector = ObjectDetector(config=test_config)
        detection_lists = detector.detect_sources_batch(images, metadata_list)
        
        tracker = AsteroidTracker(config=test_config)
        
        # Should handle gracefully when insufficient frames
        with pytest.raises(Exception):  # Should fail with insufficient data
            tracker.track_asteroids(detection_lists, metadata_list)
    
    def test_pipeline_error_handling(self, tmp_path, test_config):
        """Test pipeline error handling with invalid inputs."""
        
        # Test with non-existent FITS file
        fake_path = tmp_path / "nonexistent.fits"
        processor = ImageProcessor()
        
        with pytest.raises(Exception):
            processor.process_image_sequence([fake_path])
        
        # Test detector with mismatched inputs
        detector = ObjectDetector(config=test_config)
        
        with pytest.raises(Exception):
            detector.detect_sources_batch([], [])  # Empty inputs
    
    @pytest.mark.slow
    def test_performance_with_many_images(self, synthetic_image_sequence, sample_metadata, test_config):
        """Test pipeline performance with many images (marked as slow)."""
        
        # Create more images for performance test
        extended_images = synthetic_image_sequence * 4  # 20 images total
        extended_metadata = sample_metadata * 4
        
        # Update timestamps to avoid duplicates
        for i, metadata in enumerate(extended_metadata):
            metadata.observation_time = metadata.observation_time + i * 600  # 10 min intervals
        
        processor = ImageProcessor()
        images, metadata_list, _ = processor.process_image_sequence(
            extended_images, align_images=False  # Skip alignment for speed
        )
        
        detector = ObjectDetector(config=test_config)
        detection_lists = detector.detect_sources_batch(images, metadata_list)
        
        assert len(detection_lists) == len(extended_images)
        
        # Should handle larger datasets without issues
        total_detections = sum(len(dets) for dets in detection_lists)
        assert total_detections > 0


@pytest.mark.integration  
class TestModuleIntegration:
    """Test integration between individual modules."""
    
    def test_preprocessor_detector_integration(self, temp_fits_sequence, sample_metadata):
        """Test integration between preprocessing and detection."""
        
        processor = ImageProcessor()
        images, metadata_list, _ = processor.process_image_sequence(temp_fits_sequence)
        
        detector = ObjectDetector()
        detection_lists = detector.detect_sources_batch(images, metadata_list)
        
        # Verify integration works correctly
        assert len(detection_lists) == len(images)
        for i, (image, detections) in enumerate(zip(images, detection_lists)):
            assert image.shape[0] > 0 and image.shape[1] > 0
            
            # Detections should be within image bounds
            for det in detections:
                assert 0 <= det.x < image.shape[1]
                assert 0 <= det.y < image.shape[0]
    
    def test_detector_tracker_integration(self, sample_detections, sample_metadata, test_config):
        """Test integration between detection and tracking."""
        
        # Create detection lists with proper timestamps
        detection_lists = []
        for i in range(len(sample_metadata)):
            frame_detections = [sample_detections[i]] if i < len(sample_detections) else []
            # Ensure detection has timestamp from metadata
            if frame_detections:
                frame_detections[0].timestamp = sample_metadata[i].observation_time
            detection_lists.append(frame_detections)
        
        tracker = AsteroidTracker(config=test_config)
        moving_objects = tracker.track_asteroids(detection_lists, sample_metadata)
        
        # Should create at least one moving object from the linked detections
        assert len(moving_objects) >= 1
        
        for obj in moving_objects:
            # Verify tracking results are reasonable
            assert len(obj.detections) >= test_config.min_detections
            assert obj.trajectory_rms >= 0
            assert 0 <= obj.classification_score <= 1
    
    def test_tracker_visualization_integration(self, sample_detections, sample_metadata, test_config, tmp_path):
        """Test integration between tracking and visualization."""
        
        # Create a simple moving object
        detection_lists = [[det] for det in sample_detections]
        
        tracker = AsteroidTracker(config=test_config)
        moving_objects = tracker.track_asteroids(detection_lists, sample_metadata)
        
        if moving_objects:  # Only test if objects were detected
            plotter = AsteroidPlotter()
            
            # Test track visualization
            fig = plotter.plot_asteroid_tracks(
                moving_objects,
                output_path=tmp_path / "integration_tracks.html",
                show_plot=False
            )
            
            assert (tmp_path / "integration_tracks.html").exists()
            
            # Test velocity analysis
            vel_fig = plotter.plot_velocity_distribution(
                moving_objects,
                output_path=tmp_path / "integration_velocity.html", 
                show_plot=False
            )
            
            assert (tmp_path / "integration_velocity.html").exists()
        
        # Test reporter integration
        reporter = DetectionReporter()
        report = reporter.generate_summary_report(
            detection_lists, sample_metadata, moving_objects, {}
        )
        
        assert len(report) > 0
        assert "ASTEROID DETECTION PIPELINE REPORT" in report