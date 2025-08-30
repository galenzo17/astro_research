#!/usr/bin/env python3
"""Demo script to show the visualizations even without moving objects detected."""

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time

from astro_research.preprocessing.processor import ImageProcessor
from astro_research.detection.detector import ObjectDetector
from astro_research.visualization.plotter import AsteroidPlotter
from astro_research.core.types import ProcessingConfig

def main():
    # Set up configuration
    config = ProcessingConfig(
        detection_threshold=2.5,
        min_detections=3,
    )
    
    # Find FITS files
    test_data_dir = Path("./test_data")
    fits_files = sorted(test_data_dir.glob("*.fits"))
    
    if not fits_files:
        print("No FITS files found in test_data/")
        return
    
    print(f"Processing {len(fits_files)} FITS files...")
    
    # Process images
    processor = ImageProcessor()
    images, metadata_list, processing_info = processor.process_image_sequence(
        fits_files,
        align_images=True,
    )
    print(f"Processed {len(images)} images")
    
    # Detect sources
    detector = ObjectDetector(config)
    detection_lists = detector.detect_sources_batch(images, metadata_list)
    
    total_detections = sum(len(dets) for dets in detection_lists)
    print(f"Found {total_detections} total detections")
    
    # Show detection stats per image
    for i, detections in enumerate(detection_lists):
        print(f"  Image {i+1}: {len(detections)} detections")
    
    # Create visualizations
    plotter = AsteroidPlotter()
    output_dir = Path("./demo_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating visualizations...")
    
    # Sky detections plot - this should show all detections
    print("  - Sky detections map...")
    sky_fig = plotter.plot_detections_sky(
        detection_lists, 
        output_path=output_dir / "sky_detections.html",
        show_plot=False
    )
    
    print(f"Created sky plot: {output_dir / 'sky_detections.html'}")
    
    # Even without moving objects, let's create some sample statistics
    if total_detections > 0:
        all_detections = [det for dets in detection_lists for det in dets]
        
        print(f"\nDetection Statistics:")
        print(f"Total detections: {len(all_detections)}")
        print(f"Mean detections per frame: {total_detections/len(detection_lists):.1f}")
        
        # Magnitude statistics
        magnitudes = [det.magnitude for det in all_detections if det.magnitude < 50]
        if magnitudes:
            print(f"Magnitude range: {min(magnitudes):.1f} - {max(magnitudes):.1f}")
            print(f"Mean magnitude: {np.mean(magnitudes):.1f}")
        
        # Position statistics  
        x_positions = [det.x for det in all_detections]
        y_positions = [det.y for det in all_detections]
        print(f"X position range: {min(x_positions):.0f} - {max(x_positions):.0f}")
        print(f"Y position range: {min(y_positions):.0f} - {max(y_positions):.0f}")
    
    print(f"\nOpen {output_dir / 'sky_detections.html'} in your browser to see the detections!")
    print("You should see different colored points for each frame, with the synthetic stars")
    print("and asteroid visible as sources moving across the field.")

if __name__ == "__main__":
    main()