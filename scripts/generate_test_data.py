#!/usr/bin/env python3
"""Generate synthetic test data for the asteroid detection pipeline."""

import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import argparse


def create_synthetic_image_with_asteroid(
    size=(1024, 1024),
    num_stars=50,
    asteroid_pos=(500, 500),
    asteroid_brightness=800,
    noise_level=100,
):
    """Create a synthetic astronomical image with stars and an asteroid."""
    np.random.seed(42)  # For reproducible results
    
    # Create base image with Poisson noise
    image = np.random.poisson(noise_level, size=size).astype(np.float64)
    
    # Add random stars
    for _ in range(num_stars):
        x = np.random.randint(50, size[1] - 50)
        y = np.random.randint(50, size[0] - 50)
        brightness = np.random.exponential(500) + 200
        
        # Create 2D Gaussian star
        xx, yy = np.meshgrid(
            np.arange(max(0, x-8), min(size[1], x+8)),
            np.arange(max(0, y-8), min(size[0], y+8))
        )
        
        if len(xx) > 0 and len(yy) > 0:
            fwhm = np.random.uniform(2.0, 4.0)
            gaussian = brightness * np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * (fwhm/2.355)**2))
            
            y_start, y_end = max(0, y-8), min(size[0], y+8)
            x_start, x_end = max(0, x-8), min(size[1], x+8)
            
            image[y_start:y_end, x_start:x_end] += gaussian
    
    # Add asteroid
    ast_x, ast_y = asteroid_pos
    xx, yy = np.meshgrid(
        np.arange(max(0, int(ast_x)-6), min(size[1], int(ast_x)+6)),
        np.arange(max(0, int(ast_y)-6), min(size[0], int(ast_y)+6))
    )
    
    if len(xx) > 0 and len(yy) > 0:
        gaussian = asteroid_brightness * np.exp(-((xx-ast_x)**2 + (yy-ast_y)**2) / (2 * 2.5**2))
        
        y_start, y_end = max(0, int(ast_y)-6), min(size[0], int(ast_y)+6)
        x_start, x_end = max(0, int(ast_x)-6), min(size[1], int(ast_x)+6)
        
        image[y_start:y_end, x_start:x_end] += gaussian
    
    return image


def create_fits_header(obs_time, ra_center=150.0, dec_center=30.0, pixel_scale=0.396):
    """Create a basic FITS header with WCS information."""
    header = fits.Header()
    
    # Basic observation info
    header['OBJECT'] = 'Test Field'
    header['EXPTIME'] = 300.0
    header['FILTER'] = 'r'
    header['DATE-OBS'] = obs_time.isot  # Use ISOT format instead of ISO
    header['MJD-OBS'] = obs_time.mjd
    
    # WCS information
    header['CRVAL1'] = ra_center
    header['CRVAL2'] = dec_center
    header['CRPIX1'] = 512.0
    header['CRPIX2'] = 512.0
    header['CDELT1'] = -pixel_scale / 3600.0  # degrees/pixel (negative for RA)
    header['CDELT2'] = pixel_scale / 3600.0   # degrees/pixel
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    
    # Additional headers
    header['TELESCOP'] = 'Test Telescope'
    header['INSTRUME'] = 'Test Camera'
    header['OBSERVER'] = 'Test Observer'
    header['AIRMASS'] = 1.2
    header['SEEING'] = 2.1
    
    return header


def generate_test_sequence(
    output_dir,
    num_images=5,
    time_interval_minutes=30,
    asteroid_velocity_pixels_per_hour=4.0,
):
    """Generate a sequence of test images with a moving asteroid."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} test images in {output_dir}")
    
    # Starting parameters
    start_time = Time('2024-01-01T20:00:00')
    start_pos = (400, 400)  # Starting asteroid position
    
    velocity_x = asteroid_velocity_pixels_per_hour * np.cos(np.radians(45))  # Moving NE
    velocity_y = asteroid_velocity_pixels_per_hour * np.sin(np.radians(45))
    
    for i in range(num_images):
        # Calculate time and position for this frame
        obs_time = start_time + i * time_interval_minutes / (24 * 60)  # Convert to days
        time_hours = i * time_interval_minutes / 60.0
        
        ast_x = start_pos[0] + velocity_x * time_hours
        ast_y = start_pos[1] + velocity_y * time_hours
        
        print(f"  Image {i+1}: t={obs_time.iso}, asteroid at ({ast_x:.1f}, {ast_y:.1f})")
        
        # Generate image
        image = create_synthetic_image_with_asteroid(
            asteroid_pos=(ast_x, ast_y),
            asteroid_brightness=600 + np.random.normal(0, 50),  # Slight brightness variation
        )
        
        # Create FITS file
        header = create_fits_header(obs_time)
        hdu = fits.PrimaryHDU(data=image, header=header)
        
        filename = f"test_image_{i+1:02d}.fits"
        filepath = output_dir / filename
        hdu.writeto(filepath, overwrite=True)
        
        print(f"    Saved: {filepath}")
    
    print(f"\nGenerated {num_images} test images")
    print(f"Asteroid moves from ({start_pos[0]}, {start_pos[1]}) to ({ast_x:.1f}, {ast_y:.1f})")
    print(f"Velocity: {asteroid_velocity_pixels_per_hour:.1f} pixels/hour")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument("--output-dir", "-o", default="./test_data", 
                       help="Output directory for test images")
    parser.add_argument("--num-images", "-n", type=int, default=5,
                       help="Number of images to generate")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Time interval between images in minutes")
    parser.add_argument("--velocity", "-v", type=float, default=4.0,
                       help="Asteroid velocity in pixels per hour")
    
    args = parser.parse_args()
    
    generate_test_sequence(
        output_dir=args.output_dir,
        num_images=args.num_images,
        time_interval_minutes=args.interval,
        asteroid_velocity_pixels_per_hour=args.velocity,
    )
    
    print("\nTo test the pipeline with these images, run:")
    print(f"uv run astro-detect process --input {args.output_dir} --output ./results")


if __name__ == "__main__":
    main()