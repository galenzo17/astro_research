"""Image calibration for astronomical data."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from astro_research.core.exceptions import PreprocessingError
from astro_research.core.logger import get_logger
from astro_research.core.types import ImageMetadata


class ImageCalibrator:
    """Handles astrometric and photometric calibration of astronomical images."""
    
    def __init__(self):
        """Initialize the image calibrator."""
        self.logger = get_logger("preprocessing.calibration")
    
    def basic_calibration(
        self,
        image_data: np.ndarray,
        metadata: ImageMetadata,
        subtract_background: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Perform basic image calibration.
        
        Args:
            image_data: Raw image data array
            metadata: Image metadata
            subtract_background: Whether to subtract background
            
        Returns:
            Tuple of (calibrated_image, calibration_info)
        """
        try:
            calibrated = image_data.astype(np.float32)
            info = {"steps": []}
            
            if subtract_background:
                background = self._estimate_background(calibrated)
                calibrated -= background
                info["background_level"] = background
                info["steps"].append("background_subtraction")
            
            calibrated = self._handle_bad_pixels(calibrated)
            info["steps"].append("bad_pixel_correction")
            
            info["mean"] = float(np.mean(calibrated))
            info["std"] = float(np.std(calibrated))
            info["min"] = float(np.min(calibrated))
            info["max"] = float(np.max(calibrated))
            
            return calibrated, info
            
        except Exception as e:
            raise PreprocessingError(
                "Failed during basic image calibration",
                details={"error": str(e), "image_shape": image_data.shape}
            )
    
    def _estimate_background(self, image: np.ndarray) -> float:
        """
        Estimate the background level using sigma clipping.
        
        Args:
            image: Image array
            
        Returns:
            Background level
        """
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        return median
    
    def _handle_bad_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Replace bad pixels (NaN, inf) with median values.
        
        Args:
            image: Image array
            
        Returns:
            Image with bad pixels replaced
        """
        bad_mask = ~np.isfinite(image)
        
        if np.any(bad_mask):
            median_val = np.nanmedian(image)
            image[bad_mask] = median_val
            self.logger.info(f"Replaced {np.sum(bad_mask)} bad pixels")
        
        return image
    
    def load_and_calibrate(
        self,
        fits_path: Path,
        output_path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, ImageMetadata, dict]:
        """
        Load and calibrate a FITS image.
        
        Args:
            fits_path: Path to FITS file
            output_path: Optional output path for calibrated image
            
        Returns:
            Tuple of (calibrated_image, metadata, calibration_info)
        """
        try:
            with fits.open(fits_path) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header
                
                if image_data is None:
                    raise PreprocessingError(
                        f"No image data found in {fits_path}"
                    )
                
                metadata = self._parse_metadata(fits_path, header)
                
                calibrated_image, cal_info = self.basic_calibration(
                    image_data, metadata
                )
                
                if output_path:
                    self._save_calibrated_image(
                        calibrated_image, header, output_path
                    )
                
                return calibrated_image, metadata, cal_info
                
        except Exception as e:
            raise PreprocessingError(
                f"Failed to load and calibrate {fits_path}",
                details={"error": str(e), "file": str(fits_path)}
            )
    
    def _parse_metadata(self, fits_path: Path, header: fits.Header) -> ImageMetadata:
        """Parse basic metadata from FITS header."""
        try:
            from astropy.time import Time
            
            return ImageMetadata(
                file_path=fits_path,
                observation_time=Time(
                    header.get("DATE-OBS", header.get("MJD-OBS", 0)),
                    format="isot" if "DATE-OBS" in header else "mjd"
                ),
                exposure_time=header.get("EXPTIME", 0),
                filter_band=header.get("FILTER", "unknown"),
                ra_center=header.get("CRVAL1", 0),
                dec_center=header.get("CRVAL2", 0),
                field_of_view=(
                    header.get("NAXIS1", 0) * abs(header.get("CDELT1", 0.00027)) * 3600,
                    header.get("NAXIS2", 0) * abs(header.get("CDELT2", 0.00027)) * 3600,
                ),
                pixel_scale=abs(header.get("CDELT1", 0.00027)) * 3600,
            )
        except Exception as e:
            raise PreprocessingError(f"Failed to parse metadata: {e}")
    
    def _save_calibrated_image(
        self,
        image: np.ndarray,
        original_header: fits.Header,
        output_path: Path,
    ):
        """Save calibrated image to FITS file."""
        try:
            header = original_header.copy()
            header["HISTORY"] = "Calibrated by astro_research pipeline"
            
            hdu = fits.PrimaryHDU(data=image, header=header)
            hdu.writeto(output_path, overwrite=True)
            
            self.logger.info(f"Saved calibrated image to {output_path}")
            
        except Exception as e:
            raise PreprocessingError(
                f"Failed to save calibrated image: {e}"
            )