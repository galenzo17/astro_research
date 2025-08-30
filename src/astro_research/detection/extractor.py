"""Source extraction using SEP (Source Extractor Python)."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sep
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from astro_research.core.exceptions import DetectionError
from astro_research.core.logger import get_logger
from astro_research.core.types import Detection, ImageMetadata


class SourceExtractor:
    """Extract point sources from astronomical images using SEP."""
    
    def __init__(
        self,
        threshold: float = 5.0,
        min_area: int = 5,
        deblend_nthresh: int = 32,
        deblend_cont: float = 0.005,
        clean: bool = True,
        mask_threshold: float = 0.0,
    ):
        """
        Initialize the source extractor.
        
        Args:
            threshold: Detection threshold in units of background noise sigma
            min_area: Minimum number of pixels for detection
            deblend_nthresh: Number of deblending thresholds
            deblend_cont: Minimum contrast ratio for deblending
            clean: Perform cleaning of spurious detections
            mask_threshold: Threshold for masking bright pixels
        """
        self.threshold = threshold
        self.min_area = min_area
        self.deblend_nthresh = deblend_nthresh
        self.deblend_cont = deblend_cont
        self.clean = clean
        self.mask_threshold = mask_threshold
        self.logger = get_logger("detection.extractor")
    
    def extract_sources(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        metadata: Optional[ImageMetadata] = None,
    ) -> List[Detection]:
        """
        Extract sources from an image.
        
        Args:
            image: Input image array
            mask: Optional mask array (True = masked pixel)
            metadata: Optional image metadata for WCS
            
        Returns:
            List of Detection objects
        """
        try:
            image = image.astype(np.float64)
            
            if mask is not None:
                mask = mask.astype(bool)
            
            mean, median, std = sigma_clipped_stats(
                image, mask=mask, sigma=3.0, maxiters=5
            )
            
            if std <= 0:
                raise DetectionError("Invalid background statistics")
            
            image_sub = image - median
            
            self.logger.info(
                f"Background: mean={mean:.2f}, median={median:.2f}, std={std:.2f}"
            )
            
            objects = sep.extract(
                image_sub,
                thresh=self.threshold * std,
                err=np.ones_like(image_sub) * std,
                mask=mask,
                minarea=self.min_area,
                deblend_nthresh=self.deblend_nthresh,
                deblend_cont=self.deblend_cont,
                clean=self.clean,
            )
            
            detections = self._create_detections(
                objects, image_sub, std, metadata
            )
            
            self.logger.info(f"Extracted {len(detections)} sources")
            return detections
            
        except Exception as e:
            raise DetectionError(
                f"Failed to extract sources: {e}",
                details={
                    "threshold": self.threshold,
                    "image_shape": image.shape,
                    "has_mask": mask is not None,
                }
            )
    
    def _create_detections(
        self,
        objects: np.ndarray,
        image: np.ndarray,
        noise_level: float,
        metadata: Optional[ImageMetadata] = None,
    ) -> List[Detection]:
        """Convert SEP objects to Detection instances."""
        detections = []
        
        wcs = None
        if metadata and hasattr(metadata, 'file_path'):
            try:
                from astropy.io import fits
                with fits.open(metadata.file_path) as hdul:
                    wcs = WCS(hdul[0].header)
            except Exception as e:
                self.logger.warning(f"Could not load WCS: {e}")
        
        for i, obj in enumerate(objects):
            try:
                x, y = float(obj['x']), float(obj['y'])
                
                if wcs is not None:
                    try:
                        ra, dec = wcs.pixel_to_world_values(x, y)
                        ra, dec = float(ra), float(dec)
                    except Exception:
                        ra = dec = 0.0
                else:
                    ra = dec = 0.0
                
                flux = float(obj['flux'])
                # Handle different field names for flux error
                if 'fluxerr' in obj.dtype.names:
                    flux_err = float(obj['fluxerr']) if obj['fluxerr'] > 0 else flux / 10
                elif 'flux_err' in obj.dtype.names:
                    flux_err = float(obj['flux_err']) if obj['flux_err'] > 0 else flux / 10
                else:
                    flux_err = flux / 10  # Default to 10% error
                
                magnitude = -2.5 * np.log10(max(flux, 1e-10)) + 25.0
                mag_err = 2.5 * flux_err / (flux * np.log(10)) if flux > 0 else 99.0
                
                fwhm = 2.0 * np.sqrt(
                    (obj['a'] * obj['b']) / np.pi
                ) if obj['a'] > 0 and obj['b'] > 0 else 2.0
                
                ellipticity = 1.0 - (obj['b'] / obj['a']) if obj['a'] > 0 else 0.0
                
                detection = Detection(
                    x=x,
                    y=y,
                    ra=ra,
                    dec=dec,
                    magnitude=magnitude,
                    magnitude_err=min(mag_err, 9.99),
                    flux=flux,
                    flux_err=flux_err,
                    fwhm=float(fwhm),
                    ellipticity=float(ellipticity),
                    timestamp=metadata.observation_time if metadata else None,
                    image_id=str(metadata.file_path.stem) if metadata else f"img_{i}",
                )
                
                detections.append(detection)
                
            except Exception as e:
                self.logger.warning(f"Failed to create detection {i}: {e}")
        
        return detections
    
    def photometry(
        self,
        image: np.ndarray,
        positions: List[Tuple[float, float]],
        radius: float = 5.0,
        annulus_radii: Tuple[float, float] = (10.0, 15.0),
    ) -> List[Dict]:
        """
        Perform aperture photometry on specified positions.
        
        Args:
            image: Input image
            positions: List of (x, y) positions
            radius: Aperture radius in pixels
            annulus_radii: Inner and outer radii for background annulus
            
        Returns:
            List of photometry results
        """
        try:
            image = image.astype(np.float64)
            
            positions_array = np.array(positions)
            x = positions_array[:, 0]
            y = positions_array[:, 1]
            
            flux, fluxerr, flag = sep.sum_circle(
                image,
                x, y, radius,
                err=np.sqrt(np.abs(image)),
                gain=1.0,
            )
            
            bkg_flux, bkg_fluxerr, bkg_flag = sep.sum_circann(
                image,
                x, y,
                annulus_radii[0], annulus_radii[1],
                err=np.sqrt(np.abs(image)),
            )
            
            bkg_area = np.pi * (annulus_radii[1]**2 - annulus_radii[0]**2)
            aperture_area = np.pi * radius**2
            
            bkg_per_pixel = bkg_flux / bkg_area
            net_flux = flux - bkg_per_pixel * aperture_area
            
            results = []
            for i in range(len(positions)):
                magnitude = -2.5 * np.log10(max(net_flux[i], 1e-10)) + 25.0
                
                results.append({
                    'x': float(x[i]),
                    'y': float(y[i]),
                    'flux': float(net_flux[i]),
                    'flux_err': float(fluxerr[i]),
                    'magnitude': magnitude,
                    'background': float(bkg_per_pixel[i]),
                    'flag': int(flag[i]),
                })
            
            return results
            
        except Exception as e:
            raise DetectionError(f"Photometry failed: {e}")
    
    def filter_detections(
        self,
        detections: List[Detection],
        min_snr: float = 5.0,
        max_ellipticity: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = None,
    ) -> List[Detection]:
        """
        Filter detections based on quality criteria.
        
        Args:
            detections: List of detections to filter
            min_snr: Minimum signal-to-noise ratio
            max_ellipticity: Maximum ellipticity (1.0 = line, 0.0 = circle)
            magnitude_range: Optional (min_mag, max_mag) range
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for det in detections:
            snr = det.flux / det.flux_err if det.flux_err > 0 else 0
            
            if snr < min_snr:
                continue
            
            if det.ellipticity > max_ellipticity:
                continue
            
            if magnitude_range:
                if not (magnitude_range[0] <= det.magnitude <= magnitude_range[1]):
                    continue
            
            filtered.append(det)
        
        self.logger.info(
            f"Filtered {len(detections)} -> {len(filtered)} detections"
        )
        
        return filtered