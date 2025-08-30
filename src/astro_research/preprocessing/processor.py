"""High-level image processing pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from astro_research.core.exceptions import PreprocessingError
from astro_research.core.logger import get_logger
from astro_research.core.types import ImageMetadata
from astro_research.preprocessing.alignment import ImageAligner
from astro_research.preprocessing.calibration import ImageCalibrator


class ImageProcessor:
    """High-level processor for astronomical image preprocessing pipeline."""
    
    def __init__(
        self,
        calibrator: Optional[ImageCalibrator] = None,
        aligner: Optional[ImageAligner] = None,
    ):
        """
        Initialize the image processor.
        
        Args:
            calibrator: Image calibrator instance
            aligner: Image aligner instance
        """
        self.calibrator = calibrator or ImageCalibrator()
        self.aligner = aligner or ImageAligner()
        self.logger = get_logger("preprocessing.processor")
    
    def process_image_sequence(
        self,
        fits_paths: List[Path],
        output_dir: Optional[Path] = None,
        align_images: bool = True,
        reference_idx: int = 0,
    ) -> Tuple[List[np.ndarray], List[ImageMetadata], Dict]:
        """
        Process a sequence of FITS images through the full pipeline.
        
        Args:
            fits_paths: List of FITS file paths
            output_dir: Optional output directory for processed images
            align_images: Whether to align images to reference
            reference_idx: Index of reference image for alignment
            
        Returns:
            Tuple of (processed_images, metadata_list, processing_info)
        """
        if not fits_paths:
            raise PreprocessingError("No FITS files provided")
        
        self.logger.info(f"Processing {len(fits_paths)} images")
        
        images = []
        metadata_list = []
        calibration_info = []
        
        for i, fits_path in enumerate(fits_paths):
            try:
                self.logger.info(f"Processing image {i+1}/{len(fits_paths)}: {fits_path.name}")
                
                image, metadata, cal_info = self.calibrator.load_and_calibrate(
                    fits_path
                )
                
                images.append(image)
                metadata_list.append(metadata)
                calibration_info.append(cal_info)
                
            except Exception as e:
                self.logger.error(f"Failed to process {fits_path}: {e}")
                raise
        
        if align_images and len(images) > 1:
            self.logger.info("Aligning images to reference")
            
            try:
                aligned_images, transforms = self.aligner.align_image_sequence(
                    images, reference_idx
                )
                images = aligned_images
                
                for i, transform in enumerate(transforms):
                    if i != reference_idx:
                        self.logger.debug(f"Transform matrix for image {i}: {transform}")
                        
            except Exception as e:
                self.logger.error(f"Alignment failed: {e}")
                self.logger.warning("Proceeding with unaligned images")
        
        processing_info = {
            "num_images": len(images),
            "reference_idx": reference_idx,
            "aligned": align_images and len(images) > 1,
            "calibration_info": calibration_info,
        }
        
        if output_dir:
            self._save_processed_images(images, metadata_list, output_dir)
        
        self.logger.info("Image processing completed")
        return images, metadata_list, processing_info
    
    def create_master_image(
        self,
        images: List[np.ndarray],
        method: str = "median",
        sigma_clip: Optional[float] = 3.0,
    ) -> np.ndarray:
        """
        Create a master (combined) image from multiple exposures.
        
        Args:
            images: List of aligned images
            method: Combination method ('median', 'mean', 'sum')
            sigma_clip: Optional sigma for sigma clipping
            
        Returns:
            Combined master image
        """
        if not images:
            raise PreprocessingError("No images provided for combination")
        
        self.logger.info(f"Combining {len(images)} images using {method}")
        
        image_stack = np.stack(images, axis=0)
        
        if sigma_clip is not None:
            from astropy.stats import sigma_clip
            image_stack = sigma_clip(
                image_stack, sigma=sigma_clip, axis=0, masked=True
            ).filled(np.nan)
        
        if method == "median":
            combined = np.nanmedian(image_stack, axis=0)
        elif method == "mean":
            combined = np.nanmean(image_stack, axis=0)
        elif method == "sum":
            combined = np.nansum(image_stack, axis=0)
        else:
            raise PreprocessingError(f"Unknown combination method: {method}")
        
        self.logger.info(f"Master image created with shape {combined.shape}")
        return combined
    
    def create_difference_images(
        self,
        images: List[np.ndarray],
        reference_image: Optional[np.ndarray] = None,
        reference_idx: int = 0,
    ) -> List[np.ndarray]:
        """
        Create difference images by subtracting a reference.
        
        Args:
            images: List of images
            reference_image: Reference image (if None, uses images[reference_idx])
            reference_idx: Index of reference image if reference_image is None
            
        Returns:
            List of difference images
        """
        if not images:
            return []
        
        if reference_image is None:
            if reference_idx >= len(images):
                reference_idx = 0
            reference_image = images[reference_idx]
        
        self.logger.info(f"Creating difference images against reference")
        
        diff_images = []
        for i, image in enumerate(images):
            diff = image - reference_image
            diff_images.append(diff)
        
        return diff_images
    
    def _save_processed_images(
        self,
        images: List[np.ndarray],
        metadata_list: List[ImageMetadata],
        output_dir: Path,
    ):
        """Save processed images to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            output_path = output_dir / f"processed_{metadata.file_path.stem}.fits"
            
            try:
                from astropy.io import fits
                
                hdu = fits.PrimaryHDU(data=image)
                hdu.header["HISTORY"] = "Processed by astro_research pipeline"
                hdu.header["PROCTIME"] = metadata.observation_time.iso
                hdu.writeto(output_path, overwrite=True)
                
                self.logger.debug(f"Saved processed image: {output_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save processed image {i}: {e}")