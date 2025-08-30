"""High-level object detection for astronomical images."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from astro_research.core.exceptions import DetectionError
from astro_research.core.logger import get_logger
from astro_research.core.types import Detection, ImageMetadata, ProcessingConfig
from astro_research.detection.extractor import SourceExtractor


class ObjectDetector:
    """High-level detector for astronomical objects."""
    
    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        extractor: Optional[SourceExtractor] = None,
    ):
        """
        Initialize the object detector.
        
        Args:
            config: Processing configuration
            extractor: Source extractor instance
        """
        self.config = config or ProcessingConfig()
        self.extractor = extractor or SourceExtractor(
            threshold=self.config.detection_threshold,
            min_area=5,
        )
        self.logger = get_logger("detection.detector")
    
    def detect_sources_batch(
        self,
        images: List[np.ndarray],
        metadata_list: List[ImageMetadata],
        masks: Optional[List[np.ndarray]] = None,
    ) -> List[List[Detection]]:
        """
        Detect sources in a batch of images.
        
        Args:
            images: List of image arrays
            metadata_list: List of image metadata
            masks: Optional list of mask arrays
            
        Returns:
            List of detection lists, one per image
        """
        if len(images) != len(metadata_list):
            raise DetectionError(
                f"Number of images ({len(images)}) != metadata ({len(metadata_list)})"
            )
        
        if masks is not None and len(masks) != len(images):
            raise DetectionError(
                f"Number of masks ({len(masks)}) != images ({len(images)})"
            )
        
        all_detections = []
        
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            self.logger.info(f"Detecting sources in image {i+1}/{len(images)}")
            
            try:
                mask = masks[i] if masks else None
                detections = self.extractor.extract_sources(
                    image, mask, metadata
                )
                
                filtered_detections = self.extractor.filter_detections(
                    detections,
                    min_snr=self.config.min_snr,
                    max_ellipticity=0.5,
                )
                
                all_detections.append(filtered_detections)
                
                self.logger.info(
                    f"Image {i+1}: {len(detections)} total, "
                    f"{len(filtered_detections)} after filtering"
                )
                
            except Exception as e:
                self.logger.error(f"Detection failed for image {i+1}: {e}")
                all_detections.append([])
        
        return all_detections
    
    def create_detection_masks(
        self,
        images: List[np.ndarray],
        detection_lists: List[List[Detection]],
        mask_radius: float = 10.0,
    ) -> List[np.ndarray]:
        """
        Create masks around detected sources.
        
        Args:
            images: List of image arrays
            detection_lists: List of detection lists
            mask_radius: Radius around each detection to mask
            
        Returns:
            List of boolean masks
        """
        masks = []
        
        for image, detections in zip(images, detection_lists):
            mask = np.zeros(image.shape, dtype=bool)
            
            for det in detections:
                y_indices, x_indices = np.ogrid[
                    :image.shape[0], :image.shape[1]
                ]
                
                distances = np.sqrt(
                    (x_indices - det.x)**2 + (y_indices - det.y)**2
                )
                
                mask[distances <= mask_radius] = True
            
            masks.append(mask)
        
        return masks
    
    def compute_detection_statistics(
        self,
        detection_lists: List[List[Detection]],
    ) -> Dict:
        """
        Compute statistics across multiple detection lists.
        
        Args:
            detection_lists: List of detection lists
            
        Returns:
            Dictionary of statistics
        """
        total_detections = sum(len(dets) for dets in detection_lists)
        
        if total_detections == 0:
            return {
                "total_detections": 0,
                "mean_per_image": 0,
                "magnitude_stats": {},
                "fwhm_stats": {},
            }
        
        all_detections = [det for dets in detection_lists for det in dets]
        
        magnitudes = [det.magnitude for det in all_detections if det.magnitude < 50]
        fwhms = [det.fwhm for det in all_detections if det.fwhm > 0]
        
        stats = {
            "total_detections": total_detections,
            "mean_per_image": total_detections / len(detection_lists),
            "detections_per_image": [len(dets) for dets in detection_lists],
        }
        
        if magnitudes:
            stats["magnitude_stats"] = {
                "mean": float(np.mean(magnitudes)),
                "median": float(np.median(magnitudes)),
                "std": float(np.std(magnitudes)),
                "min": float(np.min(magnitudes)),
                "max": float(np.max(magnitudes)),
            }
        
        if fwhms:
            stats["fwhm_stats"] = {
                "mean": float(np.mean(fwhms)),
                "median": float(np.median(fwhms)),
                "std": float(np.std(fwhms)),
                "min": float(np.min(fwhms)),
                "max": float(np.max(fwhms)),
            }
        
        return stats
    
    def save_detections(
        self,
        detection_lists: List[List[Detection]],
        output_path: Path,
        format: str = "csv",
    ):
        """
        Save detections to file.
        
        Args:
            detection_lists: List of detection lists
            output_path: Output file path
            format: Output format ('csv', 'json')
        """
        try:
            all_detections = []
            
            for img_idx, detections in enumerate(detection_lists):
                for det in detections:
                    det_dict = {
                        "image_id": det.image_id,
                        "image_index": img_idx,
                        "x": det.x,
                        "y": det.y,
                        "ra": det.ra,
                        "dec": det.dec,
                        "magnitude": det.magnitude,
                        "magnitude_err": det.magnitude_err,
                        "flux": det.flux,
                        "flux_err": det.flux_err,
                        "fwhm": det.fwhm,
                        "ellipticity": det.ellipticity,
                        "timestamp": det.timestamp.iso if det.timestamp else None,
                    }
                    all_detections.append(det_dict)
            
            if format.lower() == "csv":
                import pandas as pd
                df = pd.DataFrame(all_detections)
                df.to_csv(output_path, index=False)
                
            elif format.lower() == "json":
                import json
                with open(output_path, 'w') as f:
                    json.dump(all_detections, f, indent=2)
            
            else:
                raise DetectionError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved {len(all_detections)} detections to {output_path}")
            
        except Exception as e:
            raise DetectionError(f"Failed to save detections: {e}")
    
    def cross_match_catalogs(
        self,
        detections: List[Detection],
        catalog_ra: np.ndarray,
        catalog_dec: np.ndarray,
        match_radius: float = 2.0,  # arcseconds
    ) -> Tuple[List[int], List[int]]:
        """
        Cross-match detections with a reference catalog.
        
        Args:
            detections: List of detections
            catalog_ra: Catalog RA coordinates in degrees
            catalog_dec: Catalog Dec coordinates in degrees
            match_radius: Matching radius in arcseconds
            
        Returns:
            Tuple of (detection_indices, catalog_indices) for matches
        """
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        
        if not detections:
            return [], []
        
        det_coords = SkyCoord(
            ra=[det.ra for det in detections] * u.deg,
            dec=[det.dec for det in detections] * u.deg,
        )
        
        cat_coords = SkyCoord(
            ra=catalog_ra * u.deg,
            dec=catalog_dec * u.deg,
        )
        
        idx_det, idx_cat, d2d, _ = det_coords.search_around_sky(
            cat_coords, match_radius * u.arcsec
        )
        
        self.logger.info(
            f"Cross-matched {len(idx_det)} detections with catalog "
            f"(radius={match_radius} arcsec)"
        )
        
        return idx_det.tolist(), idx_cat.tolist()