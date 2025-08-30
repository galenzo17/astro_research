"""Image alignment for astronomical data using star matching."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

from astro_research.core.exceptions import PreprocessingError
from astro_research.core.logger import get_logger
from astro_research.core.types import ImageMetadata


class ImageAligner:
    """Handles alignment of astronomical images using stellar reference points."""
    
    def __init__(self, detection_threshold: float = 5.0, fwhm: float = 3.0):
        """
        Initialize the image aligner.
        
        Args:
            detection_threshold: Threshold for star detection (sigma)
            fwhm: Full width at half maximum for star detection
        """
        self.detection_threshold = detection_threshold
        self.fwhm = fwhm
        self.logger = get_logger("preprocessing.alignment")
    
    def detect_stars(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect stars in an image.
        
        Args:
            image: Image array
            mask: Optional mask for bad pixels
            
        Returns:
            List of (x, y, flux) tuples for detected stars
        """
        try:
            if mask is not None:
                image = image.copy()
                image[mask] = np.nanmedian(image)
            
            from astropy.stats import sigma_clipped_stats
            mean, median, std = sigma_clipped_stats(
                image, sigma=3.0, maxiters=5
            )
            
            finder = DAOStarFinder(
                threshold=self.detection_threshold * std,
                fwhm=self.fwhm,
                brightest=500,  # Limit to brightest stars
                exclude_border=True,
            )
            
            sources = finder(image - median)
            
            if sources is None:
                return []
            
            stars = []
            for star in sources:
                x, y, flux = float(star["xcentroid"]), float(star["ycentroid"]), float(star["flux"])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(flux):
                    stars.append((x, y, flux))
            
            self.logger.info(f"Detected {len(stars)} stars")
            return stars
            
        except Exception as e:
            raise PreprocessingError(
                f"Failed to detect stars: {e}",
                details={"threshold": self.detection_threshold, "fwhm": self.fwhm}
            )
    
    def match_stars(
        self,
        stars1: List[Tuple[float, float, float]],
        stars2: List[Tuple[float, float, float]],
        tolerance: float = 10.0,
        min_matches: int = 10,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Match stars between two images.
        
        Args:
            stars1: Stars from first image
            stars2: Stars from second image
            tolerance: Matching tolerance in pixels
            min_matches: Minimum number of matches required
            
        Returns:
            List of matched star pairs
        """
        try:
            if len(stars1) < min_matches or len(stars2) < min_matches:
                raise PreprocessingError(
                    f"Not enough stars for matching (need {min_matches})"
                )
            
            coords1 = np.array([(x, y) for x, y, _ in stars1])
            coords2 = np.array([(x, y) for x, y, _ in stars2])
            fluxes1 = np.array([f for _, _, f in stars1])
            fluxes2 = np.array([f for _, _, f in stars2])
            
            bright_idx1 = np.argsort(fluxes1)[-50:]  # 50 brightest
            bright_idx2 = np.argsort(fluxes2)[-50:]
            
            matches = []
            
            for i in bright_idx1:
                x1, y1 = coords1[i]
                distances = np.sqrt(
                    (coords2[bright_idx2, 0] - x1) ** 2 +
                    (coords2[bright_idx2, 1] - y1) ** 2
                )
                
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < tolerance:
                    j = bright_idx2[closest_idx]
                    matches.append(((x1, y1), (coords2[j, 0], coords2[j, 1])))
            
            if len(matches) < min_matches:
                raise PreprocessingError(
                    f"Found only {len(matches)} star matches, need {min_matches}"
                )
            
            self.logger.info(f"Matched {len(matches)} stars")
            return matches
            
        except Exception as e:
            raise PreprocessingError(f"Failed to match stars: {e}")
    
    def compute_transform(
        self,
        matches: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> np.ndarray:
        """
        Compute transformation matrix from star matches.
        
        Args:
            matches: List of matched star pairs
            
        Returns:
            3x3 transformation matrix
        """
        try:
            if len(matches) < 4:
                raise PreprocessingError(
                    f"Need at least 4 matches for transformation, got {len(matches)}"
                )
            
            src_pts = np.array([match[0] for match in matches], dtype=np.float32)
            dst_pts = np.array([match[1] for match in matches], dtype=np.float32)
            
            if len(matches) >= 4:
                transform, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    confidence=0.99,
                )
                
                if transform is None:
                    raise PreprocessingError("Failed to compute homography")
                
                inliers = np.sum(mask.ravel())
                self.logger.info(f"Used {inliers}/{len(matches)} matches for transform")
                
                return transform
            else:
                transform = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
                affine_3x3 = np.eye(3)
                affine_3x3[:2, :] = transform
                return affine_3x3
                
        except Exception as e:
            raise PreprocessingError(f"Failed to compute transform: {e}")
    
    def align_image(
        self,
        image: np.ndarray,
        reference_image: np.ndarray,
        tolerance: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align an image to a reference image.
        
        Args:
            image: Image to align
            reference_image: Reference image
            tolerance: Star matching tolerance
            
        Returns:
            Tuple of (aligned_image, transformation_matrix)
        """
        try:
            stars_ref = self.detect_stars(reference_image)
            stars_img = self.detect_stars(image)
            
            matches = self.match_stars(stars_img, stars_ref, tolerance)
            transform = self.compute_transform(matches)
            
            aligned = cv2.warpPerspective(
                image.astype(np.float32),
                transform,
                (reference_image.shape[1], reference_image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            
            self.logger.info("Image alignment completed successfully")
            return aligned, transform
            
        except Exception as e:
            raise PreprocessingError(f"Failed to align image: {e}")
    
    def align_image_sequence(
        self,
        images: List[np.ndarray],
        reference_idx: int = 0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Align a sequence of images to a reference.
        
        Args:
            images: List of images to align
            reference_idx: Index of reference image
            
        Returns:
            Tuple of (aligned_images, transforms)
        """
        if not images:
            return [], []
        
        if reference_idx >= len(images):
            reference_idx = 0
        
        reference = images[reference_idx]
        aligned_images = []
        transforms = []
        
        for i, image in enumerate(images):
            if i == reference_idx:
                aligned_images.append(image)
                transforms.append(np.eye(3))
            else:
                try:
                    aligned, transform = self.align_image(image, reference)
                    aligned_images.append(aligned)
                    transforms.append(transform)
                except Exception as e:
                    self.logger.error(f"Failed to align image {i}: {e}")
                    aligned_images.append(image)
                    transforms.append(np.eye(3))
        
        return aligned_images, transforms