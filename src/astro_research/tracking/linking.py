"""Object linking algorithms for connecting detections across frames."""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from astro_research.core.exceptions import DetectionError
from astro_research.core.logger import get_logger
from astro_research.core.types import Detection, MovingObject, ProcessingConfig


class ObjectLinker:
    """Links detections across multiple frames to identify moving objects."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the object linker.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.logger = get_logger("tracking.linking")
    
    def link_detections(
        self,
        detection_lists: List[List[Detection]],
        time_tolerance: float = 3600.0,  # seconds
    ) -> List[MovingObject]:
        """
        Link detections across frames to identify moving objects.
        
        Args:
            detection_lists: List of detection lists, one per image
            time_tolerance: Maximum time gap for linking (seconds)
            
        Returns:
            List of MovingObject instances
        """
        if len(detection_lists) < self.config.min_detections:
            raise DetectionError(
                f"Need at least {self.config.min_detections} frames for tracking"
            )
        
        all_detections = []
        frame_indices = []
        
        for frame_idx, detections in enumerate(detection_lists):
            for det in detections:
                if det.timestamp is not None:
                    all_detections.append(det)
                    frame_indices.append(frame_idx)
        
        if len(all_detections) < self.config.min_detections:
            return []
        
        self.logger.info(f"Linking {len(all_detections)} detections across {len(detection_lists)} frames")
        
        # Group detections by potential tracklets using clustering
        tracklets = self._find_tracklets(all_detections, frame_indices)
        
        # Filter tracklets and create MovingObject instances
        moving_objects = []
        for tracklet in tracklets:
            if len(tracklet) >= self.config.min_detections:
                moving_obj = self._create_moving_object(tracklet)
                if moving_obj:
                    moving_objects.append(moving_obj)
        
        self.logger.info(f"Found {len(moving_objects)} potential moving objects")
        return moving_objects
    
    def _find_tracklets(
        self,
        detections: List[Detection],
        frame_indices: List[int],
    ) -> List[List[Detection]]:
        """
        Find potential tracklets using spatial-temporal clustering.
        
        Args:
            detections: All detections
            frame_indices: Frame index for each detection
            
        Returns:
            List of tracklets (lists of detections)
        """
        if not detections:
            return []
        
        # Create feature matrix for clustering
        features = []
        for i, det in enumerate(detections):
            # Use RA, Dec, time, and frame as features
            time_hours = det.timestamp.unix / 3600.0 if det.timestamp else 0
            features.append([
                det.ra,
                det.dec,
                time_hours / 24.0,  # Normalize time to days
                frame_indices[i] / 10.0,  # Normalize frame index
            ])
        
        features = np.array(features)
        
        # Use DBSCAN clustering to group nearby detections
        clustering = DBSCAN(
            eps=self.config.tracking_radius / 3600.0,  # Convert arcsec to degrees
            min_samples=self.config.min_detections,
        )
        
        labels = clustering.fit_predict(features)
        
        # Group detections by cluster label
        tracklets = []
        unique_labels = set(labels) - {-1}  # Exclude noise (-1)
        
        for label in unique_labels:
            cluster_detections = [
                detections[i] for i, l in enumerate(labels) if l == label
            ]
            
            # Sort by time
            cluster_detections.sort(key=lambda d: d.timestamp.unix if d.timestamp else 0)
            
            # Check if detections span multiple frames
            cluster_frames = set(frame_indices[i] for i, l in enumerate(labels) if l == label)
            
            if len(cluster_frames) >= self.config.min_detections:
                tracklets.append(cluster_detections)
        
        return tracklets
    
    def _create_moving_object(self, detections: List[Detection]) -> Optional[MovingObject]:
        """
        Create a MovingObject from a list of linked detections.
        
        Args:
            detections: List of linked detections
            
        Returns:
            MovingObject instance or None if invalid
        """
        try:
            if len(detections) < 2:
                return None
            
            # Sort detections by time
            detections.sort(key=lambda d: d.timestamp.unix if d.timestamp else 0)
            
            # Calculate velocity
            velocity = self._calculate_velocity(detections)
            
            # Check if velocity is reasonable for an asteroid
            velocity_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
            if velocity_magnitude > self.config.max_velocity:
                return None
            
            # Calculate trajectory RMS
            trajectory_rms = self._calculate_trajectory_rms(detections)
            
            # Calculate magnitude statistics
            magnitudes = [det.magnitude for det in detections if det.magnitude < 50]
            if magnitudes:
                mag_mean = np.mean(magnitudes)
                mag_std = np.std(magnitudes)
            else:
                mag_mean = mag_std = 0.0
            
            # Calculate classification score based on trajectory linearity and consistency
            classification_score = self._calculate_classification_score(
                detections, velocity, trajectory_rms
            )
            
            moving_object = MovingObject(
                detections=detections,
                velocity=velocity,
                trajectory_rms=trajectory_rms,
                magnitude_mean=mag_mean,
                magnitude_std=mag_std,
                classification_score=classification_score,
            )
            
            return moving_object
            
        except Exception as e:
            self.logger.warning(f"Failed to create moving object: {e}")
            return None
    
    def _calculate_velocity(self, detections: List[Detection]) -> Tuple[float, float]:
        """
        Calculate mean velocity in RA and Dec.
        
        Args:
            detections: List of detections
            
        Returns:
            Tuple of (ra_velocity, dec_velocity) in arcsec/hour
        """
        if len(detections) < 2:
            return (0.0, 0.0)
        
        ra_velocities = []
        dec_velocities = []
        
        for i in range(len(detections) - 1):
            det1, det2 = detections[i], detections[i + 1]
            
            if det1.timestamp and det2.timestamp:
                dt_hours = (det2.timestamp.unix - det1.timestamp.unix) / 3600.0
                
                if dt_hours > 0:
                    dra = (det2.ra - det1.ra) * 3600.0  # arcsec
                    ddec = (det2.dec - det1.dec) * 3600.0  # arcsec
                    
                    # Account for RA wrapping
                    if abs(dra) > 180 * 3600:
                        dra = dra - np.sign(dra) * 360 * 3600
                    
                    # Account for cos(dec) factor in RA
                    dra *= np.cos(np.radians(det1.dec))
                    
                    ra_velocities.append(dra / dt_hours)
                    dec_velocities.append(ddec / dt_hours)
        
        if ra_velocities and dec_velocities:
            return (np.mean(ra_velocities), np.mean(dec_velocities))
        else:
            return (0.0, 0.0)
    
    def _calculate_trajectory_rms(self, detections: List[Detection]) -> float:
        """
        Calculate RMS deviation from linear trajectory.
        
        Args:
            detections: List of detections
            
        Returns:
            RMS deviation in arcseconds
        """
        if len(detections) < 3:
            return 0.0
        
        # Convert to arrays
        times = np.array([
            det.timestamp.unix if det.timestamp else 0
            for det in detections
        ])
        ras = np.array([det.ra for det in detections])
        decs = np.array([det.dec for det in detections])
        
        # Fit linear trajectories
        ra_coeffs = np.polyfit(times, ras, 1)
        dec_coeffs = np.polyfit(times, decs, 1)
        
        # Calculate predicted positions
        ra_pred = np.polyval(ra_coeffs, times)
        dec_pred = np.polyval(dec_coeffs, times)
        
        # Calculate residuals in arcseconds
        ra_residuals = (ras - ra_pred) * 3600.0
        dec_residuals = (decs - dec_pred) * 3600.0
        
        # Account for cos(dec) factor
        mean_dec = np.mean(decs)
        ra_residuals *= np.cos(np.radians(mean_dec))
        
        # Calculate RMS
        rms = np.sqrt(np.mean(ra_residuals**2 + dec_residuals**2))
        
        return float(rms)
    
    def _calculate_classification_score(
        self,
        detections: List[Detection],
        velocity: Tuple[float, float],
        trajectory_rms: float,
    ) -> float:
        """
        Calculate a classification score for how likely this is a real asteroid.
        
        Args:
            detections: List of detections
            velocity: Calculated velocity
            trajectory_rms: Trajectory RMS
            
        Returns:
            Score between 0 and 1 (1 = very likely asteroid)
        """
        score = 1.0
        
        # Penalize high trajectory RMS (non-linear motion)
        if trajectory_rms > 5.0:  # arcsec
            score *= max(0.1, 1.0 - (trajectory_rms - 5.0) / 10.0)
        
        # Penalize extreme velocities
        velocity_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
        if velocity_magnitude > 50.0:  # arcsec/hour
            score *= max(0.1, 1.0 - (velocity_magnitude - 50.0) / 100.0)
        
        # Reward consistent number of detections
        if len(detections) >= 5:
            score *= 1.2
        
        # Penalize magnitude inconsistency
        magnitudes = [det.magnitude for det in detections if det.magnitude < 50]
        if magnitudes and len(magnitudes) > 2:
            mag_std = np.std(magnitudes)
            if mag_std > 1.0:  # mag
                score *= max(0.5, 1.0 - (mag_std - 1.0) / 2.0)
        
        return min(1.0, max(0.0, score))