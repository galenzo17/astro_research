"""High-level asteroid tracking system."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.time import Time

from astro_research.core.exceptions import DetectionError
from astro_research.core.logger import get_logger
from astro_research.core.types import Detection, ImageMetadata, MovingObject, ProcessingConfig
from astro_research.tracking.linking import ObjectLinker


class AsteroidTracker:
    """High-level asteroid tracking system combining detection and linking."""
    
    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        linker: Optional[ObjectLinker] = None,
    ):
        """
        Initialize the asteroid tracker.
        
        Args:
            config: Processing configuration
            linker: Object linker instance
        """
        self.config = config or ProcessingConfig()
        self.linker = linker or ObjectLinker(config)
        self.logger = get_logger("tracking.tracker")
    
    def track_asteroids(
        self,
        detection_lists: List[List[Detection]],
        metadata_list: List[ImageMetadata],
    ) -> List[MovingObject]:
        """
        Track asteroids across multiple images.
        
        Args:
            detection_lists: List of detection lists, one per image
            metadata_list: List of image metadata
            
        Returns:
            List of MovingObject instances representing asteroid tracks
        """
        if len(detection_lists) != len(metadata_list):
            raise DetectionError(
                f"Mismatch: {len(detection_lists)} detection lists, "
                f"{len(metadata_list)} metadata entries"
            )
        
        if len(detection_lists) < self.config.min_detections:
            raise DetectionError(
                f"Need at least {self.config.min_detections} images for tracking"
            )
        
        self.logger.info(
            f"Tracking asteroids across {len(detection_lists)} images"
        )
        
        # Ensure detections have proper timestamps from metadata
        self._assign_timestamps(detection_lists, metadata_list)
        
        # Link detections across frames
        moving_objects = self.linker.link_detections(detection_lists)
        
        # Filter by quality criteria
        filtered_objects = self._filter_moving_objects(moving_objects)
        
        self.logger.info(
            f"Found {len(moving_objects)} candidates, "
            f"{len(filtered_objects)} after quality filtering"
        )
        
        return filtered_objects
    
    def _assign_timestamps(
        self,
        detection_lists: List[List[Detection]],
        metadata_list: List[ImageMetadata],
    ):
        """Assign timestamps to detections based on image metadata."""
        for detections, metadata in zip(detection_lists, metadata_list):
            for detection in detections:
                if detection.timestamp is None and metadata.observation_time:
                    detection.timestamp = metadata.observation_time
    
    def _filter_moving_objects(
        self,
        moving_objects: List[MovingObject],
    ) -> List[MovingObject]:
        """
        Filter moving objects based on quality criteria.
        
        Args:
            moving_objects: List of MovingObject instances
            
        Returns:
            Filtered list of MovingObject instances
        """
        filtered = []
        
        for obj in moving_objects:
            # Minimum number of detections
            if len(obj.detections) < self.config.min_detections:
                continue
            
            # Velocity constraints
            velocity_magnitude = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
            if velocity_magnitude > self.config.max_velocity:
                self.logger.debug(
                    f"Rejected object: velocity {velocity_magnitude:.1f} > "
                    f"max {self.config.max_velocity}"
                )
                continue
            
            # Trajectory quality
            if obj.trajectory_rms > 10.0:  # arcsec
                self.logger.debug(
                    f"Rejected object: trajectory RMS {obj.trajectory_rms:.1f} > 10.0"
                )
                continue
            
            # Classification score threshold
            if obj.classification_score < 0.3:
                self.logger.debug(
                    f"Rejected object: classification score {obj.classification_score:.2f} < 0.3"
                )
                continue
            
            filtered.append(obj)
        
        return filtered
    
    def predict_positions(
        self,
        moving_object: MovingObject,
        prediction_times: List[Time],
    ) -> List[Tuple[float, float]]:
        """
        Predict positions of a moving object at given times.
        
        Args:
            moving_object: MovingObject instance
            prediction_times: List of times for prediction
            
        Returns:
            List of (ra, dec) positions in degrees
        """
        if not moving_object.detections:
            return []
        
        # Use first detection as reference
        ref_detection = moving_object.detections[0]
        ref_time = ref_detection.timestamp
        
        if not ref_time:
            return []
        
        predicted_positions = []
        
        for pred_time in prediction_times:
            dt_hours = (pred_time.unix - ref_time.unix) / 3600.0
            
            # Apply velocity to get predicted position
            pred_ra = ref_detection.ra + (moving_object.velocity[0] / 3600.0) * dt_hours / np.cos(np.radians(ref_detection.dec))
            pred_dec = ref_detection.dec + (moving_object.velocity[1] / 3600.0) * dt_hours
            
            predicted_positions.append((pred_ra, pred_dec))
        
        return predicted_positions
    
    def compute_orbital_elements(
        self,
        moving_object: MovingObject,
    ) -> Optional[Dict]:
        """
        Compute preliminary orbital elements (simplified).
        
        Args:
            moving_object: MovingObject instance
            
        Returns:
            Dictionary of orbital elements or None if computation fails
        """
        if len(moving_object.detections) < 3:
            return None
        
        try:
            # This is a simplified implementation
            # In practice, you'd use a proper orbit determination library
            
            detections = moving_object.detections
            
            # Calculate mean motion from velocity
            velocity_magnitude = np.sqrt(
                moving_object.velocity[0]**2 + moving_object.velocity[1]**2
            )
            
            # Very rough estimates (not accurate orbital mechanics)
            elements = {
                "mean_motion_deg_per_day": velocity_magnitude * 24 / 3600,  # deg/day
                "velocity_arcsec_per_hour": velocity_magnitude,
                "position_angle": np.degrees(np.arctan2(
                    moving_object.velocity[1], moving_object.velocity[0]
                )),
                "mean_magnitude": moving_object.magnitude_mean,
                "num_observations": len(detections),
                "arc_length_hours": (
                    detections[-1].timestamp.unix - detections[0].timestamp.unix
                ) / 3600.0 if detections[0].timestamp and detections[-1].timestamp else 0,
                "rms_residual_arcsec": moving_object.trajectory_rms,
            }
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Failed to compute orbital elements: {e}")
            return None
    
    def save_tracks(
        self,
        moving_objects: List[MovingObject],
        output_path: Path,
        format: str = "mpc",
    ):
        """
        Save asteroid tracks to file.
        
        Args:
            moving_objects: List of MovingObject instances
            output_path: Output file path
            format: Output format ('mpc', 'json', 'csv')
        """
        try:
            if format.lower() == "mpc":
                self._save_mpc_format(moving_objects, output_path)
            elif format.lower() == "json":
                self._save_json_format(moving_objects, output_path)
            elif format.lower() == "csv":
                self._save_csv_format(moving_objects, output_path)
            else:
                raise DetectionError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved {len(moving_objects)} tracks to {output_path}")
            
        except Exception as e:
            raise DetectionError(f"Failed to save tracks: {e}")
    
    def _save_mpc_format(self, moving_objects: List[MovingObject], output_path: Path):
        """Save in Minor Planet Center observation format."""
        with open(output_path, 'w') as f:
            for i, obj in enumerate(moving_objects):
                for j, det in enumerate(obj.detections):
                    # Simplified MPC format (80-column)
                    # This is a basic implementation - real MPC format is more complex
                    
                    if det.timestamp:
                        date_str = det.timestamp.datetime.strftime("%Y %m %d.%f")[:-3]
                    else:
                        date_str = "0000 00 00.000"
                    
                    ra_h = int(det.ra / 15)
                    ra_m = int((det.ra / 15 - ra_h) * 60)
                    ra_s = ((det.ra / 15 - ra_h) * 60 - ra_m) * 60
                    
                    dec_sign = '+' if det.dec >= 0 else '-'
                    dec_d = int(abs(det.dec))
                    dec_m = int((abs(det.dec) - dec_d) * 60)
                    dec_s = ((abs(det.dec) - dec_d) * 60 - dec_m) * 60
                    
                    line = f"AST{i:04d}  C{date_str} {ra_h:02d} {ra_m:02d} {ra_s:05.2f} {dec_sign}{dec_d:02d} {dec_m:02d} {dec_s:04.1f}         {det.magnitude:5.1f} G      T12"
                    
                    f.write(line + '\n')
    
    def _save_json_format(self, moving_objects: List[MovingObject], output_path: Path):
        """Save in JSON format."""
        import json
        
        data = []
        for i, obj in enumerate(moving_objects):
            obj_data = {
                "object_id": f"AST{i:04d}",
                "velocity_ra_arcsec_per_hour": obj.velocity[0],
                "velocity_dec_arcsec_per_hour": obj.velocity[1],
                "trajectory_rms_arcsec": obj.trajectory_rms,
                "magnitude_mean": obj.magnitude_mean,
                "magnitude_std": obj.magnitude_std,
                "classification_score": obj.classification_score,
                "num_detections": len(obj.detections),
                "detections": [
                    {
                        "ra": det.ra,
                        "dec": det.dec,
                        "magnitude": det.magnitude,
                        "timestamp": det.timestamp.iso if det.timestamp else None,
                        "x": det.x,
                        "y": det.y,
                    }
                    for det in obj.detections
                ]
            }
            data.append(obj_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_csv_format(self, moving_objects: List[MovingObject], output_path: Path):
        """Save in CSV format."""
        import pandas as pd
        
        rows = []
        for i, obj in enumerate(moving_objects):
            for det in obj.detections:
                row = {
                    "object_id": f"AST{i:04d}",
                    "ra": det.ra,
                    "dec": det.dec,
                    "magnitude": det.magnitude,
                    "timestamp": det.timestamp.iso if det.timestamp else None,
                    "velocity_ra": obj.velocity[0],
                    "velocity_dec": obj.velocity[1],
                    "trajectory_rms": obj.trajectory_rms,
                    "classification_score": obj.classification_score,
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)