"""Detection reporting and summary generation."""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from astro_research.core.logger import get_logger
from astro_research.core.types import Detection, MovingObject, ImageMetadata


class DetectionReporter:
    """Generate reports and summaries for asteroid detection results."""
    
    def __init__(self):
        """Initialize the detection reporter."""
        self.logger = get_logger("visualization.reporter")
    
    def generate_summary_report(
        self,
        detection_lists: List[List[Detection]],
        metadata_list: List[ImageMetadata],
        moving_objects: List[MovingObject],
        processing_info: Dict,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            detection_lists: List of detection lists
            metadata_list: List of image metadata
            moving_objects: List of moving objects
            processing_info: Processing information dictionary
            output_path: Optional path to save the report
            
        Returns:
            Report text as string
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "ASTEROID DETECTION PIPELINE REPORT",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ])
        
        # Processing Summary
        report_lines.extend([
            "PROCESSING SUMMARY",
            "-" * 20,
            f"Number of images processed: {len(metadata_list)}",
            f"Total processing time: {processing_info.get('total_time', 'N/A')}",
            f"Images aligned: {processing_info.get('aligned', False)}",
            f"Reference image index: {processing_info.get('reference_idx', 0)}",
            "",
        ])
        
        # Image Information
        if metadata_list:
            report_lines.extend([
                "IMAGE INFORMATION",
                "-" * 18,
            ])
            
            for i, metadata in enumerate(metadata_list):
                report_lines.append(
                    f"Image {i+1}: {metadata.file_path.name} "
                    f"({metadata.filter_band}, {metadata.exposure_time}s, "
                    f"{metadata.observation_time.iso if metadata.observation_time else 'N/A'})"
                )
            
            # Observation time span
            times = [m.observation_time for m in metadata_list if m.observation_time]
            if times:
                time_span = (max(times) - min(times)).to_value('hour')
                report_lines.append(f"Observation span: {time_span:.2f} hours")
            
            report_lines.append("")
        
        # Detection Statistics
        total_detections = sum(len(dets) for dets in detection_lists)
        report_lines.extend([
            "DETECTION STATISTICS",
            "-" * 20,
            f"Total detections: {total_detections}",
            f"Average detections per image: {total_detections/len(detection_lists):.1f}",
        ])
        
        if detection_lists:
            det_counts = [len(dets) for dets in detection_lists]
            report_lines.extend([
                f"Min detections per image: {min(det_counts)}",
                f"Max detections per image: {max(det_counts)}",
            ])
        
        report_lines.append("")
        
        # Moving Object Results
        report_lines.extend([
            "MOVING OBJECTS DETECTED",
            "-" * 24,
            f"Number of candidates: {len(moving_objects)}",
        ])
        
        if moving_objects:
            # Statistics
            velocities = [np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2) 
                         for obj in moving_objects]
            scores = [obj.classification_score for obj in moving_objects]
            
            report_lines.extend([
                f"Velocity range: {min(velocities):.1f} - {max(velocities):.1f} arcsec/hour",
                f"Mean velocity: {np.mean(velocities):.1f} arcsec/hour",
                f"Classification scores: {min(scores):.2f} - {max(scores):.2f}",
                f"Mean score: {np.mean(scores):.2f}",
                "",
                "INDIVIDUAL OBJECTS:",
                "-" * 19,
            ])
            
            for i, obj in enumerate(moving_objects):
                vel_mag = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
                vel_angle = np.degrees(np.arctan2(obj.velocity[1], obj.velocity[0]))
                
                report_lines.extend([
                    f"Object {i+1}:",
                    f"  Detections: {len(obj.detections)}",
                    f"  Velocity: {vel_mag:.1f} arcsec/hour at {vel_angle:.0f}°",
                    f"  Trajectory RMS: {obj.trajectory_rms:.2f} arcsec",
                    f"  Mean magnitude: {obj.magnitude_mean:.2f} ± {obj.magnitude_std:.2f}",
                    f"  Classification score: {obj.classification_score:.2f}",
                    "",
                ])
        else:
            report_lines.append("No moving objects detected.")
            report_lines.append("")
        
        # Quality Assessment
        report_lines.extend([
            "QUALITY ASSESSMENT",
            "-" * 18,
        ])
        
        if moving_objects:
            high_quality = sum(1 for obj in moving_objects if obj.classification_score > 0.7)
            medium_quality = sum(1 for obj in moving_objects 
                               if 0.4 <= obj.classification_score <= 0.7)
            low_quality = len(moving_objects) - high_quality - medium_quality
            
            report_lines.extend([
                f"High quality candidates (score > 0.7): {high_quality}",
                f"Medium quality candidates (0.4-0.7): {medium_quality}",
                f"Low quality candidates (< 0.4): {low_quality}",
            ])
        
        # Add recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 15,
        ])
        
        if not moving_objects:
            report_lines.extend([
                "• No asteroids detected - consider:",
                "  - Lowering detection threshold",
                "  - Increasing observation time span",
                "  - Checking image quality and alignment",
            ])
        elif len(moving_objects) < 3:
            report_lines.extend([
                "• Few candidates found - consider:",
                "  - Extending observation period",
                "  - Improving alignment accuracy",
                "  - Reducing minimum detection requirements",
            ])
        else:
            high_quality_count = sum(1 for obj in moving_objects if obj.classification_score > 0.7)
            if high_quality_count == 0:
                report_lines.extend([
                    "• No high-quality detections - consider:",
                    "  - Manual review of candidates",
                    "  - Cross-checking with known asteroid catalogs",
                    "  - Extending observation arc for confirmation",
                ])
            else:
                report_lines.extend([
                    f"• {high_quality_count} high-quality detections found",
                    "• Recommend follow-up observations for confirmation",
                    "• Submit to Minor Planet Center if confirmed",
                ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Saved summary report to {output_path}")
        
        return report_text
    
    def create_detection_table(
        self,
        moving_objects: List[MovingObject],
        output_path: Optional[Path] = None,
        format: str = "csv",
    ) -> pd.DataFrame:
        """
        Create a table of all detections.
        
        Args:
            moving_objects: List of moving objects
            output_path: Optional path to save the table
            format: Output format ('csv', 'html', 'json')
            
        Returns:
            Pandas DataFrame
        """
        rows = []
        
        for obj_idx, obj in enumerate(moving_objects):
            for det_idx, det in enumerate(obj.detections):
                row = {
                    'object_id': f'AST{obj_idx+1:04d}',
                    'detection_id': f'{obj_idx+1:04d}_{det_idx+1:02d}',
                    'ra_deg': det.ra,
                    'dec_deg': det.dec,
                    'x_pixel': det.x,
                    'y_pixel': det.y,
                    'magnitude': det.magnitude,
                    'magnitude_error': det.magnitude_err,
                    'flux': det.flux,
                    'flux_error': det.flux_err,
                    'fwhm_pixel': det.fwhm,
                    'ellipticity': det.ellipticity,
                    'timestamp': det.timestamp.iso if det.timestamp else None,
                    'image_id': det.image_id,
                    'object_velocity_ra': obj.velocity[0],
                    'object_velocity_dec': obj.velocity[1],
                    'object_trajectory_rms': obj.trajectory_rms,
                    'object_mean_magnitude': obj.magnitude_mean,
                    'object_magnitude_std': obj.magnitude_std,
                    'classification_score': obj.classification_score,
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if output_path:
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'html':
                df.to_html(output_path, index=False)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved detection table to {output_path}")
        
        return df
    
    def create_mpc_report(
        self,
        moving_objects: List[MovingObject],
        observatory_code: str = "T12",
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Create a Minor Planet Center (MPC) formatted report.
        
        Args:
            moving_objects: List of moving objects
            observatory_code: MPC observatory code
            output_path: Optional path to save the report
            
        Returns:
            MPC formatted report string
        """
        mpc_lines = []
        
        for obj_idx, obj in enumerate(moving_objects):
            for det in obj.detections:
                if not det.timestamp:
                    continue
                
                # Format observation date and time
                dt = det.timestamp.datetime
                date_str = dt.strftime("%Y %m %d.%f")[:-3]  # YYYY MM DD.ddddd
                
                # Format RA (HH MM SS.ss)
                ra_hours = det.ra / 15.0
                ra_h = int(ra_hours)
                ra_m = int((ra_hours - ra_h) * 60)
                ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
                
                # Format Dec (sDD MM SS.s)
                dec_sign = '+' if det.dec >= 0 else '-'
                dec_abs = abs(det.dec)
                dec_d = int(dec_abs)
                dec_m = int((dec_abs - dec_d) * 60)
                dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
                
                # MPC 80-column format
                line = (
                    f"AST{obj_idx+1:04d}  "  # Designation (12 chars)
                    f"C{date_str} "  # Discovery flag + date (17 chars)
                    f"{ra_h:02d} {ra_m:02d} {ra_s:05.2f} "  # RA (12 chars)
                    f"{dec_sign}{dec_d:02d} {dec_m:02d} {dec_s:04.1f}         "  # Dec (12 chars) + spaces
                    f"{det.magnitude:5.1f} G      "  # Magnitude + band (11 chars)
                    f"{observatory_code}"  # Observatory code (3 chars)
                ).ljust(80)
                
                mpc_lines.append(line)
        
        mpc_report = "\n".join(mpc_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(mpc_report)
            self.logger.info(f"Saved MPC report to {output_path}")
        
        return mpc_report
    
    def generate_html_report(
        self,
        detection_lists: List[List[Detection]],
        metadata_list: List[ImageMetadata],
        moving_objects: List[MovingObject],
        processing_info: Dict,
        output_path: Path,
        include_plots: bool = True,
    ):
        """
        Generate a comprehensive HTML report with embedded plots.
        
        Args:
            detection_lists: List of detection lists
            metadata_list: List of image metadata
            moving_objects: List of moving objects
            processing_info: Processing information dictionary
            output_path: Path to save the HTML report
            include_plots: Whether to include interactive plots
        """
        from astro_research.visualization.plotter import AsteroidPlotter
        
        html_content = []
        
        # HTML header
        html_content.extend([
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Asteroid Detection Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #2c3e50; }",
            "h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }",
            ".plot-container { margin: 20px 0; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Asteroid Detection Report</h1>",
            f"<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>",
        ])
        
        # Summary section
        summary_text = self.generate_summary_report(
            detection_lists, metadata_list, moving_objects, processing_info
        )
        
        html_content.extend([
            "<div class='summary'>",
            "<h2>Summary</h2>",
            f"<pre>{summary_text}</pre>",
            "</div>",
        ])
        
        # Add plots if requested
        if include_plots and moving_objects:
            plotter = AsteroidPlotter()
            
            # Save plots as temporary HTML files and embed them
            temp_dir = output_path.parent / "temp_plots"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Sky plot
                sky_plot_path = temp_dir / "sky_plot.html"
                sky_fig = plotter.plot_detections_sky(
                    detection_lists, sky_plot_path, show_plot=False
                )
                
                # Track plot
                track_plot_path = temp_dir / "track_plot.html"
                track_fig = plotter.plot_asteroid_tracks(
                    moving_objects, track_plot_path, show_plot=False
                )
                
                # Velocity plot
                vel_plot_path = temp_dir / "velocity_plot.html"
                vel_fig = plotter.plot_velocity_distribution(
                    moving_objects, vel_plot_path, show_plot=False
                )
                
                # Embed plots
                html_content.extend([
                    "<h2>Visualizations</h2>",
                    "<div class='plot-container'>",
                    "<h3>Sky Detections</h3>",
                    sky_fig.to_html(include_plotlyjs='inline', div_id="sky_plot"),
                    "</div>",
                    "<div class='plot-container'>",
                    "<h3>Asteroid Tracks</h3>",
                    track_fig.to_html(include_plotlyjs='inline', div_id="track_plot"),
                    "</div>",
                    "<div class='plot-container'>",
                    "<h3>Velocity Analysis</h3>",
                    vel_fig.to_html(include_plotlyjs='inline', div_id="velocity_plot"),
                    "</div>",
                ])
                
            except Exception as e:
                self.logger.error(f"Failed to embed plots: {e}")
                html_content.append("<p>Error generating plots</p>")
        
        # Detection table
        if moving_objects:
            df = self.create_detection_table(moving_objects)
            html_content.extend([
                "<h2>Detection Table</h2>",
                df.to_html(classes='detection-table', table_id='detections'),
            ])
        
        # HTML footer
        html_content.extend([
            "</body>",
            "</html>",
        ])
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        self.logger.info(f"Generated HTML report: {output_path}")
        
        # Clean up temporary files
        try:
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp files: {e}")