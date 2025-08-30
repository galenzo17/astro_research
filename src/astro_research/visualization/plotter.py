"""Interactive plotting for asteroid detection results using Plotly."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from astropy.time import Time

from astro_research.core.logger import get_logger
from astro_research.core.types import Detection, MovingObject, ImageMetadata


class AsteroidPlotter:
    """Interactive plotting for asteroid detection and tracking results."""
    
    def __init__(self):
        """Initialize the asteroid plotter."""
        self.logger = get_logger("visualization.plotter")
    
    def plot_detections_sky(
        self,
        detection_lists: List[List[Detection]],
        output_path: Optional[Path] = None,
        show_plot: bool = True,
    ) -> go.Figure:
        """
        Plot all detections on a sky map.
        
        Args:
            detection_lists: List of detection lists
            output_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, detections in enumerate(detection_lists):
            if not detections:
                continue
            
            ras = [det.ra for det in detections]
            decs = [det.dec for det in detections]
            mags = [det.magnitude for det in detections]
            
            hover_text = [
                f"RA: {det.ra:.4f}°<br>"
                f"Dec: {det.dec:.4f}°<br>"
                f"Mag: {det.magnitude:.1f}<br>"
                f"Image: {det.image_id}<br>"
                f"Time: {det.timestamp.iso if det.timestamp else 'N/A'}"
                for det in detections
            ]
            
            fig.add_trace(go.Scatter(
                x=ras,
                y=decs,
                mode='markers',
                marker=dict(
                    size=[max(3, 20 - mag) for mag in mags],
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=1, color='black'),
                ),
                name=f'Frame {i+1} ({len(detections)} sources)',
                hovertext=hover_text,
                hoverinfo='text',
            ))
        
        fig.update_layout(
            title="Source Detections on Sky",
            xaxis_title="Right Ascension (deg)",
            yaxis_title="Declination (deg)",
            hovermode='closest',
            showlegend=True,
            width=800,
            height=600,
        )
        
        # Reverse RA axis to match astronomical convention
        fig.update_xaxes(autorange="reversed")
        
        if output_path:
            fig.write_html(output_path)
            self.logger.info(f"Saved sky plot to {output_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_asteroid_tracks(
        self,
        moving_objects: List[MovingObject],
        output_path: Optional[Path] = None,
        show_plot: bool = True,
    ) -> go.Figure:
        """
        Plot asteroid tracks with trajectories.
        
        Args:
            moving_objects: List of MovingObject instances
            output_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, obj in enumerate(moving_objects):
            detections = obj.detections
            if len(detections) < 2:
                continue
            
            ras = [det.ra for det in detections]
            decs = [det.dec for det in detections]
            times = [det.timestamp.iso if det.timestamp else f"T{j}" 
                    for j, det in enumerate(detections)]
            
            # Plot trajectory line
            fig.add_trace(go.Scatter(
                x=ras,
                y=decs,
                mode='lines',
                line=dict(
                    color=colors[i % len(colors)],
                    width=2,
                ),
                name=f'Track {i+1} (trajectory)',
                showlegend=False,
            ))
            
            # Plot detection points
            hover_text = [
                f"Object: {i+1}<br>"
                f"RA: {det.ra:.4f}°<br>"
                f"Dec: {det.dec:.4f}°<br>"
                f"Mag: {det.magnitude:.1f}<br>"
                f"Time: {time}<br>"
                f"Velocity: {np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2):.1f} \"/h<br>"
                f"Score: {obj.classification_score:.2f}"
                for det, time in zip(detections, times)
            ]
            
            fig.add_trace(go.Scatter(
                x=ras,
                y=decs,
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    symbol='star',
                    line=dict(width=2, color='black'),
                ),
                name=f'Asteroid {i+1} ({len(detections)} obs)',
                hovertext=hover_text,
                hoverinfo='text',
            ))
        
        fig.update_layout(
            title="Asteroid Tracks",
            xaxis_title="Right Ascension (deg)",
            yaxis_title="Declination (deg)",
            hovermode='closest',
            showlegend=True,
            width=800,
            height=600,
        )
        
        fig.update_xaxes(autorange="reversed")
        
        if output_path:
            fig.write_html(output_path)
            self.logger.info(f"Saved track plot to {output_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_velocity_distribution(
        self,
        moving_objects: List[MovingObject],
        output_path: Optional[Path] = None,
        show_plot: bool = True,
    ) -> go.Figure:
        """
        Plot velocity distribution of detected objects.
        
        Args:
            moving_objects: List of MovingObject instances
            output_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly Figure object
        """
        if not moving_objects:
            fig = go.Figure()
            fig.update_layout(title="No moving objects to plot")
            return fig
        
        velocities = []
        angles = []
        scores = []
        
        for obj in moving_objects:
            vel_mag = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
            vel_angle = np.degrees(np.arctan2(obj.velocity[1], obj.velocity[0]))
            
            velocities.append(vel_mag)
            angles.append(vel_angle)
            scores.append(obj.classification_score)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Velocity Distribution', 'Velocity vs Score',
                           'Direction Distribution', 'Velocity Vector Plot'),
            specs=[[{}, {}],
                   [{"type": "polar"}, {}]]
        )
        
        # Velocity histogram
        fig.add_trace(
            go.Histogram(x=velocities, nbinsx=20, name='Velocity'),
            row=1, col=1
        )
        
        # Velocity vs score scatter
        fig.add_trace(
            go.Scatter(
                x=velocities,
                y=scores,
                mode='markers',
                marker=dict(size=8, opacity=0.7),
                name='Vel vs Score'
            ),
            row=1, col=2
        )
        
        # Direction polar histogram
        fig.add_trace(
            go.Barpolar(
                r=[angles.count(a) for a in np.arange(-180, 180, 20)],
                theta=np.arange(-180, 180, 20),
                name='Direction'
            ),
            row=2, col=1
        )
        
        # Velocity vector plot
        fig.add_trace(
            go.Scatter(
                x=[obj.velocity[0] for obj in moving_objects],
                y=[obj.velocity[1] for obj in moving_objects],
                mode='markers',
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Classification Score")
                ),
                name='Velocity Vectors'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Velocity Analysis ({len(moving_objects)} objects)",
            height=800,
        )
        
        fig.update_xaxes(title_text="Velocity (arcsec/hour)", row=1, col=1)
        fig.update_xaxes(title_text="Velocity (arcsec/hour)", row=1, col=2)
        fig.update_xaxes(title_text="RA Velocity (arcsec/hour)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Classification Score", row=1, col=2)
        fig.update_yaxes(title_text="Dec Velocity (arcsec/hour)", row=2, col=2)
        
        if output_path:
            fig.write_html(output_path)
            self.logger.info(f"Saved velocity plot to {output_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_lightcurve(
        self,
        moving_object: MovingObject,
        output_path: Optional[Path] = None,
        show_plot: bool = True,
    ) -> go.Figure:
        """
        Plot lightcurve for a moving object.
        
        Args:
            moving_object: MovingObject instance
            output_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly Figure object
        """
        detections = moving_object.detections
        
        if not detections:
            fig = go.Figure()
            fig.update_layout(title="No detections to plot")
            return fig
        
        times = []
        magnitudes = []
        mag_errors = []
        
        for det in detections:
            if det.timestamp and det.magnitude < 50:  # Filter out bad magnitudes
                times.append(det.timestamp.unix)
                magnitudes.append(det.magnitude)
                mag_errors.append(det.magnitude_err)
        
        if not times:
            fig = go.Figure()
            fig.update_layout(title="No valid magnitude data to plot")
            return fig
        
        # Convert to relative time in hours
        t0 = min(times)
        rel_times = [(t - t0) / 3600 for t in times]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rel_times,
            y=magnitudes,
            error_y=dict(
                type='data',
                array=mag_errors,
                visible=True
            ),
            mode='markers+lines',
            marker=dict(size=8),
            name='Lightcurve'
        ))
        
        fig.update_layout(
            title=f"Lightcurve (Mean: {moving_object.magnitude_mean:.2f} ± {moving_object.magnitude_std:.2f})",
            xaxis_title="Time (hours from first observation)",
            yaxis_title="Magnitude",
            yaxis=dict(autorange="reversed"),  # Brighter = lower magnitude
        )
        
        if output_path:
            fig.write_html(output_path)
            self.logger.info(f"Saved lightcurve to {output_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def create_blink_animation(
        self,
        images: List[np.ndarray],
        metadata_list: List[ImageMetadata],
        moving_objects: Optional[List[MovingObject]] = None,
        output_path: Optional[Path] = None,
        show_plot: bool = True,
        frame_duration: int = 1000,  # milliseconds
    ) -> go.Figure:
        """
        Create a blink animation showing asteroid motion.
        
        Args:
            images: List of image arrays
            metadata_list: List of image metadata
            moving_objects: Optional list of moving objects to highlight
            output_path: Optional path to save the animation
            show_plot: Whether to display the animation
            frame_duration: Duration of each frame in milliseconds
            
        Returns:
            Plotly Figure object with animation
        """
        if not images or len(images) != len(metadata_list):
            fig = go.Figure()
            fig.update_layout(title="Invalid input for blink animation")
            return fig
        
        frames = []
        
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            # Normalize image for display
            img_norm = (image - np.percentile(image, 1)) / (np.percentile(image, 99) - np.percentile(image, 1))
            img_norm = np.clip(img_norm, 0, 1)
            
            # Create frame
            frame_data = [go.Heatmap(
                z=img_norm,
                colorscale='gray',
                showscale=False,
                hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>'
            )]
            
            # Add moving object markers if provided
            if moving_objects:
                for obj_idx, obj in enumerate(moving_objects):
                    obj_detections = [det for det in obj.detections 
                                    if det.image_id == metadata.file_path.stem or
                                       det.image_id == f"img_{i}"]
                    
                    if obj_detections:
                        det = obj_detections[0]  # Should be only one per frame
                        frame_data.append(go.Scatter(
                            x=[det.x],
                            y=[det.y],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='circle-open',
                                line=dict(width=3)
                            ),
                            name=f'Asteroid {obj_idx+1}',
                            showlegend=i == 0,  # Only show legend for first frame
                        ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=f'Frame {i+1}',
                layout=go.Layout(
                    title=f'Frame {i+1} - {metadata.observation_time.iso if metadata.observation_time else "N/A"}'
                )
            ))
        
        # Create initial figure with first frame
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Asteroid Blink Animation",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"frame": {"duration": frame_duration, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 50}}],
                            label="Play",
                            method="animate"
                        ),
                        dict(
                            args=[{"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate", "transition": {"duration": 0}}],
                            label="Pause",
                            method="animate"
                        )
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.011,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue={
                        "font": {"size": 20},
                        "prefix": "Frame:",
                        "visible": True,
                        "xanchor": "right"
                    },
                    transition={"duration": 300, "easing": "cubic-in-out"},
                    pad={"b": 10, "t": 50},
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [f"Frame {i+1}"],
                                {"frame": {"duration": 300, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 300}}
                            ],
                            label=f"Frame {i+1}",
                            method="animate"
                        ) for i in range(len(frames))
                    ]
                )
            ]
        )
        
        fig.update_yaxes(autorange="reversed")  # Match image coordinate system
        
        if output_path:
            fig.write_html(output_path)
            self.logger.info(f"Saved blink animation to {output_path}")
        
        if show_plot:
            fig.show()
        
        return fig