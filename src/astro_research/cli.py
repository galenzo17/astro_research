"""Command-line interface for the asteroid detection pipeline."""

from pathlib import Path
from typing import List, Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from astro_research import __version__, setup_logger
from astro_research.core.types import ProcessingConfig, Survey
from astro_research.download.manager import DownloadManager
from astro_research.preprocessing.processor import ImageProcessor
from astro_research.detection.detector import ObjectDetector
from astro_research.tracking.tracker import AsteroidTracker
from astro_research.visualization.plotter import AsteroidPlotter
from astro_research.visualization.reporter import DetectionReporter

app = typer.Typer(
    name="astro-detect",
    help="Asteroid detection pipeline for astronomical images",
    add_completion=False,
)

console = Console()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """Asteroid Detection Pipeline"""
    if version:
        typer.echo(f"astro-research {__version__}")
        raise typer.Exit()
    
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level)


@app.command()
def download(
    ra: float = typer.Option(..., "--ra", help="Right ascension in degrees"),
    dec: float = typer.Option(..., "--dec", help="Declination in degrees"),
    radius: float = typer.Option(0.5, "--radius", help="Search radius in degrees"),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)"),
    survey: str = typer.Option("ztf", "--survey", help="Survey source (ztf, atlas, panstarrs)"),
    output_dir: Path = typer.Option("./data", "--output", "-o", help="Output directory"),
    filters: Optional[List[str]] = typer.Option(None, "--filter", help="Filter bands"),
):
    """Download astronomical survey data."""
    try:
        survey_enum = Survey(survey.lower())
    except ValueError:
        rprint(f"[red]Error:[/red] Unknown survey '{survey}'. Available: ztf, atlas, panstarrs")
        raise typer.Exit(1)
    
    try:
        start_time = datetime.fromisoformat(start_date)
        end_time = datetime.fromisoformat(end_date)
    except ValueError:
        rprint("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
        raise typer.Exit(1)
    
    rprint(f"[blue]Downloading data from {survey.upper()}[/blue]")
    rprint(f"Position: RA={ra}°, Dec={dec}°, Radius={radius}°")
    rprint(f"Time range: {start_date} to {end_date}")
    
    manager = DownloadManager(output_dir)
    
    filter_dict = {survey_enum: filters} if filters else None
    
    results = manager.download_multi_survey(
        ra=ra,
        dec=dec,
        radius=radius,
        start_time=start_time,
        end_time=end_time,
        surveys=[survey_enum],
        filters=filter_dict,
    )
    
    total_images = sum(len(metadata_list) for metadata_list in results.values())
    rprint(f"[green]Downloaded {total_images} images[/green]")


@app.command()
def process(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Input directory with FITS files"),
    output_dir: Path = typer.Option("./output", "--output", "-o", help="Output directory"),
    detection_threshold: float = typer.Option(5.0, "--threshold", help="Detection threshold (sigma)"),
    min_detections: int = typer.Option(3, "--min-detections", help="Minimum detections for tracking"),
    align_images: bool = typer.Option(True, "--align/--no-align", help="Align images"),
    create_plots: bool = typer.Option(True, "--plots/--no-plots", help="Create visualization plots"),
    format: str = typer.Option("mpc", "--format", help="Output format (mpc, json, csv)"),
):
    """Process FITS images to detect asteroids."""
    if not input_dir.exists():
        rprint(f"[red]Error:[/red] Input directory {input_dir} does not exist")
        raise typer.Exit(1)
    
    # Find FITS files
    fits_patterns = ["*.fits", "*.fit", "*.fts"]
    fits_files = []
    for pattern in fits_patterns:
        fits_files.extend(input_dir.glob(pattern))
    
    if not fits_files:
        rprint(f"[red]Error:[/red] No FITS files found in {input_dir}")
        raise typer.Exit(1)
    
    rprint(f"[blue]Processing {len(fits_files)} FITS files[/blue]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up processing configuration
    config = ProcessingConfig(
        detection_threshold=detection_threshold,
        min_detections=min_detections,
    )
    
    # Process images
    processor = ImageProcessor()
    
    with console.status("[bold green]Processing images..."):
        images, metadata_list, processing_info = processor.process_image_sequence(
            fits_files,
            output_dir=output_dir / "processed",
            align_images=align_images,
        )
    
    rprint(f"[green]Processed {len(images)} images[/green]")
    
    # Detect sources
    detector = ObjectDetector(config)
    
    with console.status("[bold green]Detecting sources..."):
        detection_lists = detector.detect_sources_batch(images, metadata_list)
    
    total_detections = sum(len(dets) for dets in detection_lists)
    rprint(f"[green]Found {total_detections} source detections[/green]")
    
    # Track asteroids
    tracker = AsteroidTracker(config)
    
    with console.status("[bold green]Tracking moving objects..."):
        moving_objects = tracker.track_asteroids(detection_lists, metadata_list)
    
    rprint(f"[green]Detected {len(moving_objects)} moving objects[/green]")
    
    if moving_objects:
        # Display results table
        table = Table(title="Detected Moving Objects")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Detections", style="green")
        table.add_column("Velocity (arcsec/h)", style="yellow")
        table.add_column("Mean Mag", style="magenta")
        table.add_column("Score", style="blue")
        
        for i, obj in enumerate(moving_objects):
            vel_mag = (obj.velocity[0]**2 + obj.velocity[1]**2)**0.5
            table.add_row(
                f"AST{i+1:04d}",
                str(len(obj.detections)),
                f"{vel_mag:.1f}",
                f"{obj.magnitude_mean:.1f}",
                f"{obj.classification_score:.2f}",
            )
        
        console.print(table)
        
        # Save tracks
        output_path = output_dir / f"asteroid_tracks.{format}"
        tracker.save_tracks(moving_objects, output_path, format)
        rprint(f"[green]Saved tracks to {output_path}[/green]")
        
        # Generate report
        reporter = DetectionReporter()
        report_path = output_dir / "detection_report.txt"
        reporter.generate_summary_report(
            detection_lists, metadata_list, moving_objects, processing_info, report_path
        )
        rprint(f"[green]Generated report: {report_path}[/green]")
        
        # Create plots if requested
        if create_plots:
            plotter = AsteroidPlotter()
            
            # Sky plot
            sky_plot_path = output_dir / "sky_detections.html"
            plotter.plot_detections_sky(detection_lists, sky_plot_path, show_plot=False)
            
            # Track plot
            track_plot_path = output_dir / "asteroid_tracks.html"
            plotter.plot_asteroid_tracks(moving_objects, track_plot_path, show_plot=False)
            
            # Velocity plot
            vel_plot_path = output_dir / "velocity_analysis.html"
            plotter.plot_velocity_distribution(moving_objects, vel_plot_path, show_plot=False)
            
            rprint(f"[green]Created plots in {output_dir}[/green]")
    
    else:
        rprint("[yellow]No moving objects detected[/yellow]")


@app.command()
def survey(
    ra: float = typer.Option(..., "--ra", help="Right ascension in degrees"),
    dec: float = typer.Option(..., "--dec", help="Declination in degrees"),
    radius: float = typer.Option(0.5, "--radius", help="Search radius in degrees"),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)"),
    survey_source: str = typer.Option("ztf", "--survey", help="Survey source"),
    output_dir: Path = typer.Option("./pipeline_output", "--output", "-o", help="Output directory"),
    detection_threshold: float = typer.Option(5.0, "--threshold", help="Detection threshold"),
    min_detections: int = typer.Option(3, "--min-detections", help="Minimum detections"),
):
    """Download survey data and process it end-to-end."""
    rprint("[blue]Running full pipeline: download + process[/blue]")
    
    # Download data
    ctx = typer.Context(app)
    ctx.invoke(
        download,
        ra=ra,
        dec=dec,
        radius=radius,
        start_date=start_date,
        end_date=end_date,
        survey=survey_source,
        output_dir=output_dir / "data",
    )
    
    # Process downloaded data
    data_dir = output_dir / "data" / survey_source
    if data_dir.exists():
        ctx.invoke(
            process,
            input_dir=data_dir,
            output_dir=output_dir / "results",
            detection_threshold=detection_threshold,
            min_detections=min_detections,
        )
    else:
        rprint(f"[red]Error:[/red] No data downloaded to {data_dir}")


@app.command()
def monitor(
    config_file: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    output_dir: Path = typer.Option("./monitoring", "--output", "-o", help="Output directory"),
    interval: int = typer.Option(3600, "--interval", help="Check interval in seconds"),
):
    """Monitor for new observations (placeholder for real-time processing)."""
    rprint("[yellow]Real-time monitoring mode (placeholder)[/yellow]")
    rprint(f"Config: {config_file}")
    rprint(f"Output: {output_dir}")
    rprint(f"Check interval: {interval}s")
    rprint("[blue]This feature would implement real-time survey monitoring[/blue]")


@app.command()
def visualize(
    tracks_file: Path = typer.Option(..., "--input", "-i", help="Tracks file (JSON or CSV)"),
    output_dir: Path = typer.Option("./plots", "--output", "-o", help="Output directory"),
    plot_type: str = typer.Option("all", "--type", help="Plot type (all, sky, tracks, velocity)"),
):
    """Create visualizations from saved tracking results."""
    if not tracks_file.exists():
        rprint(f"[red]Error:[/red] Tracks file {tracks_file} does not exist")
        raise typer.Exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rprint(f"[blue]Creating visualizations from {tracks_file}[/blue]")
    rprint("[yellow]Note: This would load tracks and create plots[/yellow]")
    rprint(f"Plots will be saved to {output_dir}")


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: show, create, edit"),
    config_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Configuration file"),
):
    """Manage configuration files."""
    if action == "show":
        rprint("[blue]Default Configuration:[/blue]")
        config_obj = ProcessingConfig()
        
        table = Table(title="Processing Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description", style="yellow")
        
        table.add_row("min_detections", str(config_obj.min_detections), "Minimum detections for valid track")
        table.add_row("max_velocity", str(config_obj.max_velocity), "Maximum velocity (arcsec/sec)")
        table.add_row("detection_threshold", str(config_obj.detection_threshold), "Detection threshold (sigma)")
        table.add_row("alignment_tolerance", str(config_obj.alignment_tolerance), "Alignment tolerance (pixels)")
        table.add_row("tracking_radius", str(config_obj.tracking_radius), "Tracking radius (arcsec)")
        table.add_row("min_snr", str(config_obj.min_snr), "Minimum SNR")
        table.add_row("parallel_workers", str(config_obj.parallel_workers), "Parallel workers")
        
        console.print(table)
        
    elif action == "create":
        if config_file is None:
            config_file = Path("astro_config.yaml")
        
        rprint(f"[green]Creating default config: {config_file}[/green]")
        # This would create a YAML configuration file
        rprint("[yellow]Feature not yet implemented[/yellow]")
        
    else:
        rprint(f"[red]Error:[/red] Unknown action '{action}'. Use: show, create, edit")


if __name__ == "__main__":
    app()