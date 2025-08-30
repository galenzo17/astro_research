# Asteroid Detection Pipeline

Automated asteroid detection in astronomical images using blink comparison technique.

## Overview

This pipeline automates the detection of asteroids in astronomical images by comparing multiple exposures of the same stellar field taken at different times. Asteroids move relatively quickly against the background of fixed stars, appearing in different positions between shots.

## Features

- **Multi-frame tracking**: Detect moving objects across multiple exposures
- **Survey data integration**: Download and process images from ZTF, ATLAS, Pan-STARRS
- **Astrometric calibration**: Precise coordinate system alignment
- **Source detection**: Using SEP (Source Extractor for Python)
- **MPC validation**: Cross-reference with Minor Planet Center catalogs
- **Real-time processing**: Stream processing for large data volumes
- **ML-powered validation**: Optional machine learning to reduce false positives

## Technical Stack

- **Python 3.11** with uv package manager
- **Astropy**: FITS handling and celestial coordinates
- **Photutils/SEP**: Source detection
- **OpenCV**: Image processing
- **Typer**: CLI interface
- **FastAPI**: Optional REST API
- **Plotly**: Interactive visualizations

## Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/astro_research.git
cd astro_research

# Install dependencies
uv sync
```

## Usage

```bash
# Basic detection on FITS images
uv run astro-detect process --input path/to/fits/

# Download and process survey data
uv run astro-detect survey --source ztf --date 2024-01-01

# Real-time monitoring mode
uv run astro-detect monitor --observatory las-campanas
```

## Architecture

```
astro_research/
├── src/
│   ├── download/      # Data acquisition from surveys
│   ├── preprocessing/ # Image calibration and alignment
│   ├── detection/     # Source extraction and identification
│   ├── tracking/      # Multi-frame object tracking
│   ├── validation/    # MPC catalog cross-reference
│   └── visualization/ # Output generation and plotting
├── tests/
├── config/
└── data/
```

## Output Formats

- **MPC Reports**: Standard Minor Planet Center format
- **JSON**: Structured detection data
- **Interactive HTML**: Plotly visualizations with trajectories
- **FITS**: Annotated images with detections

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project uses data from:
- Zwicky Transient Facility (ZTF)
- Asteroid Terrestrial-impact Last Alert System (ATLAS)
- Panoramic Survey Telescope and Rapid Response System (Pan-STARRS)
- Minor Planet Center (MPC)