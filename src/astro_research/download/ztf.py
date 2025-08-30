"""ZTF (Zwicky Transient Facility) data downloader."""

import time
from pathlib import Path
from typing import List, Optional

import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery.vizier import Vizier

from astro_research.core.exceptions import DownloadError
from astro_research.core.types import ImageMetadata, Survey
from astro_research.download.base import SurveyDownloader


class ZTFDownloader(SurveyDownloader):
    """Downloader for ZTF survey data."""
    
    BASE_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"
    
    def __init__(self, data_dir: Path, cache_dir: Optional[Path] = None):
        """Initialize ZTF downloader."""
        super().__init__(Survey.ZTF, data_dir, cache_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "astro_research/0.1.0"
        })
    
    def search(
        self,
        coordinates: SkyCoord,
        radius: float,
        start_time: Time,
        end_time: Time,
        filters: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Search ZTF observations using IRSA API.
        
        Args:
            coordinates: Sky coordinates
            radius: Search radius in degrees
            start_time: Start time
            end_time: End time
            filters: ZTF filters (zg, zr, zi)
            
        Returns:
            List of observation metadata
        """
        if filters is None:
            filters = ["zg", "zr", "zi"]
        
        observations = []
        
        try:
            params = {
                "POS": f"{coordinates.ra.deg},{coordinates.dec.deg}",
                "SIZE": radius,
                "TIME": f"{start_time.mjd}..{end_time.mjd}",
                "FORMAT": "json",
            }
            
            url = f"{self.BASE_URL}/query"
            
            for filter_band in filters:
                filter_params = {**params, "FILTER": filter_band}
                
                response = self.session.get(url, params=filter_params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                for obs in data.get("data", []):
                    observations.append({
                        "obsid": obs.get("obsid"),
                        "filter": filter_band,
                        "mjd": obs.get("obsmjd"),
                        "ra": obs.get("ra"),
                        "dec": obs.get("dec"),
                        "exptime": obs.get("exptime"),
                        "seeing": obs.get("seeing"),
                        "airmass": obs.get("airmass"),
                        "url": obs.get("url"),
                    })
                
                time.sleep(0.5)
            
            return observations
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to search ZTF catalog: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during ZTF search: {e}")
            return []
    
    def download_images(
        self,
        observations: List[dict],
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Download ZTF FITS images.
        
        Args:
            observations: List of observation metadata
            output_dir: Output directory for FITS files
            
        Returns:
            List of downloaded file paths
        """
        if output_dir is None:
            output_dir = self.data_dir / "ztf"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_paths = []
        
        for obs in observations:
            try:
                obsid = obs.get("obsid", "unknown")
                filter_band = obs.get("filter", "unknown")
                filename = f"ztf_{obsid}_{filter_band}.fits"
                file_path = output_dir / filename
                
                if file_path.exists():
                    self.logger.info(f"File already exists: {filename}")
                    downloaded_paths.append(file_path)
                    continue
                
                url = obs.get("url")
                if not url:
                    self.logger.warning(f"No URL for observation {obsid}")
                    continue
                
                self.logger.info(f"Downloading {filename}...")
                
                response = self.session.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_paths.append(file_path)
                self.logger.info(f"Downloaded {filename}")
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to download observation {obs.get('obsid')}: {e}")
        
        return downloaded_paths
    
    def parse_metadata(self, fits_path: Path) -> ImageMetadata:
        """
        Parse metadata from ZTF FITS file.
        
        Args:
            fits_path: Path to FITS file
            
        Returns:
            ImageMetadata object
        """
        try:
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                
                metadata = ImageMetadata(
                    file_path=fits_path,
                    observation_time=Time(header.get("MJD-OBS", 0), format="mjd"),
                    exposure_time=header.get("EXPTIME", 0),
                    filter_band=header.get("FILTER", ""),
                    ra_center=header.get("CRVAL1", 0),
                    dec_center=header.get("CRVAL2", 0),
                    field_of_view=(
                        header.get("NAXIS1", 0) * header.get("CD1_1", 0.00027) * 3600,
                        header.get("NAXIS2", 0) * header.get("CD2_2", 0.00027) * 3600,
                    ),
                    pixel_scale=abs(header.get("CD1_1", 0.00027)) * 3600,
                    survey=Survey.ZTF,
                    telescope="P48",
                    instrument=header.get("INSTRUME", "ZTF"),
                    airmass=header.get("AIRMASS"),
                    seeing=header.get("SEEING"),
                )
                
                return metadata
                
        except Exception as e:
            raise DownloadError(
                f"Failed to parse metadata from {fits_path}",
                details={"error": str(e), "file": str(fits_path)},
            )