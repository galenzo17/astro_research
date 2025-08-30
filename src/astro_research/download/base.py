"""Base class for survey data downloaders."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from astropy.coordinates import SkyCoord
from astropy.time import Time

from astro_research.core.exceptions import DownloadError
from astro_research.core.logger import get_logger
from astro_research.core.types import ImageMetadata, Survey


class SurveyDownloader(ABC):
    """Abstract base class for downloading data from astronomical surveys."""
    
    def __init__(
        self,
        survey: Survey,
        data_dir: Path,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the survey downloader.
        
        Args:
            survey: Survey identifier
            data_dir: Directory to store downloaded data
            cache_dir: Optional directory for caching
        """
        self.survey = survey
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.logger = get_logger(f"download.{survey.value}")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def search(
        self,
        coordinates: SkyCoord,
        radius: float,
        start_time: Time,
        end_time: Time,
        filters: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Search for observations in the survey catalog.
        
        Args:
            coordinates: Sky coordinates of the search center
            radius: Search radius in degrees
            start_time: Start of time range
            end_time: End of time range
            filters: Optional list of filter bands
            
        Returns:
            List of observation metadata dictionaries
        """
        pass
    
    @abstractmethod
    def download_images(
        self,
        observations: List[dict],
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Download FITS images for the given observations.
        
        Args:
            observations: List of observation metadata
            output_dir: Optional output directory
            
        Returns:
            List of paths to downloaded FITS files
        """
        pass
    
    @abstractmethod
    def parse_metadata(self, fits_path: Path) -> ImageMetadata:
        """
        Parse metadata from a FITS file.
        
        Args:
            fits_path: Path to FITS file
            
        Returns:
            ImageMetadata object
        """
        pass
    
    def download_field(
        self,
        ra: float,
        dec: float,
        radius: float,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[List[str]] = None,
    ) -> List[ImageMetadata]:
        """
        Download all images for a field within a time range.
        
        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            start_time: Start of observation period
            end_time: End of observation period
            filters: Optional filter bands
            
        Returns:
            List of ImageMetadata for downloaded images
        """
        try:
            coordinates = SkyCoord(ra=ra, dec=dec, unit="deg")
            t_start = Time(start_time)
            t_end = Time(end_time)
            
            self.logger.info(
                f"Searching {self.survey.value} for field at "
                f"RA={ra:.2f}, Dec={dec:.2f}, radius={radius:.2f} deg"
            )
            
            observations = self.search(
                coordinates=coordinates,
                radius=radius,
                start_time=t_start,
                end_time=t_end,
                filters=filters,
            )
            
            if not observations:
                self.logger.warning("No observations found matching criteria")
                return []
            
            self.logger.info(f"Found {len(observations)} observations")
            
            fits_paths = self.download_images(observations)
            
            metadata_list = []
            for path in fits_paths:
                try:
                    metadata = self.parse_metadata(path)
                    metadata_list.append(metadata)
                except Exception as e:
                    self.logger.error(f"Failed to parse metadata from {path}: {e}")
            
            return metadata_list
            
        except Exception as e:
            raise DownloadError(
                f"Failed to download field from {self.survey.value}",
                details={"error": str(e), "ra": ra, "dec": dec},
            )