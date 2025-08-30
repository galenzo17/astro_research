"""Download manager for coordinating multiple survey sources."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from astro_research.core.logger import get_logger
from astro_research.core.types import ImageMetadata, Survey
from astro_research.download.base import SurveyDownloader
from astro_research.download.ztf import ZTFDownloader


class DownloadManager:
    """Manages downloads from multiple astronomical surveys."""
    
    def __init__(self, data_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize the download manager.
        
        Args:
            data_dir: Root directory for downloaded data
            cache_dir: Optional cache directory
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.logger = get_logger("download.manager")
        
        self.downloaders: Dict[Survey, SurveyDownloader] = {}
        self._register_downloaders()
    
    def _register_downloaders(self):
        """Register available survey downloaders."""
        self.downloaders[Survey.ZTF] = ZTFDownloader(
            self.data_dir / "ztf",
            self.cache_dir / "ztf"
        )
    
    def download_multi_survey(
        self,
        ra: float,
        dec: float,
        radius: float,
        start_time: datetime,
        end_time: datetime,
        surveys: Optional[List[Survey]] = None,
        filters: Optional[Dict[Survey, List[str]]] = None,
    ) -> Dict[Survey, List[ImageMetadata]]:
        """
        Download data from multiple surveys.
        
        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            start_time: Start of observation period
            end_time: End of observation period
            surveys: List of surveys to query
            filters: Dictionary of filters per survey
            
        Returns:
            Dictionary mapping surveys to lists of ImageMetadata
        """
        if surveys is None:
            surveys = list(self.downloaders.keys())
        
        if filters is None:
            filters = {}
        
        results = {}
        
        for survey in surveys:
            if survey not in self.downloaders:
                self.logger.warning(f"No downloader available for {survey.value}")
                continue
            
            self.logger.info(f"Downloading from {survey.value}")
            
            try:
                downloader = self.downloaders[survey]
                survey_filters = filters.get(survey)
                
                metadata_list = downloader.download_field(
                    ra=ra,
                    dec=dec,
                    radius=radius,
                    start_time=start_time,
                    end_time=end_time,
                    filters=survey_filters,
                )
                
                results[survey] = metadata_list
                
                self.logger.info(
                    f"Downloaded {len(metadata_list)} images from {survey.value}"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to download from {survey.value}: {e}")
                results[survey] = []
        
        return results
    
    def get_available_surveys(self) -> List[Survey]:
        """Get list of available surveys."""
        return list(self.downloaders.keys())