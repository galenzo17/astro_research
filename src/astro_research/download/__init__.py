"""Data download module for astronomical surveys."""

from astro_research.download.base import SurveyDownloader
from astro_research.download.ztf import ZTFDownloader
from astro_research.download.manager import DownloadManager

__all__ = ["SurveyDownloader", "ZTFDownloader", "DownloadManager"]