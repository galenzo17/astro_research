"""Configuration management and settings for the asteroid detection pipeline."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from astro_research.core.exceptions import ConfigurationError
from astro_research.core.logger import get_logger
from astro_research.core.types import ProcessingConfig, Survey


class ConfigManager:
    """Manages configuration for the asteroid detection pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = get_logger("config.manager")
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        
        if config_path and config_path.exists():
            self.load_from_file(config_path)
        else:
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration."""
        self._config = {
            "processing": {
                "min_detections": 3,
                "max_velocity": 10.0,
                "detection_threshold": 5.0,
                "alignment_tolerance": 2.0,
                "tracking_radius": 10.0,
                "min_snr": 5.0,
                "use_ml_validation": False,
                "parallel_workers": 4,
            },
            "download": {
                "default_survey": "ztf",
                "data_directory": "./data",
                "cache_directory": "./data/.cache",
                "timeout_seconds": 60,
                "retry_attempts": 3,
            },
            "preprocessing": {
                "subtract_background": True,
                "align_images": True,
                "reference_frame_index": 0,
                "sigma_clipping": 3.0,
            },
            "detection": {
                "extractor": "sep",
                "min_area": 5,
                "deblend_nthresh": 32,
                "deblend_cont": 0.005,
                "clean_detections": True,
            },
            "tracking": {
                "linking_algorithm": "dbscan",
                "time_tolerance_hours": 1.0,
                "position_tolerance_arcsec": 10.0,
                "min_track_length": 3,
            },
            "output": {
                "default_format": "mpc",
                "save_plots": True,
                "plot_format": "html",
                "generate_report": True,
                "output_directory": "./output",
            },
            "logging": {
                "level": "INFO",
                "log_file": None,
                "console_output": True,
            },
        }
    
    def load_from_file(self, config_path: Path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config is None:
                self.logger.warning(f"Empty configuration file: {config_path}")
                self._load_defaults()
                return
            
            # Start with defaults and update with file config
            self._load_defaults()
            self._deep_update(self._config, file_config)
            
            self.config_path = config_path
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {config_path}: {e}"
            )
    
    def save_to_file(self, config_path: Optional[Path] = None):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Optional path to save configuration
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ConfigurationError("No configuration path specified")
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            self.config_path = config_path
            self.logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {config_path}: {e}"
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "processing.min_detections")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "processing.min_detections")
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_processing_config(self) -> ProcessingConfig:
        """
        Get ProcessingConfig object from configuration.
        
        Returns:
            ProcessingConfig instance
        """
        processing_cfg = self._config.get("processing", {})
        
        return ProcessingConfig(
            min_detections=processing_cfg.get("min_detections", 3),
            max_velocity=processing_cfg.get("max_velocity", 10.0),
            detection_threshold=processing_cfg.get("detection_threshold", 5.0),
            alignment_tolerance=processing_cfg.get("alignment_tolerance", 2.0),
            tracking_radius=processing_cfg.get("tracking_radius", 10.0),
            min_snr=processing_cfg.get("min_snr", 5.0),
            use_ml_validation=processing_cfg.get("use_ml_validation", False),
            parallel_workers=processing_cfg.get("parallel_workers", 4),
        )
    
    def get_surveys(self) -> list[Survey]:
        """Get list of enabled surveys."""
        enabled_surveys = self.get("download.enabled_surveys", ["ztf"])
        surveys = []
        
        for survey_name in enabled_surveys:
            try:
                surveys.append(Survey(survey_name.lower()))
            except ValueError:
                self.logger.warning(f"Unknown survey: {survey_name}")
        
        return surveys or [Survey.ZTF]  # Fallback to ZTF
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Validate processing parameters
        min_det = self.get("processing.min_detections")
        if not isinstance(min_det, int) or min_det < 1:
            issues.append("processing.min_detections must be a positive integer")
        
        max_vel = self.get("processing.max_velocity")
        if not isinstance(max_vel, (int, float)) or max_vel <= 0:
            issues.append("processing.max_velocity must be a positive number")
        
        det_thresh = self.get("processing.detection_threshold")
        if not isinstance(det_thresh, (int, float)) or det_thresh <= 0:
            issues.append("processing.detection_threshold must be a positive number")
        
        # Validate paths
        data_dir = self.get("download.data_directory")
        if data_dir and not Path(data_dir).parent.exists():
            issues.append(f"Data directory parent does not exist: {data_dir}")
        
        output_dir = self.get("output.output_directory")
        if output_dir and not Path(output_dir).parent.exists():
            issues.append(f"Output directory parent does not exist: {output_dir}")
        
        # Validate surveys
        enabled_surveys = self.get("download.enabled_surveys", ["ztf"])
        for survey_name in enabled_surveys:
            try:
                Survey(survey_name.lower())
            except ValueError:
                issues.append(f"Unknown survey: {survey_name}")
        
        return issues
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> dict:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path={self.config_path})"


def load_config(config_path: Path) -> ConfigManager:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


def save_config(config: ConfigManager, config_path: Path):
    """
    Save configuration to file.
    
    Args:
        config: ConfigManager instance
        config_path: Path to save configuration
    """
    config.save_to_file(config_path)


def create_default_config(config_path: Path) -> ConfigManager:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to save the configuration
        
    Returns:
        ConfigManager instance with defaults
    """
    manager = ConfigManager()
    manager.save_to_file(config_path)
    return manager