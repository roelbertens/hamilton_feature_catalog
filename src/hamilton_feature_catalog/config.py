from typing import Any

DEFAULT_CONFIG = {
    "folders_to_include": ["raw", "intermediate", "features"],
    "caching_base_dir": "feature_catalog_cache",  # directory to use for caching
    "hamilton.enable_power_user_mode": True,
    "cache_enabled": False,  # for caching on databricks
    "transactions_clean__with__is_special_customer": False,  # when True, transactions_clean will have an additional column 'is_special_customer'
}


def update_default_config(config: dict[str, Any]) -> dict:
    """Update the DEFAULT_CONFIG with the given config."""
    _config = DEFAULT_CONFIG.copy()
    _config.update(config)
    return _config
