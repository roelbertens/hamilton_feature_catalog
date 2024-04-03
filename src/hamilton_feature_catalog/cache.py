import hashlib
import importlib.metadata
import os
from typing import Any, Optional

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from hamilton_feature_catalog.config import update_default_config
from hamilton_feature_catalog.utils import get_dbutils, get_logger

LOGGER = get_logger()


def _get_run_id_hash(
    config: dict[str, Any], inputs: dict[str, Any], overrides: dict[str, Any]
) -> str:
    """Get a string that is unique with respect to the code, config, inputs and overrides.
    This means that when this string is the same we expect the same outputs if we compute nodes.

    NOTE: this is NEVER for production

    WARNING: when the 'inputs' or 'overrides' dict contains e.g. a spark df the content of this df doesn't change the hash,
    but it might actually influence the outputs.
    """
    version = importlib.metadata.version("hamilton_feature_catalog")
    run_id_string = version + str(config) + str(inputs) + str(overrides)
    return hashlib.sha256(run_id_string.encode()).hexdigest()


def _get_caching_dir(
    config: dict[str, Any], inputs: dict[str, Any], overrides: dict[str, Any]
) -> str:
    run_id_hash = _get_run_id_hash(config=config, inputs=inputs, overrides=overrides)
    return os.path.join(config["caching_base_dir"], run_id_hash)


def load_nodes_from_cache(
    spark: SparkSession,
    config: dict[str, Any],
    inputs: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, SparkDataFrame]:
    """If caching is enabled, try to load cached nodes/dataframes

    Args:
        spark: spark session
        config: hamilton config
        inputs: dict containing inputs to the DAG such as spark and aggregation_level
        overrides: nodes to overwrite (see hamilton docs)

    Returns:
        The loaded dataframes in a dict with name as key and df as value
    """
    if config["cache_enabled"]:
        LOGGER.info("Trying to load nodes/dataframes from cache")

        dbutils = get_dbutils(spark)
        if dbutils is None:
            message = "Could not get dbutils."
            LOGGER.error(message)
            raise Exception(message)
        else:
            caching_dir = _get_caching_dir(
                config=config, inputs=inputs, overrides=overrides
            )
            LOGGER.info(f"Caching directory: {caching_dir}")
            nodes = {}
            try:
                files = dbutils.fs.ls(caching_dir)
            except:
                files = []
                LOGGER.info("Caching directory doesn't exist yet")

            for file_info in files:
                node_name = file_info.name.rstrip("/")
                LOGGER.info(f"Trying to load node {node_name} from {file_info.path}")
                try:
                    nodes[node_name] = spark.read.parquet(file_info.path)
                    LOGGER.info(f"Successfully loaded {node_name}")
                except:
                    LOGGER.warning(f"Failed to load node {node_name}")

            return nodes
    else:
        LOGGER.info("Caching disabled")
        return {}


def initialize_caching(
    spark: SparkSession,
    config: dict[str, Any],
    inputs: dict[str, Any],
    overrides: dict[str, Any],
) -> Optional[str]:
    """If caching is possible it returns the caching dir, otherwise returns None"""
    dbutils = get_dbutils(spark)
    if dbutils is None:
        LOGGER.warning("Could not get dbutils, so caching will be disabled.")
    else:
        caching_dir = _get_caching_dir(
            config=config, inputs=inputs, overrides=overrides
        )
        try:
            if dbutils.fs.mkdirs(caching_dir):
                LOGGER.info(
                    f"Successfully initialized caching directory: {caching_dir}"
                )
                return caching_dir
            else:
                LOGGER.warning(f"Could not create caching dir '{caching_dir}'")
        except:
            LOGGER.warning(f"Caching dir '{caching_dir}' doesn't seem to be accessible")
    return None


def cache_and_reload_nodes(
    nodes: dict[str, SparkDataFrame],
    caching_dir: str,
    spark: SparkSession,
    is_local: bool = False,
) -> dict[str, SparkDataFrame]:
    """Cache the resulting nodes (dataframes) in the caching_dir and return the reloaded cached nodes to avoid recomputation

    Args:
        nodes: the nodes/dataframes to cache
        caching_dir: the path to store the results
        spark: spark session
        is_local: default is False. Used for testing; if True then don't prepend dbfs:/ to path

    Returns:
        The reloaded cached nodes
    """
    for df_name in nodes:
        df_path = os.path.join(caching_dir, df_name)
        try:
            nodes[df_name].write.mode("overwrite").parquet(df_path)
            LOGGER.info(f"Successfully written {df_name} to {df_path}.")
        except:
            LOGGER.warning(f"Failed to cache {df_name} to {df_path}.")

        try:
            if not is_local:
                df_path = "dbfs:/" + df_path
            nodes[df_name] = spark.read.parquet(df_path)
            LOGGER.info(
                f"Loaded {df_name} from {df_path} to avoid recomputation later on."
            )
        except:
            LOGGER.warning(f"Failed to load {df_name} from {df_path} after caching it.")

    return nodes
