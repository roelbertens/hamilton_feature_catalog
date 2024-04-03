import os
import shutil

from pyspark.sql import SparkSession
from pyspark_test import assert_pyspark_df_equal

from hamilton_feature_catalog.cache import cache_and_reload_nodes


def test_cache_and_reload_nodes(spark: SparkSession):
    """Test if the caching writes the result to disk and reloaded it"""
    caching_dir = ".tmp_cache"
    os.mkdir(caching_dir)

    sdf = spark.createDataFrame([[1, 2, 3]])
    result = {"sdf": sdf}
    loaded_nodes = cache_and_reload_nodes(
        nodes=result, caching_dir=caching_dir, spark=spark, is_local=True
    )
    loaded_sdf = loaded_nodes["sdf"]

    assert (
        loaded_sdf != sdf
    ), "The sdf should be reloaded so it should be a different object"
    assert_pyspark_df_equal(sdf, loaded_sdf)
    shutil.rmtree(caching_dir)
