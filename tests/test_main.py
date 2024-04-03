import pytest
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from hamilton_feature_catalog.config import DEFAULT_CONFIG
from hamilton_feature_catalog.main import (
    _get_driver,
    _get_nodes_of_interest,
    compute_features,
    compute_nodes,
    display_nodes,
)
from hamilton_feature_catalog.utils import (
    UnsupportedFeatureName,
    create_feature_overview,
    get_feature_names_from_feature_groups_docstring,
    import_and_list_modules,
)


def test_get_driver():
    """Test if we can create the driver."""
    _get_driver(config=DEFAULT_CONFIG)


def test_display_nodes():
    display_nodes()


def test_get_nodes_of_interest():
    assert len(_get_nodes_of_interest([])) == 0


def test_get_nodes_of_interest__unsupported_feature():
    with pytest.raises(UnsupportedFeatureName):
        _get_nodes_of_interest(["this_feature_does_not_exist"])


def test_duplicate_feature_names():
    """Check for duplicate feature names"""
    feature_names = []
    for module in import_and_list_modules("feature_groups"):
        docstring = getattr(module, module.__name__).__doc__
        names = get_feature_names_from_feature_groups_docstring(docstring)
        overlap = set(feature_names) & set(names)
        assert len(overlap) == 0, f"Duplicate feature name(s): {overlap}"
        feature_names.extend(names)


def test_compute_features(
    spark: SparkSession,
    aggregation_level: str,
    scope: SparkDataFrame,
    number_of_decimals: int,
):
    """Test if we can create and join compatible feature groups.

    We test all features groups that support the specified aggregation level.
    """
    inputs = {
        "aggregation_level": aggregation_level,
        "scope": scope,
        "spark": spark,
        "number_of_decimals": number_of_decimals,
    }

    feature_overview = create_feature_overview()
    rel_id_features = [
        feature_overview.at[i, "name"]
        for i in range(len(feature_overview))
        if aggregation_level in feature_overview.at[i, "aggregation levels"]
    ]

    result = compute_features(
        feature_names=rel_id_features,
        inputs=inputs,
    )

    assert set(result.drop(aggregation_level).columns) == set(rel_id_features)


def test_compute_nodes(
    spark: SparkSession,
    aggregation_level: str,
    scope: SparkDataFrame,
    number_of_decimals: int,
):
    """Test if we can compute all nodes"""
    inputs = {
        "aggregation_level": aggregation_level,
        "scope": scope,
        "spark": spark,
        "number_of_decimals": number_of_decimals,
    }

    all_possible_outputs = [
        variable.name
        for variable in _get_driver(config=DEFAULT_CONFIG).list_available_variables()
        if not variable.is_external_input
    ]

    result = compute_nodes(
        node_names=all_possible_outputs,
        inputs=inputs,
    )

    assert set(result.keys()) == set(all_possible_outputs)
