import inspect
from typing import Any, Dict, Optional, Tuple

from graphviz.graphs import Digraph
from hamilton import base, graph_types, telemetry
from hamilton.driver import Driver
from pyspark.sql import DataFrame as SparkDataFrame

from hamilton_feature_catalog.cache import (
    cache_and_reload_nodes,
    initialize_caching,
    load_nodes_from_cache,
)
from hamilton_feature_catalog.config import update_default_config
from hamilton_feature_catalog.utils import (
    MissingInput,
    UnsupportedAggregationLevel,
    UnsupportedFeatureName,
    check_scope,
    create_feature_overview,
    get_logger,
    get_supported_feature_names,
    import_and_list_modules,
)

LOGGER = get_logger()


def _get_driver(
    config: dict[str, Any],
    adapter: base.SimplePythonGraphAdapter = None,
) -> Driver:
    """Get the hamilton driver object

    Args:
        config: config which appends and/or overwrites the DEFAULT_CONFIG
        adapter: hamilton adapter

    Raises:
        Exception: if it is not able to includes modules from the specified folders

    Returns:
        Hamilton driver object
    """
    telemetry.disable_telemetry()
    modules = []
    for folder_name in config["folders_to_include"]:
        try:
            modules.extend(import_and_list_modules(folder_name))
        except:
            message = f"Not able to include folder: {folder_name}"
            LOGGER.warning(message)
            raise Exception(message)
    return Driver(config, *modules, adapter=adapter)


def _get_nodes_of_interest(
    feature_names: list[str], aggregation_level: Optional[str] = None
) -> list[str]:
    """Return a list of unique nodes that needs to be computed given the specified feature_names

    Args:
        feature_names: names of the features/column you are interested in
        aggregation_level: when specified also do a check and throw error when features don't support this aggregation_level
    """
    feature_overview = create_feature_overview()

    supported_features = get_supported_feature_names()
    for feature_name in feature_names:
        if feature_name not in supported_features:
            error_message = f"Error: feature name '{feature_name}' not in supported features: {supported_features}."
            LOGGER.error(error_message)
            raise UnsupportedFeatureName(error_message)

    nodes_of_interest = set()
    for i in range(len(feature_overview)):
        feature_name = feature_overview.at[i, "name"]
        if feature_name in feature_names:
            supported_levels = feature_overview.at[i, "aggregation levels"]
            if (
                aggregation_level is not None
                and aggregation_level not in supported_levels
            ):
                message = f"Error: feature '{feature_name}' supports levels {supported_levels} but not '{aggregation_level}'.\nTIP: use 'compute_nodes' when you want to compute results that cannot all be joined together."
                LOGGER.error(message)
                raise UnsupportedAggregationLevel(message)
            nodes_of_interest.add(feature_overview.at[i, "module"])
    return list(nodes_of_interest)


def _is_input_node_with_default(node: graph_types.HamiltonNode) -> bool:
    """Return True if the input node has a default value specified

    NOTE: also return True if the default is None
    """
    if (
        node.is_external_input and node.originating_functions is not None
    ):  # exclude config nodes
        origin_function = node.originating_functions[0]
        param = inspect.signature(origin_function).parameters[node.name]
        return False if param.default is inspect._empty else True
    return False


def _check_required_inputs(
    driver: Driver,
    node_names: list[str],
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Verify if all required inputs are passed

    NOTE: inputs that have a default value are not required
    """
    for node_name in node_names:
        for upstream_node in driver.what_is_upstream_of(node_name):
            if (
                upstream_node.is_external_input
                and upstream_node.name not in inputs
                and upstream_node.name not in config
                and not _is_input_node_with_default(upstream_node)
            ):
                message = f"Input with name '{upstream_node.name}' is missing."
                LOGGER.error(message)
                raise MissingInput(message)


def _custom_style_function(
    *, node: graph_types.HamiltonNode, node_class: str
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Custom style function for the visualization.

    Args:
        node: node that Hamilton is styling.
        node_class: class used to style the default visualization

    Returns:
        A triple of (style, node_class, legend_name) where
            style: dictionary of graphviz attributes https://graphviz.org/docs/nodes/,
            node_class: class used to style the default visualization - what you provide will be applied on top. If you don't want it, pass back None.
            legend_name: text to display in the legend. Return `None` for no legend entry.
    """
    if node.tags.get("data_type") == "raw":
        style = ({"fillcolor": "#c1f5cf"}, node_class, "raw dataframe")
    elif node.tags.get("data_type") == "intermediate":
        style = ({"fillcolor": "#c1e0f5"}, node_class, "intermediate dataframe")
    elif node.tags.get("data_type") == "optional":
        style = ({"fillcolor": "#ebf5c1"}, node_class, "optional dataframe")
    elif node.tags.get("data_type") == "features":
        style = ({"fillcolor": "#eddaf7"}, node_class, "features dataframe")
    else:
        style = ({}, node_class, None)

    return style


def display_nodes(
    node_names: Optional[list[str]] = None,
    show_schema: bool = False,
    deduplicate_inputs: bool = True,
    output_file_path: str = None,
    config: dict[str, Any] = {},
    **kwargs,
) -> Digraph:
    """Show the lineage (upstream) of all/selected nodes

    Args:
        node_names: the nodes to include. Show all when None.
        show_schema: if True, show the schema of all (intermediate) dataframes in the diagram (see hamilton docs)
        deduplicate_inputs: if True, try to deduplicate the input nodes (see hamilton docs)
        output_file_path: path to store the DAG. Pass in None to not save.
        config: hamilton config

    Returns:
        A graphviz.graphs.Digraph object
    """
    config = update_default_config(config)
    driver = _get_driver(config=config)
    if node_names is None:
        graph = driver.display_all_functions(
            deduplicate_inputs=deduplicate_inputs,
            show_schema=show_schema,
            custom_style_function=_custom_style_function,
            **kwargs,
        )
    else:
        graph = driver.display_upstream_of(
            *node_names,
            deduplicate_inputs=deduplicate_inputs,
            show_schema=show_schema,
            custom_style_function=_custom_style_function,
            **kwargs,
        )

    if output_file_path is not None:
        render_kwargs = {"view": False, "format": "png"}
        graph.render(output_file_path, **render_kwargs)

    return graph


def compute_features(
    feature_names: list[str],
    inputs: dict[str, Any],
    overrides: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = {},
) -> SparkDataFrame:
    """Compute the specified features.

    Args:
        feature_names: the features to compute
        inputs: dict containing inputs to the DAG such as spark and aggregation_level
        overrides: nodes to overwrite (see hamilton docs)
        config: hamilton config

    Returns:
        A df with aggregation_level and the feature_names as columns
    """
    config = update_default_config(config)
    aggregation_level = inputs["aggregation_level"]
    scope = inputs["scope"]
    check_scope(scope=scope, aggregation_level=aggregation_level, logger=LOGGER)

    nodes_of_interest = _get_nodes_of_interest(
        feature_names=feature_names, aggregation_level=aggregation_level
    )
    feature_nodes = compute_nodes(
        node_names=nodes_of_interest,
        inputs=inputs,
        overrides=overrides,
        config=config,
    )

    def _combine_features(feature_nodes: Dict[str, SparkDataFrame]) -> SparkDataFrame:
        LOGGER.info("Combining all feature dateframes in one dataframe")
        result = scope
        for df in feature_nodes.values():
            result = result.join(df, on=aggregation_level, how="left")

        if feature_names != []:
            return result.select(aggregation_level, *feature_names)
        return result

    return _combine_features(feature_nodes=feature_nodes)


def compute_nodes(
    node_names: list[str],
    inputs: dict[str, Any],
    overrides: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = {},
) -> dict[str, SparkDataFrame]:
    """Compute any node(s) of interest.

    NOTE: it returns all nodes as separate dataframes which you might needs to join together yourself.

    Args:
        node_names: the nodes to compute
        inputs: dict containing inputs to the DAG such as spark and aggregation_level
        overrides: nodes to overwrite (see hamilton docs)
        config: hamilton config

    Returns:
        A dict of dataframes with the node names as keys
    """
    config = update_default_config(config)
    LOGGER.info(
        f"Computing nodes: {node_names}, with inputs: {inputs}, config: {config}, and overrides: {overrides}."
    )
    adapter = base.SimplePythonGraphAdapter(base.DictResult())
    driver = _get_driver(config=config, adapter=adapter)
    _check_required_inputs(
        driver=driver, node_names=node_names, inputs=inputs, config=config
    )

    spark = inputs["spark"]
    loaded_nodes = load_nodes_from_cache(
        spark=spark, config=config, inputs=inputs, overrides=overrides
    )
    node_names__still_to_compute = list(set(node_names) - set(loaded_nodes.keys()))

    if len(node_names__still_to_compute) > 0:
        computed_nodes = driver.execute(
            final_vars=node_names__still_to_compute,
            inputs=inputs,
            overrides=overrides,
        )
        LOGGER.info(f"Computed the following nodes: {computed_nodes.keys()}")

        if config["cache_enabled"]:
            if caching_dir := initialize_caching(
                spark=spark, config=config, inputs=inputs, overrides=overrides
            ):
                computed_nodes = cache_and_reload_nodes(
                    nodes=computed_nodes, caching_dir=caching_dir, spark=spark
                )
    else:
        computed_nodes = {}

    result = computed_nodes | loaded_nodes

    return result
