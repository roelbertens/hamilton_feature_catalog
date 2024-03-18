import importlib
import os
from typing import Any, Optional, Tuple

from hamilton import base, driver, graph_types
from hamilton.plugins.h_experiments.hook import _get_default_input


def check_required_inputs(
    node_names: list[str], dr: driver.Driver, inputs: list[Any] = []
) -> list[str]:
    """Check if all required inputs are passed"""
    for node_name in node_names:
        upstream_nodes = dr.what_is_upstream_of(node_name)
        for upstream_node in upstream_nodes:
            if (
                upstream_node.is_external_input
                and upstream_node.name not in inputs
                and not _get_default_input(upstream_node)
                and upstream_node.originating_functions
                is not None  # to exclude config nodes
            ):
                raise Exception(f"Required input '{upstream_node.name}' is missing.")


def get_driver(config: dict = {}) -> driver.Driver:
    config.update({"hamilton.enable_power_user_mode": True})
    return driver.Driver(
        config,
        *_get_modules(),
        adapter=base.SimplePythonGraphAdapter(base.DictResult()),
    )


def custom_style_function(
    *, node: graph_types.HamiltonNode, node_class: str
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Custom style function for the visualization.

    :param node: node that Hamilton is styling.
    :param node_class: class used to style the default visualization
    :return: a triple of (style, node_class, legend_name) where
        style: dictionary of graphviz attributes https://graphviz.org/docs/nodes/,
        node_class: class used to style the default visualization - what you provide will be applied on top. If you don't want it, pass back None.
        legend_name: text to display in the legend. Return `None` for no legend entry.
    """
    if node.tags.get("data_type") == "raw_data":
        style = ({"fillcolor": "#c1f5cf"}, node_class, "raw dataframe")
    elif node.tags.get("data_type") == "intermediate_data":
        style = ({"fillcolor": "#c1e0f5"}, node_class, "intermediate dataframe")
    elif node.tags.get("data_type") == "optional_data":
        style = ({"fillcolor": "#ebf5c1"}, node_class, "optional dataframe")
    elif node.tags.get("data_type") == "features":
        style = ({"fillcolor": "#eddaf7"}, node_class, "features dataframe")
    else:
        style = ({}, node_class, None)

    return style


def _get_modules() -> list:
    modules = {}
    for folder in ["raw", "intermediate", "features"]:
        module_names = [
            f.replace(".py", "") for f in os.listdir(folder) if f.endswith(".py")
        ]
        for module_name in module_names:
            module = importlib.import_module(f"{folder}.{module_name}")
            modules[module_name] = module
    return modules.values()
