from hamilton_feature_catalog.config import DEFAULT_CONFIG
from hamilton_feature_catalog.main import _get_driver
from hamilton_feature_catalog.utils import import_and_list_modules


def test_all_nodes_must_have_data_type_tag():
    """All nodes should be tagged with a data_type, e.g. @tag(data_type="raw")"""
    available_types = DEFAULT_CONFIG["folders_to_include"]
    variables = _get_driver(config=DEFAULT_CONFIG).list_available_variables()
    for v in variables:
        if not v.is_external_input:
            type = v.tags.get("data_type")
            assert (
                type in available_types
            ), f"Node {v.name} is not tagged with data_type or the used type '{type}' is not one of: {available_types}."


def test_if_module_name_matches_function_name():
    """ "In each intermediate data module we expect a function with the same name as the module name"""

    for module in import_and_list_modules("feature_group") + import_and_list_modules(
        "intermediate"
    ):
        if module.__name__ == "__init__":
            continue
        assert module.__name__ in dir(
            module
        ), f"Function with name '{module.__name__}' missing in module {module.__name__}."


def test_if_module_contains_schema_variable():
    """Each module with name x.py is expected to define the schema of the resulting df as X_SCHEMA"""
    for module in import_and_list_modules("feature_group") + import_and_list_modules(
        "intermediate"
    ):
        if module.__name__ == "__init__":
            continue
        assert hasattr(
            module, f"{module.__name__.upper()}__SCHEMA"
        ), f"Module {module.__name__} has not defined {module.__name__.upper()}__SCHEMA"


def test_if_module_contains_supported_levels_variable():
    """Each feature group module with name x.py is expected to define the schema of the resulting df as X_SCHEMA"""
    for module in import_and_list_modules("feature_group"):
        assert hasattr(
            module, f"{module.__name__.upper()}__SUPPORTED_LEVELS"
        ), f"Module {module.__name__} has not defined {module.__name__.upper()}__SUPPORTED_LEVELS"
