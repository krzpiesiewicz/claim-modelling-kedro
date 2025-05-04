from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog

from claim_modelling_kedro.pipelines.utils.datasets import remove_dataset


_catolog = None


class MyCatalogHook:
    @hook_impl
    def before_node_run(self, node, inputs, catalog: DataCatalog, **kwargs):
        # Pass the catalog to the function or set it globally
        global _catalog
        _catalog = catalog


def get_catalog():
    return _catalog


class RemoveOutputDatasetsHook:
    @hook_impl
    def before_node_run(self, node, inputs, catalog, **kwargs):
        """
        This hook runs before the node execution and removes its output datasets.

        Args:
            node: The node object that is about to be executed.
            inputs: The inputs to the node.
            catalog: The Kedro DataCatalog object.
            kwargs: Other keyword arguments passed by Kedro.
        """
        # Get the node's output datasets (list of strings)
        output_datasets = node.outputs

        # Remove each output dataset from the catalog
        for output_name in output_datasets:
            remove_dataset(catalog, output_name)
