"""
data_handlers/registry.py — Data loader plugin registry.

To add a new data source:
  1. Import its loader class here
  2. Add an entry to LOADER_REGISTRY

That's it. No changes to rag/, core/, or main.py required.
"""
from data_handlers.ct_loader import CorporateTurnaroundLoader
from data_handlers.enriched_loader import EnrichedLoader

LOADER_REGISTRY: dict = {
    "corporate_turnaround": CorporateTurnaroundLoader,
    "enriched": EnrichedLoader,

    # Future loaders — implement BaseDataLoader and add here:
    # "pdf":        PDFLoader,
    # "sql":        SQLLoader,
    # "api":        APILoader,
    # "json":       JSONLoader,
    # "confluence": ConfluenceLoader,
}


def get_loader(name: str) -> type:
    """
    Resolve a loader name to its class.
    Raises ValueError for unrecognized loader names.
    """
    loader_cls = LOADER_REGISTRY.get(name)
    if not loader_cls:
        raise ValueError(
            f"Unknown loader '{name}'. "
            f"Available loaders: {list(LOADER_REGISTRY.keys())}"
        )
    return loader_cls
