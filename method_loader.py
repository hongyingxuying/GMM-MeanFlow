from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module_from_file(module_name: str, file_name: str, base_dir=None):
    """Load a Python module from a script filename that may contain hyphens."""
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    path = root / file_name
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name!r} from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_attr_from_file(module_name: str, file_name: str, attr_name: str, base_dir=None):
    module = load_module_from_file(module_name, file_name, base_dir=base_dir)
    return getattr(module, attr_name)
