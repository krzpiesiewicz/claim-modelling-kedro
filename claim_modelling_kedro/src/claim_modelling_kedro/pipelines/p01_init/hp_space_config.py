import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import yaml

logger = logging.getLogger(__name__)


class HyperOptSpaceConfigError(Exception):
    """Exception raised for errors in the hyperparameter space configuration."""
    pass


@dataclass
class UniformSpace:
    low: float
    high: float

    def __post_init__(self):
        if not (isinstance(self.low, (int, float)) and isinstance(self.high, (int, float))):
            raise HyperOptSpaceConfigError("UniformSpace: 'low' and 'high' must be numeric.")
        if self.low >= self.high:
            raise HyperOptSpaceConfigError("UniformSpace: 'low' must be less than 'high'.")


@dataclass
class QUniformSpace:
    low: float
    high: float
    q: float

    def __post_init__(self):
        if not (isinstance(self.low, (int, float)) and isinstance(self.high, (int, float)) and isinstance(self.q, (int, float))):
            raise HyperOptSpaceConfigError("QUniformSpace: 'low', 'high', and 'q' must be numeric.")
        if self.low >= self.high:
            raise HyperOptSpaceConfigError("QUniformSpace: 'low' must be less than 'high'.")
        if self.q <= 0:
            raise HyperOptSpaceConfigError("QUniformSpace: 'q' must be positive.")


@dataclass
class LogUniformSpace:
    low: float
    high: float

    def __post_init__(self):
        if not (isinstance(self.low, (int, float)) and isinstance(self.high, (int, float))):
            raise HyperOptSpaceConfigError("LogUniformSpace: 'low' and 'high' must be numeric.")
        if self.low <= 0 or self.high <= 0:
            raise HyperOptSpaceConfigError("LogUniformSpace: 'low' and 'high' must be positive.")
        if self.low >= self.high:
            raise HyperOptSpaceConfigError("LogUniformSpace: 'low' must be less than 'high'.")


@dataclass
class ChoiceSpace:
    values: List[Any]

    def __post_init__(self):
        if not isinstance(self.values, list) or len(self.values) == 0:
            raise HyperOptSpaceConfigError("ChoiceSpace: 'values' must be a non-empty list.")


@dataclass
class ConstSpace:
    value: Any


HyperParamSpaceType = Union[UniformSpace, QUniformSpace, LogUniformSpace, ChoiceSpace, ConstSpace]


@dataclass
class HyperOptSpaceConfig:
    """
    Parses and validates a hyperparameter search space configuration dictionary for a model.

    Attributes:
        space (Dict[str, HyperParamSpaceType]):
            Dictionary mapping hyperparameter names to their search space definitions (dataclasses).
        excluded_params (List[str]):
            List of parameter names to exclude from the search space.

    Raises:
        HyperOptSpaceConfigError: If the configuration is invalid or contains unknown types.

    Example usage:
        params_space = HyperOptSpaceConfig(params["params_space"][model_name], excluded_params=[...])
        print(params_space.space["learning_rate"])  # -> UniformSpace(low=0.001, high=0.5)
    """
    space: Dict[str, HyperParamSpaceType] = field(default_factory=dict)
    excluded_params: List[str] = field(default_factory=list)

    def __init__(self, params: Dict[str, Any], excluded_params: Optional[List[str]] = None):
        self.space = {}
        self.excluded_params = excluded_params or []
        for param, conf in params.items():
            if param in self.excluded_params:
                continue
            if isinstance(conf, dict) and "type" in conf:
                t = conf["type"]
                if t == "uniform":
                    self.space[param] = UniformSpace(conf["low"], conf["high"])
                elif t == "quniform":
                    self.space[param] = QUniformSpace(conf["low"], conf["high"], conf["q"])
                elif t == "loguniform":
                    self.space[param] = LogUniformSpace(conf["low"], conf["high"])
                elif t == "choice":
                    self.space[param] = ChoiceSpace(conf["values"])
                else:
                    raise HyperOptSpaceConfigError(f"Unknown space type: {t} for parameter {param}")
            else:
                # Constant value (e.g. early_stopping_rounds: 0)
                self.space[param] = ConstSpace(conf)
        logger.debug(f"Hyperparameter search space: {self.space}")


def build_hyperopt_space(config: HyperOptSpaceConfig) -> Dict[str, Any]:
    """
    Build a Hyperopt search space from a HyperOptSpaceConfig object.

    Args:
        config (HyperOptSpaceConfig): The parsed hyperparameter space config.

    Returns:
        dict: A dictionary suitable for use as a Hyperopt search space.
    """
    from hyperopt import hp

    space = {}
    for param, definition in config.space.items():
        if isinstance(definition, UniformSpace):
            space[param] = hp.uniform(param, definition.low, definition.high)
        elif isinstance(definition, QUniformSpace):
            space[param] = hp.quniform(param, definition.low, definition.high, definition.q)
        elif isinstance(definition, LogUniformSpace):
            space[param] = hp.loguniform(param, definition.low, definition.high)
        elif isinstance(definition, ChoiceSpace):
            space[param] = hp.choice(param, definition.values)
        elif isinstance(definition, ConstSpace):
            space[param] = definition.value
        else:
            raise HyperOptSpaceConfigError(f"Unknown hyperparameter space type for parameter '{param}'")
    return space


def hyperopt_space_to_config(space: Dict[str, Any]) -> HyperOptSpaceConfig:
    """
    Converts a Hyperopt search space dictionary (e.g., built by build_hyperopt_space) to a HyperOptSpaceConfig object.

    Args:
        space (Dict[str, Any]): Hyperopt search space dictionary (key: hyperparameter name, value: hp.uniform, hp.choice, number, etc.)

    Returns:
        HyperOptSpaceConfig: Corresponding configuration object.
    """
    from hyperopt.pyll.base import Apply
    params = {}
    for param, definition in space.items():
        # Constant value (not an Apply object)
        if not hasattr(definition, "name") and not isinstance(definition, Apply):
            params[param] = definition
            continue
        # Apply object (pyll function)
        node = definition
        if isinstance(node, Apply):
            fn = node.name
            args = node.pos_args
            logger.info(f"Processing hyperopt space for parameter '{param}': {fn} with args {[repr(_unpack_pyll_tree(a)) for a in args]}")
            if fn == "uniform":
                params[param] = {"type": "uniform", "low": float(args[1]), "high": float(args[2])}
            elif fn == "quniform":
                params[param] = {"type": "quniform", "low": float(args[1]), "high": float(args[2]), "q": float(args[3])}
            elif fn == "loguniform":
                params[param] = {"type": "loguniform", "low": float(args[1]), "high": float(args[2])}
            elif fn == "choice":
                # args[1] is a list of values
                params[param] = {"type": "choice", "values": args[1]}
            elif fn in ("float", "int", "str"):
                # This is a constant value (float/int/str) or a hyperopt_param generator
                value = args[0]
                resolved = _resolve_hyperopt_constant(param, fn, value)
                params[param] = resolved
            else:
                raise HyperOptSpaceConfigError(f"Unknown space type: {fn} ({type(fn)}) for parameter {param}")
        else:
            # Could be another type, e.g., a constant number
            params[param] = definition
    return HyperOptSpaceConfig(params)


def hyperopt_space_config_to_yaml(config: HyperOptSpaceConfig) -> str:
    """
    Convert a HyperOptSpaceConfig object to a YAML string in a human-friendly format.

    Args:
        config (HyperOptSpaceConfig): The hyperparameter space configuration.

    Returns:
        str: YAML representation of the hyperparameter space.
    """
    space = {}
    const_params = {}
    for param, definition in config.space.items():
        if isinstance(definition, UniformSpace):
            space[param] = {
                "type": "uniform",
                "low": definition.low,
                "high": definition.high
            }
        elif isinstance(definition, QUniformSpace):
            space[param] = {
                "type": "quniform",
                "low": definition.low,
                "high": definition.high,
                "q": definition.q
            }
        elif isinstance(definition, LogUniformSpace):
            space[param] = {
                "type": "loguniform",
                "low": definition.low,
                "high": definition.high
            }
        elif isinstance(definition, ChoiceSpace):
            space[param] = {
                "type": "choice",
                "values": definition.values
            }
        elif isinstance(definition, ConstSpace):
            const_params[param] = definition.value
        else:
            raise HyperOptSpaceConfigError(f"Unknown hyperparameter space type for parameter '{param}'")
    out = {
        "excluded_params": config.excluded_params,
        "const_params": const_params,
        "space": space,
    }
    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def _unpack_pyll_tree(node, depth=0):
    """
    Recursively unpack a pyll Apply/Literal tree to a nested Python structure for easier logging/debugging.
    Returns a dict or value representing the tree.
    """
    from hyperopt.pyll.base import Apply
    try:
        from hyperopt.pyll.base import Literal
    except ImportError:
        Literal = None
    if Literal and isinstance(node, Literal):
        # Unpack literal to a readable form
        return {"literal": node.obj}
    elif isinstance(node, Apply):
        children = []
        for arg in getattr(node, "pos_args", []):
            children.append(_unpack_pyll_tree(arg, depth+1))
        return {
            "name": getattr(node, "name", None),
            "args": children,
            "kwargs": getattr(node, "kwargs", None)
        }
    else:
        return node


def _extract_literal_value(val):
    """
    Extracts the value from a pyll Literal, dict {"literal": ...}, or returns the value itself if already a number/string.
    """
    try:
        from hyperopt.pyll.base import Literal
    except ImportError:
        Literal = None
    if Literal and isinstance(val, Literal):
        return val.obj
    if isinstance(val, dict) and "literal" in val:
        return val["literal"]
    return val


def _resolve_hyperopt_constant(param, fn, value):
    """
    Helper for hyperopt_space_to_config: resolves a constant value (float/int/str) possibly wrapped in Apply/Literal/hyperopt_param.
    Returns the appropriate value or dict for a generator.
    """
    from hyperopt.pyll.base import Apply
    import logging
    logger = logging.getLogger(__name__)
    # Log the fully unpacked tree for debugging
    logger.debug(f"[UNPACKED PYLL TREE for param={param}, fn={fn}]: {repr(_unpack_pyll_tree(value))}")
    # Unpack nested Apply nodes, including hyperopt_param
    while isinstance(value, Apply):
        logger.debug(f"NESTED Apply: {repr(_unpack_pyll_tree(value))}")
        if value.name == 'hyperopt_param':
            generator = value.pos_args[1]
            if isinstance(generator, Apply):
                gen_fn = generator.name
                gen_args = generator.pos_args
                # Handle all supported generator types
                if gen_fn == "uniform":
                    return {"type": "uniform", "low": float(_extract_literal_value(gen_args[0])), "high": float(_extract_literal_value(gen_args[1]))}
                elif gen_fn == "quniform":
                    return {"type": "quniform", "low": float(_extract_literal_value(gen_args[0])), "high": float(_extract_literal_value(gen_args[1])), "q": float(_extract_literal_value(gen_args[2]))}
                elif gen_fn == "loguniform":
                    return {"type": "loguniform", "low": float(_extract_literal_value(gen_args[0])), "high": float(_extract_literal_value(gen_args[1]))}
                elif gen_fn == "choice":
                    return {"type": "choice", "values": _extract_literal_value(gen_args[1])}
                else:
                    raise HyperOptSpaceConfigError(f"Unsupported generator type in hyperopt_param for parameter {param}: {gen_fn}")
            else:
                raise HyperOptSpaceConfigError(f"hyperopt_param for parameter {param} does not wrap a generator Apply node")
        if not hasattr(value, "pos_args") or not value.pos_args:
            break
        if value.name == "literal":
            value = value.pos_args[0]
        elif value.name in ("float", "int", "str"):
            value = value.pos_args[0]
        else:
            raise HyperOptSpaceConfigError(f"Unsupported nested Apply in {fn} for parameter {param}: {value.name} (pos_args={value.pos_args})")
    try:
        from hyperopt.pyll.base import Literal
    except ImportError:
        Literal = None
    if Literal and isinstance(value, Literal):
        value = value.obj
    # Always extract the literal value before casting
    value = _extract_literal_value(value)
    if fn == "float":
        if isinstance(value, (float, int)):
            return float(value)
        elif isinstance(value, str):
            if value == param:
                raise HyperOptSpaceConfigError(f"Parameter '{param}' is defined as a reference to itself ('{value}'), not a constant or valid float value. Check your hyperopt space definition.")
            try:
                return float(value)
            except ValueError:
                raise HyperOptSpaceConfigError(f"Cannot convert string to float for parameter '{param}': got '{value}' (likely a hyperparameter name, not a value)")
        else:
            raise HyperOptSpaceConfigError(f"Cannot convert value to float for parameter '{param}': got type {type(value)} and value {value}")
    elif fn == "int":
        if isinstance(value, int):
            return int(value)
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value, str):
            if value == param:
                raise HyperOptSpaceConfigError(f"Parameter '{param}' is defined as a reference to itself ('{value}'), not a constant or valid int value. Check your hyperopt space definition.")
            try:
                return int(value)
            except ValueError:
                raise HyperOptSpaceConfigError(f"Cannot convert string to int for parameter '{param}': got '{value}' (likely a hyperparameter name, not a value)")
        else:
            raise HyperOptSpaceConfigError(f"Cannot convert value to int for parameter '{param}': got type {type(value)} and value {value}")
    elif fn == "str":
        if isinstance(value, str):
            if value == param:
                raise HyperOptSpaceConfigError(f"Parameter '{param}' is defined as a reference to itself ('{value}'), not a constant or valid string value. Check your hyperopt space definition.")
            return value
        else:
            raise HyperOptSpaceConfigError(f"Cannot convert value to str for parameter '{param}': got type {type(value)} and value {value}")
    else:
        raise HyperOptSpaceConfigError(f"Unknown fn in _resolve_hyperopt_constant: {fn}")
