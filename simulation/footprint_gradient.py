"""Use the packaged pure geometry helper without requiring a ROS installation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

try:
    from omni_autonomy_next.footprint_gradient import footprint_outward_direction
except ModuleNotFoundError as error:
    if error.name not in ('omni_autonomy_next', 'omni_autonomy_next.footprint_gradient'):
        raise
    source = (Path(__file__).resolve().parents[1] / 'ros2_ws' / 'src'
              / 'omni_autonomy_next' / 'omni_autonomy_next' / 'footprint_gradient.py')
    spec = spec_from_file_location('_kosmos_footprint_gradient', source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    footprint_outward_direction = module.footprint_outward_direction
