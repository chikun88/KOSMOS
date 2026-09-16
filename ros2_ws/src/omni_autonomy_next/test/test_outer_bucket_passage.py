"""Use the native BT passage rule and sweep its outer CAD passage region."""
import ctypes
from pathlib import Path

from ament_index_python.packages import get_package_prefix
import numpy as np
import pytest
import yaml

from omni_autonomy_next.rl_residual import CadClearanceModel

CONFIG = Path(__file__).resolve().parents[1]/'config'


@pytest.fixture(scope='module')
def passage():
    library = ctypes.CDLL(str(Path(get_package_prefix('omni_route_bt'))/
        'lib/libomni_remove_passed_bucket_goals_bt_node.so'))
    function = library.omni_outer_gate_passed
    function.argtypes = [ctypes.c_double]*7
    function.restype = ctypes.c_bool
    return function


def test_only_central_bypass_gates_have_outer_passage():
    routes = yaml.safe_load((CONFIG/'routes.yaml').read_text())
    tagged = [(p['x'], p['y']) for p in routes['fixed_bucket_transit']['waypoints']
              if p.get('outer_y_passage')]
    assert tagged == [(-1.55, .15), (-1.55, -1.25)]
    assert all(not p.get('outer_y_passage')
               for approach in routes['fixed_goal_approaches'].values()
               for side in ('upper', 'lower') for p in approach[side]['waypoints'])


@pytest.mark.parametrize('side', [-1., 1.])
@pytest.mark.parametrize('gate_y,next_y', [(.15, .50), (.15, -1.25),
                                         (-1.25, .15), (-1.25, -1.55)])
def test_passage_region_is_outside_the_bucket_and_cad_clear(passage, side, gate_y, next_y):
    model = CadClearanceModel.from_yaml(CONFIG/'field_planning.yaml', CONFIG/'competition_footprints.yaml')
    yaw = np.pi if side > 0 else 0.
    positions = []
    for dx in np.linspace(0., .45, 13):
        for dy in np.linspace(-.45, .45, 25):
            p = [side*(1.55+dx), gate_y+dy]
            if passage(*p, yaw, side*1.55, gate_y, next_y, .45):
                positions.append(p)
                assert side*p[0] >= 1.55
                assert np.sign(next_y-gate_y)*(p[1]-gate_y) >= .02-1.e-9
    assert positions
    for error in (-.04, 0., .04):
        margins = model.clearance_over_poses(np.array(positions), np.full(len(positions), yaw+error))
        assert np.min(margins) > .025
