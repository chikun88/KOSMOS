"""The faster feedback loop must never run against an uncalibrated actuator."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from omni_autonomy_next.config import calibrated_tracking_parameters, ConfigError


ROBOT = yaml.safe_load((Path(__file__).parents[1]/'config/robot.yaml').read_text())['robot']


@pytest.mark.parametrize('hardware,mode,calibration,enabled,expected', [
    (True, 'simultaneous', .38, True, True),
    (False, 'simultaneous', .38, True, False),
    (True, 'staged_heading', .38, True, False),
    (True, 'simultaneous', 1., True, False),
    (True, 'simultaneous', .38, False, False),
])
def test_tuning_requires_matching_hardware_calibration(hardware, mode, calibration, enabled, expected):
    robot = deepcopy(ROBOT)
    robot['drivetrain']['linear_command_scale'] = calibration
    robot['calibrated_tracking']['enabled'] = enabled
    params = calibrated_tracking_parameters(robot, hardware=hardware, motion_mode=mode)
    assert bool(params) is expected
    if expected:
        assert params == {'position_gain': 1.6, 'yaw_gain': 1.6, 'feedback_delay_sec': .32,
                          'predictive_sprint': True, 'sprint_turn_everywhere': True}


@pytest.mark.parametrize('value', [-1., 0., float('nan'), float('inf')])
def test_invalid_feedback_gain_is_rejected(value):
    robot = deepcopy(ROBOT)
    robot['calibrated_tracking']['position_gain'] = value
    with pytest.raises(ConfigError):
        calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')


def test_old_robot_configuration_uses_conservative_node_defaults():
    assert calibrated_tracking_parameters({}, hardware=True, motion_mode='simultaneous') == {}


@pytest.mark.parametrize('key', ['predictive_sprint', 'sprint_turn_everywhere'])
@pytest.mark.parametrize('value', ['true', 1, None])
def test_predictive_sprint_requires_boolean(value, key):
    robot = deepcopy(ROBOT)
    robot['calibrated_tracking'][key] = value
    with pytest.raises(ConfigError):
        calibrated_tracking_parameters(robot, hardware=True, motion_mode='simultaneous')
