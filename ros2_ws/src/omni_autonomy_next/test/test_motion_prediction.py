import math
import numpy as np
import pytest
from omni_autonomy_next.motion_prediction import DelayedMotionPredictor
from omni_autonomy_next import trajectory_tracker_node as tracker
from test_smooth_arrival import make_node


def test_constant_body_twist_follows_arc_not_straight_extrapolation():
    model = DelayedMotionPredictor()
    model.record(0.,[3.,0.,.5])
    pose,velocity = model.predict([0.,0.,0.],[3.,0.,.5],.3)
    theta = .5*model.horizon
    assert pose == pytest.approx([6*math.sin(theta),6*(1-math.cos(theta)),theta],abs=.007)
    assert velocity == pytest.approx([3.,0.,.5])


def test_pending_stop_brakes_prediction_before_future_reference():
    model = DelayedMotionPredictor()
    model.record(0.,[3.,0.,0.])
    model.record(.15,[0.,0.,0.])
    pose,velocity = model.predict([0.,0.,0.],[3.,0.,0.],.3)
    assert 0. < pose[0] < 3*model.horizon
    assert 0. < velocity[0] < 1.


def test_record_copies_values_and_resets_clock_discontinuities():
    model = DelayedMotionPredictor()
    value = np.ones(3)
    model.record(1.,value)
    value[:]=0.
    assert model.history[-1][1] == pytest.approx([1.,1.,1.])
    model.record(1.,[2.,0.,0.])
    assert len(model.history) == 1
    model.record(.5,[0.,0.,0.])
    assert len(model.history) == 1
    model.record(2.,[0.,0.,0.])
    assert len(model.history) == 1
    model.record(2.05,[float('nan'),0.,0.])
    assert not model.history


def test_missing_history_does_not_assume_moving_robot_stops():
    model = DelayedMotionPredictor()
    pose,velocity = model.predict(np.zeros(3),[3.,0.,0.],0.)
    assert pose[0] == pytest.approx(3*model.horizon)
    assert velocity[0] == 3.


@pytest.mark.parametrize('now', [.5, -.1])
def test_stale_or_future_history_cannot_predict_unissued_motion(now):
    model = DelayedMotionPredictor()
    model.record(0., [3., 0., 1.])
    pose, velocity = model.predict(np.zeros(3), np.zeros(3), now)
    assert pose == pytest.approx(np.zeros(3))
    assert velocity == pytest.approx(np.zeros(3))


def test_rotating_frame_does_not_consume_linear_acceleration_for_constant_world_velocity():
    node = make_node()
    node.command = np.array([3.,0.,.5])
    node.prediction_frame = (.05,.025)
    node.last_prediction_frame = (0.,0.)
    target = tracker.to_body(np.array([3.,0.]),.025)
    result = tracker.TrajectoryTracker._rate_limit(node,*target,.5)
    assert result[:2] == pytest.approx(target)
    assert node.command == pytest.approx([3.,0.,.5])


def test_stale_frame_does_not_rotate_old_command():
    node = make_node()
    node.command = np.array([3.,0.,.5])
    node.prediction_frame = (1.,.4)
    node.last_prediction_frame = (0.,0.)
    result = tracker.TrajectoryTracker._rate_limit(node,3.,0.,.5)
    assert result[:2] == pytest.approx([3.,0.])


@pytest.mark.parametrize('delay,response', [(0.,.1),(.3,0.),(float('nan'),.1),(.6,.1),(.3,.4)])
def test_invalid_model_is_rejected(delay,response):
    with pytest.raises(ValueError):
        DelayedMotionPredictor(delay,response)
