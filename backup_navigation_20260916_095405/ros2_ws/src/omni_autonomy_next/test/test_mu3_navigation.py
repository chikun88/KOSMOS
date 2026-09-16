import pytest
from omni_autonomy_next.mu3_navigation import (
    SIDE_SLOT_OFFSET,
    RemoteNavigation,
    decode_slot,
)


def setup():
    events = []
    n = RemoteNavigation(['A', 'BAKETU2'], lambda k, v: events.append((k, v)))
    n.poses = {'A': {}, 'BAKETU2': {}}
    n.safety_update({'tracking_effective': True, 'armed': False}, 0)
    return n, events


def receive(n, slot, sequence, now, **kw):
    data = dict(slot=slot, sequence=sequence, pi_seq=int(now * 1000),
                mu3_alive=True, link_alive=True, uart_open=True,
                auto_engaged=False, estop_active=False, fault_latched=False)
    data.update(kw)
    n.receive(data, now)


def start(n):
    receive(n, 0, 0, .01)
    receive(n, 1, 1, .02)
    receive(n, 1, 1, .23)
    n.tick(.23)
    n.safety_update({'tracking_effective': True, 'armed': True}, .24)
    receive(n, 1, 1, .24, auto_engaged=True)
    n.tick(.24)


def test_saved_goal_is_armed_once_and_duplicate_frames_do_not_restart():
    n, events = setup()
    start(n)
    assert events.count(('goal', ('A', 'left'))) == 1
    receive(n, 1, 1, .25, auto_engaged=True)
    n.tick(.25)
    assert events.count(('goal', ('A', 'left'))) == 1
    cancels_before = events.count(('cancel', True))
    n.navigation_update({'remembered_pose': 'A', 'state': 'SUCCEEDED'})
    assert not n.owned
    assert events[-2:] == [('arm', False), ('status', 'SUCCEEDED')]
    assert events.count(('cancel', True)) == cancels_before


@pytest.mark.parametrize('loss', ['radio', 'telemetry', 'estop', 'localization', 'manual'])
def test_stop_and_no_automatic_resume(loss):
    n, events = setup()
    start(n)
    if loss == 'radio': receive(n, 1, 1, .26, mu3_alive=False)
    if loss == 'telemetry': n.tick(.6)
    if loss == 'estop':
        receive(n, 1, 1, .26, estop_active=True)
        n.tick(.26)
    if loss == 'localization':
        n.safety_update({'tracking_effective': False}, .26)
        n.tick(.26)
    if loss == 'manual': receive(n, 0, 2, .26)
    assert not n.owned
    assert ('arm', False) in events
    n.safety_update({'tracking_effective': True, 'armed': False}, .7)
    receive(n, 1, 1, .7)
    n.tick(.71)
    assert events.count(('goal', ('A', 'left'))) == 1


def test_restart_with_held_command_requires_release():
    n, events = setup()
    receive(n, 1, 12, .01)
    assert not n.owned and ('arm', True) not in events
    receive(n, 0, 13, .02)
    receive(n, 2, 14, .03)
    assert n.owned and n.target == 'BAKETU2'


def test_missing_pose_and_unhealthy_press_are_consumed_not_deferred():
    n, events = setup()
    receive(n, 0, 0, .01)
    receive(n, 3, 1, .02)
    assert not n.owned
    receive(n, 1, 2, .03, estop_active=True)
    receive(n, 1, 2, .04)
    assert not n.owned
    assert not any(k == 'goal' for k, v in events)


def test_duplicate_or_reordered_telemetry_does_not_refresh_watchdog():
    n, events = setup()
    start(n)
    receive(n, 2, 100, .25, pi_seq=1)
    receive(n, 2, 100, .26, pi_seq=240)
    assert n.target == 'A' and n.received == .24
    n.tick(.55)
    assert not n.owned


def test_repeated_tap_requires_disarm_before_switching():
    n, events = setup()
    start(n)
    receive(n, 2, 2, .3, auto_engaged=True)
    assert n.phase == 'DISARMING' and n.target == 'BAKETU2'
    receive(n, 2, 2, .51, auto_engaged=True)
    n.tick(.51)
    assert events.count(('arm', True)) == 1
    receive(n, 2, 2, .52)
    n.tick(.52)
    assert events.count(('arm', True)) == 2


def test_slot_encodes_the_point_and_the_field():
    names = ['A', 'BAKETU2', 'BAKETU3', '旗上側', '旗下側', '退避位置', '装填位置']
    for index, name in enumerate(names, start=1):
        assert decode_slot(index, names) == (name, 'left')
        assert decode_slot(index + SIDE_SLOT_OFFSET, names) == (name, 'right')
    # Slot 8 is point 0 on the right and slot 15 is past a seven-name list.
    assert decode_slot(SIDE_SLOT_OFFSET, names) == (None, None)
    assert decode_slot(0, names) == (None, None)
    assert decode_slot(len(names) + 1, names) == (None, None)


def test_right_field_slot_sends_the_same_name_with_the_mirrored_side():
    n, events = setup()
    receive(n, 0, 0, .01)
    receive(n, 2 + SIDE_SLOT_OFFSET, 1, .02)
    receive(n, 2 + SIDE_SLOT_OFFSET, 1, .23)
    n.tick(.23)
    n.safety_update({'tracking_effective': True, 'armed': True}, .24)
    receive(n, 2 + SIDE_SLOT_OFFSET, 1, .24, auto_engaged=True)
    n.tick(.24)
    assert events.count(('goal', ('BAKETU2', 'right'))) == 1
    # goal_bridge matches its status on the bare name, not the slot.
    n.navigation_update({'remembered_pose': 'BAKETU2', 'state': 'SUCCEEDED'})
    assert not n.owned


def test_unmapped_right_field_slot_is_rejected_not_driven():
    n, events = setup()
    receive(n, 0, 0, .01)
    receive(n, SIDE_SLOT_OFFSET, 1, .02)
    assert not n.owned
    assert ('status', 'UNKNOWN_SAVED_POSE') in events
    assert not any(kind == 'goal' for kind, _ in events)


def test_stale_safety_and_no_goal_subscriber_reject_command():
    n, events = setup()
    receive(n, 0, 0, 2.0)
    receive(n, 1, 1, 2.01)
    assert not n.owned
    n.safety_update({'tracking_effective': True}, 2.02)
    data = dict(n.input, sequence=2, pi_seq=2020)
    n.receive(data, 2.02, subscribers_ready=False)
    assert not n.owned
