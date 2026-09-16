"""Save field-specific loading poses through the same callback as the panel."""
import json
import pytest
from pathlib import Path
from types import MethodType, SimpleNamespace

from std_msgs.msg import String
from omni_autonomy_next import goal_bridge_node as bridge_module
from omni_autonomy_next.goal_bridge_node import GoalBridgeNode
from omni_autonomy_next.remembered_poses import load_remembered_poses, resolve_remembered_pose


@pytest.mark.parametrize('name', ['A', 'B', '作業台前'])
def test_panel_save_updates_only_selected_field_and_rejects_stale_or_mismatched_requests(tmp_path, monkeypatch, name):
    monkeypatch.setattr(bridge_module.time, 'monotonic', lambda: 10.)
    left = dict(frame_id='map', x=-1.7159082740346883,
                y=4.554163889863259, yaw=-3.128787463628977)
    right = dict(frame_id='map', x=1.7233270991489245,
                 y=4.529534090573496, yaw=-.008818611566363044)
    from omni_autonomy_next.rl_residual import CadClearanceModel
    config = Path(bridge_module.__file__).resolve().parents[1]/'config'
    model = CadClearanceModel.from_yaml(config/'field_planning.yaml', config/'competition_footprints.yaml')
    for pose in (left, right):
        assert model.body_clearance([pose['x'], pose['y']], pose['yaw']) == 0.
    statuses = []
    node = SimpleNamespace(
        field_side='left', current_pose=left, current_pose_received_at=10.,
        reverse_clearance=model,
        remembered_poses={}, remembered_pose_defaults={} if name == '作業台前' else {name: left},
        remembered_poses_file=str(tmp_path/'poses.json'),
        get_parameter=lambda name: SimpleNamespace(value=1.),
        get_logger=lambda: SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        _publish_remembered_poses=lambda state, **kw: statuses.append(state),
        _parse_remembered_request=GoalBridgeNode._parse_remembered_request)
    node.effective_remembered_poses = MethodType(GoalBridgeNode.effective_remembered_poses, node)
    for side, measured in [('left', left), ('right', right)]:
        node.field_side, node.current_pose = side, measured
        GoalBridgeNode._remember_pose_cb(node, String(data=json.dumps({'name': name, 'side': side})))
        assert statuses[-1] == 'SAVED'
    restored = load_remembered_poses(node.remembered_poses_file)[name]
    assert resolve_remembered_pose(restored, 'left') == left
    assert resolve_remembered_pose(restored, 'right') == right
    GoalBridgeNode._remember_pose_cb(node, String(data=json.dumps({'name': name, 'side': 'left'})))
    assert statuses[-1] == 'SAVE_FAILED'
    node.current_pose_received_at = 0.
    GoalBridgeNode._remember_pose_cb(node, String(data=json.dumps({'name': name, 'side': 'right'})))
    assert statuses[-1] == 'STALE_CURRENT_POSE'
    assert load_remembered_poses(node.remembered_poses_file)[name] == restored


def test_gui_save_names_the_field_and_displays_both_calibrations(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from omni_autonomy_next.speed_gui_node import ControlPanel, GuiBridge, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    left = dict(frame_id='map', x=-1.9, y=4.5, yaw=3.1)
    right = dict(frame_id='map', x=1.85, y=4.48, yaw=.02)
    state = dict(state='READY', field_side='right', poses={'A': right},
                 loading_field_poses={'left': left, 'right': right})
    messages = []
    bridge = SimpleNamespace(
        cancel_goal=lambda: None, set_profile=lambda *a: None,
        set_scale=lambda *a: None, set_motion_mode=lambda *a: None,
        remembered_poses_snapshot=lambda: state,
        remember_pose_pub=SimpleNamespace(publish=messages.append))
    bridge.remember_current_pose = MethodType(GuiBridge.remember_current_pose, bridge)
    panel = ControlPanel(bridge)
    panel.timer.stop()
    try:
        panel._sync_remembered_poses()
        assert '右フィールド' in panel.slot_buttons['A'].text()
        assert 'x=-1.900' in panel.loading_calibration_details.text()
        assert 'x=1.850' in panel.loading_calibration_details.text()
        monkeypatch.setattr(QtWidgets.QMessageBox, 'question', lambda *a: QtWidgets.QMessageBox.Yes)
        panel._remember_slot_pose('A')
        assert json.loads(messages[-1].data) == {'name': 'A', 'side': 'right'}
        state.update(field_side='left', poses={'A': left})
        panel._sync_remembered_poses()
        assert '左フィールド' in panel.slot_buttons['A'].text()
        assert 'x=-1.90' in panel.remembered_details.text()
        panel._remember_slot_pose('A')
        assert json.loads(messages[-1].data) == {'name': 'A', 'side': 'left'}
    finally:
        panel.close()
        panel.deleteLater()
