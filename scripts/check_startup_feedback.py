import os
os.environ['ROS_DOMAIN_ID'] = '97'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
import time
import rclpy
from std_msgs.msg import String
from PyQt5 import QtWidgets
from omni_autonomy_next.speed_gui_node import GuiBridge, ControlPanel

rclpy.init()
app = QtWidgets.QApplication([])
bridge = GuiBridge()
panel = ControlPanel(bridge)
def state(**kw):
    bridge._state_cb(String(data=json.dumps(dict(operation_mode='full', **kw))))
    panel._refresh_operation()
try:
    panel._refresh_operation()
    assert '未接続' in panel.mode_label.text()
    for mode, text in [('demo', '安全デモ'), ('real', 'センサー確認'), ('full', '実機自動制御')]:
        bridge._state_cb(String(data=json.dumps({'operation_mode': mode})))
        panel._refresh_operation()
        assert text in panel.mode_label.text()
        if mode == 'real':
            assert not panel._movement_is_allowed()
            assert 'メニューの3' in panel.request_label.text()
    state(tracking_effective=False, require_motor_link=True, motor_link_ok=False)
    assert '自己位置推定' in panel.mode_label.text() and '通信未接続' in panel.mode_label.text()
    bridge.state_time = time.monotonic() - 2
    panel._refresh_operation()
    assert '状態更新停止' in panel.mode_label.text()
    panel.pending_request = ('A', time.monotonic() - 4)
    panel._refresh_operation()
    assert '応答なし' in panel.request_label.text()
    panel.pending_request = ('A', time.monotonic())
    bridge._navigation_state_cb(String(data=json.dumps({'remembered_pose': 'A', 'state': 'QUEUED'})))
    panel._refresh_operation()
    assert panel.pending_request is None
    state(armed=False, require_armed=True, tracking_effective=False, require_motor_link=True, motor_link_ok=False)
    panel.show()
    app.processEvents()
    panel.grab().save('/tmp/startup-feedback.png')
    print('PASS: operation modes, stale system, sensor-only rejection, independent health reasons, missing acknowledgement and matching acknowledgement')
finally:
    panel.close()
    bridge.destroy_node()
    rclpy.shutdown()
