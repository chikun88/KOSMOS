import os
os.environ['ROS_DOMAIN_ID'] = '95'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import time
import rclpy
from std_msgs.msg import String
from omni_autonomy_next.speed_gui_node import GuiBridge, ControlPanel, QtWidgets

rclpy.init()
app = QtWidgets.QApplication([])
bridge = GuiBridge()
# Opening/reopening the GUI must preserve the guard's selected speed policy.
sent_profiles = []
bridge.set_profile = sent_profiles.append
panel = ControlPanel(bridge)
assert sent_profiles == [], sent_profiles
for active in ('sprint', 'balanced', 'precision'):
    bridge._state_cb(String(data='{"profile":"' + active + '","reason":"DISARMED"}'))
    panel._refresh()
    assert panel.profile.currentText() == active
    assert sent_profiles == []
panel.profile.setCurrentText('sprint')
assert sent_profiles == ['sprint']
bridge.mu3_state = 'GOAL_SENT:BAKETU2'
bridge.plan_points = 163
bridge.plan_time = time.monotonic()
bridge.navigation_state = {'state':'ACTIVE','remembered_pose':'BAKETU2','distance_remaining_m':1.25}
bridge._mu3_link_cb(String(data='{"pi":{"mu3_alive":true}}'))
panel._refresh_navigation()
assert '無線受信OK' in panel.mu3_label.text()
assert 'BAKETU2' in panel.navigation_label.text()
assert '163' in panel.plan_label.text()
panel.show()
app.processEvents()
assert panel.grab().save('/tmp/navigation-gui.png')
bridge.mu3_state = 'NOT_READY:LOCALIZATION_UNHEALTHY'
bridge._mu3_link_cb(String(data='{"pi":{"mu3_alive":false}}'))
panel._refresh_navigation()
assert '無線未受信' in panel.mu3_label.text()
assert '自己位置が不確か' in panel.mu3_label.text()
bridge.navigation_state = {'state':'SUCCEEDED','remembered_pose':'BAKETU2'}
panel._refresh_navigation()
assert 'BAKETU2' in panel.navigation_label.text()
panel.close()
bridge.destroy_node()
rclpy.shutdown()
print('GUI_PASS')
