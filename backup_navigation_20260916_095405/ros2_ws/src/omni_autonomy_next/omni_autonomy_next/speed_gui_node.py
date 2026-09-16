import json
import math
import sys
import threading
import time

from ament_index_python.packages import get_package_share_directory
from PyQt5 import QtCore, QtGui, QtWidgets
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from nav_msgs.msg import Path as NavPath

from .configured_goals import load_configured_poses
from .field_side import LEFT, RIGHT, normalize_side


FIELD_SIDE_NAMES = {LEFT: '左フィールド', RIGHT: '右フィールド'}

# Android button label -> saved pose name, in MU-3 slot order. The panel offers
# a one-tap re-measure for each so the operator never has to retype a name that
# the radio protocol matches exactly.
MU3_SLOT_LABELS = (
    ('A', '自動装填'),
    ('BAKETU2', 'バケツ2'),
    ('BAKETU3', 'バケツ3'),
    ('旗上側', '旗下'),
    ('旗下側', '旗上'),
    ('退避位置', '退避位置'),
    ('装填位置', 'スタート位置'),
)

GOAL_DISPLAY_NAMES = {
    '0': '装填待機',
    '1': '装填位置',
    '2': '位置取り待機',
    '3': '退避位置',
    '4': '固定バケツ②',
    '5': '固定バケツ③',
    '6': '旗上側',
    '7': '旗下側',
}

NAVIGATION_STATE_NAMES = {
    'IDLE': '待機中',
    'QUEUED': '目標受付',
    'WAITING_FOR_NAV2': 'Nav2起動待ち',
    'SENDING': '目標送信中',
    'ACTIVE': '移動中',
    'PREEMPTING': '目標切替中',
    'CANCELING': '停止処理中',
    'CANCELED': '停止しました',
    'RETRY_WAIT': '再試行待ち',
    'SUCCEEDED': '到着',
    'FINAL_APPROACH': '最終位置合わせ',
    'REVERSE_APPROACH': '保存地点Aへ低速で直線後退中',
    'FAILED': '移動失敗',
    'INVALID_GOAL_ID': '番号エラー',
    'INVALID_REMEMBERED_POSE': '記憶地点エラー',
    'FIELD_SIDE_CHANGED': 'フィールド切替',
}

REMEMBER_STATE_NAMES = {
    'READY': '保存地点を読み込みました',
    'SAVED': '現在地を保存しました',
    'INVALID_NAME': '地点名が不正です',
    'NO_CURRENT_POSE': '現在地をまだ取得できていません',
    'STALE_CURRENT_POSE': '現在地が古いため保存できません',
    'SAVE_FAILED': '現在地の保存に失敗しました',
    'FIELD_SIDE': 'フィールドを切り替えました',
    'INVALID_FIELD_SIDE': 'フィールド指定が不正です',
}


class GuiBridge(Node):
    def __init__(self):
        super().__init__('omni_speed_gui')
        share = get_package_share_directory('omni_autonomy_next')
        self.declare_parameter(
            'field_poses_file', share + '/config/field_poses.yaml'
        )
        self.configured_poses = load_configured_poses(
            self.get_parameter('field_poses_file').value
        )
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.arm_pub = self.create_publisher(Bool, '/system/armed', qos)
        self.estop_pub = self.create_publisher(Bool, '/system/emergency_stop', qos)
        self.scale_pub = self.create_publisher(Float32, '/system/speed_scale', qos)
        self.profile_pub = self.create_publisher(String, '/system/profile', qos)
        self.motion_mode_pub = self.create_publisher(String, '/navigation/motion_mode', qos)
        self.tracker_state = {}
        # Goal requests are deliberately VOLATILE. Replaying a stale target
        # after goal_bridge restarts would start an unrequested movement.
        goal_qos = QoSProfile(depth=5)
        goal_qos.reliability = ReliabilityPolicy.RELIABLE
        goal_qos.durability = DurabilityPolicy.VOLATILE
        self.goal_pub = self.create_publisher(
            String, '/navigation/goal_id_request', goal_qos
        )
        self.remember_pose_pub = self.create_publisher(
            String, '/navigation/remember_pose_request', goal_qos
        )
        self.remembered_goal_pub = self.create_publisher(
            String, '/navigation/remembered_goal_request', goal_qos
        )
        self.cancel_pub = self.create_publisher(
            Bool, '/navigation/cancel_request', goal_qos
        )
        # Latched so goal_bridge picks up the panel's field even if it restarts.
        self.field_side_pub = self.create_publisher(
            String, '/navigation/field_side', qos
        )
        # motor_udp_bridge deliberately ignores emergency_stop=false so that a
        # stale or transient publisher can never restart the drivebase; only
        # this Trigger clears its latch.  Without the call the panel's release
        # button cleared RuntimeGuard alone and the Pi stayed latched, so the
        # robot could not be restarted from the GUI after any E-stop.
        self.reset_estop_client = self.create_client(
            Trigger, '/system/reset_motor_estop'
        )
        self.state = {}
        self.state_time = None
        self.navigation_state = {'state': 'IDLE'}
        self.remembered_pose_state = {'state': 'WAITING', 'poses': {}}
        self.mu3_state = '無線指示待ち'
        self.mu3_link = None
        self.mu3_link_time = None
        self.plan_points = 0
        self.plan_time = None
        self.lock = threading.Lock()
        self.create_subscription(String, '/system/safety_state', self._state_cb, qos)
        self.create_subscription(String, '/mu3/navigation_status', self._mu3_cb, qos)
        self.create_subscription(String, '/motor/telemetry', self._mu3_link_cb, qos)
        self.create_subscription(NavPath, '/plan', self._plan_cb, 5)
        self.create_subscription(String, '/trajectory_tracker/status', self._tracker_cb, qos)
        self.create_subscription(
            String, '/navigation/goal_status', self._navigation_state_cb, qos
        )
        self.create_subscription(
            String, '/navigation/remembered_poses',
            self._remembered_poses_cb, qos,
        )
        self.set_arm(False)
        self.set_estop(False)

    def _state_cb(self, message):
        try:
            state = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.state = state
            self.state_time = time.monotonic()

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def _mu3_cb(self, message):
        with self.lock:
            self.mu3_state = str(message.data)

    def _plan_cb(self, message):
        with self.lock:
            self.plan_points = len(message.poses)
            self.plan_time = time.monotonic()

    def command_path_snapshot(self):
        with self.lock:
            return self.mu3_state, self.plan_points, self.plan_time

    def _mu3_link_cb(self, message):
        try:
            data = json.loads(message.data)
            alive = bool(data['pi']['mu3_alive'])
        except (ValueError, KeyError, TypeError):
            return
        with self.lock:
            self.mu3_link = alive
            self.mu3_link_time = time.monotonic()

    def mu3_link_snapshot(self):
        with self.lock:
            if self.mu3_link_time is None:
                return '受信状態待ち'
            if time.monotonic() - self.mu3_link_time > 1.0:
                return 'bacon6状態更新なし'
            return '無線受信OK' if self.mu3_link else '無線未受信（Android送信・接続を確認）'

    def _navigation_state_cb(self, message):
        try:
            state = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.navigation_state = state

    def navigation_snapshot(self):
        with self.lock:
            return dict(self.navigation_state)

    def _remembered_poses_cb(self, message):
        try:
            state = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(state.get('poses'), dict):
            return
        with self.lock:
            self.remembered_pose_state = state

    def remembered_poses_snapshot(self):
        with self.lock:
            state = dict(self.remembered_pose_state)
            state['poses'] = dict(state.get('poses', {}))
            return state

    def set_arm(self, value):
        self.arm_pub.publish(Bool(data=bool(value)))

    def set_estop(self, value):
        self.estop_pub.publish(Bool(data=bool(value)))

    def clear_estop(self):
        """Release the guard gate and the motor bridge's latched stop."""
        self.set_estop(False)
        if not self.reset_estop_client.service_is_ready():
            self.get_logger().warning(
                'Motor E-stop reset service is unavailable; the drivebase '
                'stays latched until motor_udp_bridge is reachable'
            )
            return False
        self.reset_estop_client.call_async(Trigger.Request())
        return True

    def set_scale(self, value):
        self.scale_pub.publish(Float32(data=float(value)))

    def set_profile(self, value):
        self.profile_pub.publish(String(data=str(value)))

    def set_motion_mode(self, value):
        self.motion_mode_pub.publish(String(data=str(value)))

    def _tracker_cb(self, message):
        try:
            state = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if isinstance(state, dict):
            with self.lock:
                self.tracker_state = state

    def tracker_snapshot(self):
        with self.lock:
            return dict(self.tracker_state)

    def request_goal(self, goal_id):
        goal_id = str(goal_id)
        if goal_id not in self.configured_poses:
            return False
        self.goal_pub.publish(String(data=goal_id))
        return True

    def remember_current_pose(self, name, side=None):
        name = str(name).strip()
        if not name:
            return False
        payload = name if side is None else json.dumps(
            {'name': name, 'side': normalize_side(side)}, ensure_ascii=False)
        self.remember_pose_pub.publish(String(data=payload))
        return True

    def request_remembered_goal(self, name, side=LEFT):
        name = str(name).strip()
        if not name:
            return False
        if self.remembered_goal_pub.get_subscription_count() == 0:
            return False
        self.remembered_goal_pub.publish(String(data=json.dumps(
            {'name': name, 'side': normalize_side(side)}, ensure_ascii=False)))
        return True

    def cancel_goal(self):
        self.cancel_pub.publish(Bool(data=True))

    def set_field_side(self, side):
        self.field_side_pub.publish(String(data=normalize_side(side)))


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.setWindowTitle('MU3 Omni Autonomy Next')
        self.resize(900, 900)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        title = QtWidgets.QLabel('MU3 自動制御パネル')
        title.setFont(QtGui.QFont('Sans', 18, QtGui.QFont.Bold))
        layout.addWidget(title)

        self.state_label = QtWidgets.QLabel('状態待機中')
        self.state_label.setFont(QtGui.QFont('Monospace', 12))
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        goal_group = QtWidgets.QGroupBox('行き先指定')
        goal_layout = QtWidgets.QVBoxLayout(goal_group)
        goal_layout.setSpacing(8)
        self.goal_buttons = {}
        goal_grid = QtWidgets.QGridLayout()
        goal_grid.setSpacing(8)
        for index, (name, label) in enumerate(MU3_SLOT_LABELS):
            button = QtWidgets.QPushButton(label)
            button.setMinimumHeight(48)
            button.clicked.connect(
                lambda _checked=False, pose_name=name: self._start_named_goal(pose_name))
            goal_grid.addWidget(button, index // 4, index % 4)
            self.goal_buttons[name] = button
        goal_layout.addLayout(goal_grid)
        self.request_label = QtWidgets.QLabel('地点を押すと、選択中のフィールドへ移動します。')
        self.request_label.setWordWrap(True)
        goal_layout.addWidget(self.request_label)
        goal_buttons = QtWidgets.QHBoxLayout()
        self.goal_cancel = QtWidgets.QPushButton('移動を停止')
        self.goal_cancel.setMinimumHeight(52)
        self.goal_cancel.clicked.connect(self.bridge.cancel_goal)
        goal_buttons.addWidget(self.goal_cancel, 1)
        goal_layout.addLayout(goal_buttons)

        self.navigation_label = QtWidgets.QLabel('移動状態: 待機中')
        self.navigation_label.setFont(QtGui.QFont('Sans', 12, QtGui.QFont.Bold))
        self.navigation_label.setWordWrap(True)
        self.navigation_label.setMinimumHeight(30)
        goal_layout.addWidget(self.navigation_label)
        self.mu3_label = QtWidgets.QLabel('Android / MU-3: 無線指示待ち')
        self.mu3_label.setWordWrap(True)
        self.mu3_label.setMinimumHeight(30)
        goal_layout.addWidget(self.mu3_label)
        self.plan_label = QtWidgets.QLabel('経路: まだ受信していません')
        self.plan_label.setWordWrap(True)
        self.plan_label.setMinimumHeight(30)
        goal_layout.addWidget(self.plan_label)
        layout.addWidget(goal_group)

        side_group = QtWidgets.QGroupBox('フィールド')
        side_layout = QtWidgets.QVBoxLayout(side_group)
        side_row = QtWidgets.QHBoxLayout()
        self.field_side = QtWidgets.QComboBox()
        self.field_side.setMinimumHeight(42)
        self.field_side.setFont(QtGui.QFont('Sans', 13, QtGui.QFont.Bold))
        for side in (LEFT, RIGHT):
            self.field_side.addItem(FIELD_SIDE_NAMES[side], userData=side)
        self.field_side.currentIndexChanged.connect(self._field_side_changed)
        side_row.addWidget(QtWidgets.QLabel('走行するフィールド'))
        side_row.addWidget(self.field_side, 1)
        side_layout.addLayout(side_row)
        self.field_side_details = QtWidgets.QLabel(
            'すべての保存地点を左右別々に調節・保存できます。フィールドを選び、'
            '機体を手動で保存したい位置・向きに合わせて停止させ、'
            '対象地点の保存ボタンを押してください。反対側の保存位置は変わりません。'
            'CADと重なる位置も保存できます。移動可否は別に判定します。'
        )
        self.field_side_details.setWordWrap(True)
        side_layout.addWidget(self.field_side_details)
        layout.addWidget(side_group)
        self._active_field_side = LEFT

        remembered_group = QtWidgets.QGroupBox('現在地を記憶して戻る')
        remembered_layout = QtWidgets.QVBoxLayout(remembered_group)

        slot_label = QtWidgets.QLabel('Androidの地点ボタン（現在地で上書き）')
        remembered_layout.addWidget(slot_label)
        slot_grid = QtWidgets.QGridLayout()
        self.slot_buttons = {}
        for index, (name, android_label) in enumerate(MU3_SLOT_LABELS):
            button = QtWidgets.QPushButton(f'{android_label}を現在地で上書き')
            button.setMinimumHeight(46)
            button.setToolTip(
                f'MU-3スロット {index + 1}（右フィールドは {index + 1 + 8}）を、'
                f'いまロボットがいる位置で上書きします。'
            )
            button.clicked.connect(
                lambda _checked=False, pose_name=name:
                self._remember_slot_pose(pose_name)
            )
            slot_grid.addWidget(button, index // 2, index % 2)
            self.slot_buttons[name] = button
        remembered_layout.addLayout(slot_grid)
        self.loading_calibration_details = QtWidgets.QLabel()
        self.loading_calibration_details.setWordWrap(True)
        remembered_layout.addWidget(self.loading_calibration_details)
        self._defaulted_slots = None

        remember_row = QtWidgets.QHBoxLayout()
        self.remember_name = QtWidgets.QLineEdit()
        self.remember_name.setPlaceholderText('例: 作業台前')
        self.remember_name.setMaxLength(64)
        self.remember_save = QtWidgets.QPushButton('この名前で現在地を保存')
        self.remember_save.clicked.connect(self._remember_current_pose)
        remember_row.addWidget(QtWidgets.QLabel('地点名'))
        remember_row.addWidget(self.remember_name, 1)
        remember_row.addWidget(self.remember_save)
        remembered_layout.addLayout(remember_row)

        recall_row = QtWidgets.QHBoxLayout()
        self.remembered_selector = QtWidgets.QComboBox()
        self.remembered_selector.currentIndexChanged.connect(
            self._remembered_selection_changed
        )
        self.remembered_start = QtWidgets.QPushButton('記憶した地点へ移動')
        self.remembered_start.setMinimumHeight(44)
        self.remembered_start.setEnabled(False)
        self.remembered_start.clicked.connect(self._start_remembered_goal)
        recall_row.addWidget(self.remembered_selector, 1)
        recall_row.addWidget(self.remembered_start)
        remembered_layout.addLayout(recall_row)

        self.remembered_details = QtWidgets.QLabel(
            '保存地点をgoal_bridgeから読み込み中…'
        )
        self.remembered_details.setWordWrap(True)
        remembered_layout.addWidget(self.remembered_details)
        layout.addWidget(remembered_group)
        self._remembered_names = ()
        self._last_remember_operation = None

        form = QtWidgets.QFormLayout()
        self.profile = QtWidgets.QComboBox()
        self.profile.addItems(['precision', 'balanced', 'sprint'])
        self.profile.setCurrentText('balanced')
        self.profile.currentTextChanged.connect(self.bridge.set_profile)
        form.addRow('走行プロファイル', self.profile)
        self.motion_mode = QtWidgets.QComboBox()
        self.motion_mode.addItem('旋回しながら移動（標準）', 'simultaneous')
        self.motion_mode.addItem('広い場所で旋回 → 向きを保持して移動', 'staged_heading')
        self.motion_mode.currentIndexChanged.connect(
            lambda _: self.bridge.set_motion_mode(self.motion_mode.currentData()))
        form.addRow('走行方式（次の目標から）', self.motion_mode)
        self.motion_status = QtWidgets.QLabel('追従器の状態待ち')
        self.motion_status.setWordWrap(True)
        form.addRow('走行段階', self.motion_status)

        scale_box = QtWidgets.QHBoxLayout()
        self.scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale.setRange(5, 100)
        # 100% means "run the selected profile as tuned and validated".
        # docs/ACCEPTANCE.md has the operator lower this before first motion.
        self.scale.setValue(100)
        self.scale_label = QtWidgets.QLabel('100 %')
        self.scale.valueChanged.connect(self._scale_changed)
        scale_box.addWidget(self.scale)
        scale_box.addWidget(self.scale_label)
        form.addRow('速度上限', scale_box)
        layout.addLayout(form)

        buttons = QtWidgets.QGridLayout()
        self.arm = QtWidgets.QPushButton('自動走行 ARM')
        self.arm.setCheckable(True)
        self.arm.setMinimumHeight(54)
        self.arm.toggled.connect(self._arm_changed)
        buttons.addWidget(self.arm, 0, 0)

        estop = QtWidgets.QPushButton('非常停止')
        estop.setMinimumHeight(72)
        estop.setStyleSheet('background:#b00020;color:white;font-size:20px;font-weight:bold')
        estop.clicked.connect(self._estop)
        buttons.addWidget(estop, 0, 1, 2, 1)

        clear = QtWidgets.QPushButton('非常停止 解除（再ARM必要）')
        clear.setMinimumHeight(44)
        clear.clicked.connect(self._clear_estop)
        buttons.addWidget(clear, 1, 0)
        outer.addLayout(buttons)

        note = QtWidgets.QLabel(
            '赤丸周辺では精度保護のため速度が自動制限されます。GUI値は物理上限を超えません。'
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

        self.bridge.set_profile(self.profile.currentText())
        self.bridge.set_scale(self.scale.value() / 100.0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)

    def _scale_changed(self, value):
        self.scale_label.setText(f'{value} %')
        self.bridge.set_scale(value / 100.0)

    def _arm_changed(self, armed):
        self.arm.setText('自動走行 ARMED' if armed else '自動走行 ARM')
        self.bridge.set_arm(armed)

    def _estop(self):
        self.arm.setChecked(False)
        self.bridge.set_estop(True)

    def _clear_estop(self):
        self.arm.setChecked(False)
        self.bridge.clear_estop()

    def _start_named_goal(self, name):
        if not self._movement_is_allowed():
            return
        side = self.field_side.currentData()
        if self.bridge.request_remembered_goal(name, side):
            label = dict(MU3_SLOT_LABELS).get(name, name)
            self.request_label.setText(f'操作受付: {label} / {FIELD_SIDE_NAMES[side]} — 地点要求を送信しました。到着は移動状態で確認してください。')
        else:
            self.request_label.setText('移動できません: 地点実行ノードが未接続です。自動制御システムを起動してください。')

    def _movement_is_allowed(self):
        safety = self.bridge.snapshot()
        if not safety or self.bridge.state_time is None or time.monotonic() - self.bridge.state_time > 1.0:
            reason = '安全状態が未受信または古い状態です。自動制御システムの接続を確認してください。'
        elif safety.get('emergency_stop'):
            reason = '非常停止を解除してください。'
        elif safety.get('require_armed') and not safety.get('armed'):
            reason = '「自動走行 ARM」を有効にしてから地点を押してください。'
        else:
            return True
        self.request_label.setText('移動できません: ' + reason)
        return False

    def _field_side_changed(self, *_args):
        side = self.field_side.currentData()
        if side == self._active_field_side:
            return
        answer = QtWidgets.QMessageBox.question(
            self, 'フィールドを切り替え',
            f'{FIELD_SIDE_NAMES[side]}に切り替えますか？\n'
            '実行中の移動があれば中止します。',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            self.field_side.blockSignals(True)
            self.field_side.setCurrentIndex(
                self.field_side.findData(self._active_field_side)
            )
            self.field_side.blockSignals(False)
            return
        self.bridge.set_field_side(side)

    def _remember_slot_pose(self, name):
        """Overwrite one Android slot with the pose the robot is standing on."""
        side = self.field_side.currentData()
        label = dict(MU3_SLOT_LABELS).get(name, name)
        answer = QtWidgets.QMessageBox.question(
            self, '地点を上書き',
            f'「{label}」を現在地・現在の向きで上書きしますか？\n'
            f'記録先: {FIELD_SIDE_NAMES[side]}'
            + '（このフィールドのみ更新）',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        if self.bridge.remember_current_pose(name, self.field_side.currentData()):
            self.remembered_details.setText(f'「{name}」を保存中…')

    def _remember_current_pose(self):
        name = self.remember_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(
                self, '保存できません', '地点名を入力してください。'
            )
            return
        poses = self.bridge.remembered_poses_snapshot().get('poses', {})
        if name in poses:
            answer = QtWidgets.QMessageBox.question(
                self, '保存地点を上書き',
                f'「{name}」を現在地で上書きしますか？\n'
                f'記録先: {FIELD_SIDE_NAMES[self.field_side.currentData()]}（このフィールドのみ更新）',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        if self.bridge.remember_current_pose(name, self.field_side.currentData()):
            self.remembered_details.setText(f'「{name}」を保存中…')

    def _remembered_selection_changed(self, *_args):
        name = self.remembered_selector.currentData()
        poses = self.bridge.remembered_poses_snapshot().get('poses', {})
        pose = poses.get(name)
        if not pose:
            self.remembered_start.setEnabled(False)
            return
        self.remembered_start.setEnabled(True)
        self.remembered_details.setText(
            f'「{name}」  x={pose["x"]:.3f} m   y={pose["y"]:.3f} m   '
            f'向き={math.degrees(pose["yaw"]):.1f}°'
        )

    def _start_remembered_goal(self):
        name = self.remembered_selector.currentData()
        if name:
            self._start_named_goal(name)

    def _sync_field_side(self, state):
        side = state.get('field_side')
        if side not in FIELD_SIDE_NAMES or side == self._active_field_side:
            return
        self._active_field_side = side
        self.field_side.blockSignals(True)
        self.field_side.setCurrentIndex(self.field_side.findData(side))
        self.field_side.blockSignals(False)

    def _sync_slot_buttons(self, state):
        """Show which Android slots are still on their configured seed."""
        defaulted = frozenset(state.get('defaulted') or ())
        # The panel refreshes at 10 Hz; restyling every tick would keep Qt
        # recomputing styles for buttons that have not changed.
        fields = state.get('loading_field_poses', {})
        token = (defaulted, self._active_field_side, json.dumps(fields, sort_keys=True))
        if token == self._defaulted_slots:
            return
        self._defaulted_slots = token
        descriptions = []
        for side in (LEFT, RIGHT):
            pose = fields.get(side)
            detail = ('未調整（従来の設定を使用）' if pose is None else
                      f'x={pose["x"]:.3f} m  y={pose["y"]:.3f} m  '
                      f'向き={math.degrees(pose["yaw"]):.1f}°')
            descriptions.append(f'{FIELD_SIDE_NAMES[side]}の自動装填: {detail}')
        self.loading_calibration_details.setText('\n'.join(descriptions))
        for name, android_label in MU3_SLOT_LABELS:
            button = self.slot_buttons[name]
            seeded = name in defaulted
            button.setStyleSheet(
                'background:#f9a825;color:black;font-weight:bold' if seeded else ''
            )
            if name == 'A':
                side = self._active_field_side
                button.setText(f'自動装填位置を保存（{FIELD_SIDE_NAMES[side]}）')
                button.setToolTip('現在の位置・向きを、選択中のフィールド専用に保存します。')
            else:
                button.setText(
                    f'{android_label}を現在地で上書き（{FIELD_SIDE_NAMES[self._active_field_side]}）'
                    + ('（既定値）' if seeded else '')
                )

    def _sync_remembered_poses(self):
        state = self.bridge.remembered_poses_snapshot()
        poses = state.get('poses', {})
        names = tuple(sorted(poses))
        operation = str(state.get('state', 'WAITING'))
        self._sync_field_side(state)
        self._sync_slot_buttons(state)
        saved_name = state.get('name') if operation == 'SAVED' else None
        if names != self._remembered_names:
            selected = self.remembered_selector.currentData()
            if saved_name in names:
                selected = saved_name
            self.remembered_selector.blockSignals(True)
            self.remembered_selector.clear()
            for name in names:
                self.remembered_selector.addItem(name, userData=name)
            if selected in names:
                self.remembered_selector.setCurrentIndex(names.index(selected))
            self.remembered_selector.blockSignals(False)
            self._remembered_names = names
            self._remembered_selection_changed()
        operation_token = (operation, state.get('name'), state.get('detail'),
                           state.get('field_side'), json.dumps(poses, sort_keys=True))
        if operation_token != self._last_remember_operation:
            self._remembered_selection_changed()
        if operation == 'READY' and not names:
            self.remembered_details.setText('保存地点はまだありません。')
        elif (
            operation_token != self._last_remember_operation
            and operation == 'SAVED'
            and saved_name in names
        ):
            index = self.remembered_selector.findData(saved_name)
            if index >= 0:
                self.remembered_selector.setCurrentIndex(index)
            pose = poses[saved_name]
            self.remembered_details.setText(
                f'保存しました: 「{saved_name}」 / {FIELD_SIDE_NAMES[self._active_field_side]}  '
                f'x={pose["x"]:.3f} m   y={pose["y"]:.3f} m   '
                f'向き={math.degrees(pose["yaw"]):.1f}°'
            )
        elif (
            operation_token != self._last_remember_operation
            and operation not in {'READY', 'SAVED'}
            and operation in REMEMBER_STATE_NAMES
        ):
            name = state.get('name')
            text = REMEMBER_STATE_NAMES[operation]
            if name:
                text += f': 「{name}」'
            self.remembered_details.setText(text)
        self._last_remember_operation = operation_token

    def _refresh_navigation(self):
        navigation = self.bridge.navigation_snapshot()
        state = str(navigation.get('state', 'IDLE'))
        state_name = NAVIGATION_STATE_NAMES.get(state, state)
        goal_id = navigation.get('goal_id')
        parts = [f'移動状態: {state_name}']
        if goal_id is not None:
            parts.append(f'目標={goal_id}')
        if navigation.get('remembered_pose'):
            parts.append(f'保存地点={navigation["remembered_pose"]}')
        if navigation.get('field_side') in FIELD_SIDE_NAMES:
            parts.append(FIELD_SIDE_NAMES[navigation['field_side']])
        distance = navigation.get('distance_remaining_m')
        if isinstance(distance, (int, float)) and math.isfinite(distance):
            parts.append(f'残り={distance:.2f} m')
        reason = navigation.get('reason')
        if reason:
            parts.append(f'理由={reason}')
        if navigation.get('detail'):
            parts.append(str(navigation['detail']))
        self.navigation_label.setText('   '.join(parts))
        remote, points, stamp = self.bridge.command_path_snapshot()
        translations = {
            'PREPARING': '地点指示受信・走行準備中',
            'NOT_READY': '実行できません',
            'ARM_TIMEOUT': 'ARM応答タイムアウト',
            'DISARM_TIMEOUT': '停止応答タイムアウト',
            'LOCALIZATION_UNHEALTHY': '自己位置が不確かです',
            'CONTROL_SUBSCRIBERS_UNAVAILABLE': '必要な制御ノードが未接続',
            'RELEASED': '停止ボタン / 指示解除',
            'MU3_LOST': 'MU-3通信断',
            'TELEMETRY_LOST': 'bacon6通信断',
            'GOAL_ACK_TIMEOUT': '地点実行ノードから応答なし',
            'GOAL_SENT': '地点要求送信済み',
            'TELEMETRY_STALE': 'bacon6受信データが古いです',
            'SAFETY_STATE_STALE': '安全状態が更新されていません',
            'MOTOR_LINK_LOST': 'モーター通信断',
            'MOTOR_UART_CLOSED': 'モーターUART未接続',
            'EMERGENCY_STOP': '非常停止中',
            'MOTOR_FAULT_LATCHED': 'モーター異常解除待ち',
            'RL_UNHEALTHY': '制御ポリシーが未準備',
            'SUCCEEDED': '到達', 'FAILED': '移動失敗',
        }
        self.mu3_label.setText('Android / MU-3: ' + self.bridge.mu3_link_snapshot() + '\n' + ' / '.join(
            translations.get(part, part) for part in remote.split(':')))
        self.plan_label.setText(
            '経路: まだ受信していません' if stamp is None else
            f'最後に受信した経路: {points} 点（{time.monotonic() - stamp:.1f}秒前）')
        if state == 'SUCCEEDED':
            color = '#0b6b2b'
        elif state in {
            'FAILED', 'INVALID_GOAL_ID', 'INVALID_REMEMBERED_POSE'
        }:
            color = '#b00020'
        elif state in {'ACTIVE', 'SENDING', 'PREEMPTING', 'CANCELING'}:
            color = '#1565c0'
        else:
            color = '#5f4b00'
        self.navigation_label.setStyleSheet(f'color:{color}')
        self.goal_cancel.setEnabled(True)

    def _refresh(self):
        tracker = self.bridge.tracker_snapshot()
        phases = {'SELECT': '旋回場所を探索', 'APPROACH': '向きを保って旋回場所へ移動',
                  'SETTLE': '停止を確認', 'ROTATE': 'その場で旋回', 'TRANSLATE': '向きを保って目的地へ移動'}
        if tracker:
            requested = tracker.get('requested_motion_mode')
            index = self.motion_mode.findData(requested)
            if index >= 0 and self.motion_mode.currentIndex() != index:
                self.motion_mode.blockSignals(True)
                self.motion_mode.setCurrentIndex(index)
                self.motion_mode.blockSignals(False)
            current = '分離旋回' if tracker.get('motion_mode') == 'staged_heading' else '同時旋回'
            stopped_states = {'STAGED_BLOCKED': '安全確認のため停止',
                              'STAGED_WAITING_CLEARANCE': '旋回範囲が空くのを待機',
                              'STAGED_PLANNING': '安全な経路を計画中',
                              'PLAN_STALE': '経路更新待ち', 'ODOMETRY_STALE': '計測輪のデータ待ち',
                              'POSE_STALE': '自己位置の更新待ち'}
            reasons = {'NO_SAFE_ROTATION_GATE': '安全に旋回できる場所が見つかりません',
                       'FIXED_HEADING_PATH_BLOCKED': 'この向きでは安全に通れません',
                       'BRAKING_SWEEP_BLOCKED': '制動中に接触のおそれがあります',
                       'MEASURED_BRAKING_SWEEP_BLOCKED': '実測の慣性移動で接触のおそれがあります',
                       'CLEARANCE_RECOVERY_STOPPING': '障害物との余裕を回復する前に停止を確認',
                       'CLEARANCE_NOT_INCREASING': '障害物から離れる方向を再計画',
                       'HEADING_TRACKING_ERROR': '機体の向きが許容範囲を超えています',
                       'GOAL_CANCELED': '移動を終了しました'}
            reason = tracker.get('reason')
            self.motion_status.setText(
                current + ' / ' + stopped_states.get(tracker.get('state'),
                    phases.get(tracker.get('phase'), str(tracker.get('state', ''))))
                + (' / ' + reasons.get(reason, str(reason)) if reason else ''))
        self._refresh_navigation()
        self._sync_remembered_poses()
        state = self.bridge.snapshot()
        if not state:
            self.state_label.setText('状態待機中（RuntimeGuard未受信）')
            self.state_label.setStyleSheet('color:#9a6700')
            return
        reason = state.get('reason', 'UNKNOWN')
        zone = ' / 赤丸精度ゾーン' if state.get('red_zone') else ''
        speed = 100.0 * float(state.get('applied_scale', 0.0))
        self.state_label.setText(
            f'{reason}{zone}\nprofile={state.get("profile")}  適用速度={speed:.0f}%\n'
            f'LiDAR追跡={state.get("tracking_ok")}  Motor Link={state.get("motor_link_ok")} '
            f'Auto={state.get("auto_engaged")}'
        )
        self.state_label.setStyleSheet(
            'color:#0b6b2b' if state.get('allowed') else 'color:#b00020'
        )
        # Keep controls clickable so a rejected request explains why.
        self.remembered_start.setEnabled(self.remembered_selector.count() > 0)
        self.arm.blockSignals(True)
        self.arm.setChecked(bool(state.get('armed', False)))
        self.arm.setText('自動走行 ARMED' if state.get('armed') else '自動走行 ARM')
        self.arm.blockSignals(False)

    def closeEvent(self, event):
        # Closing an operator panel must not leave a Nav2 goal active.  An
        # active goal could otherwise resume unexpectedly when another panel
        # re-arms the robot.
        self.bridge.cancel_goal()
        self.bridge.set_arm(False)
        super().closeEvent(event)


def main(args=None):
    rclpy.init(args=args)
    bridge = GuiBridge()

    def spin_bridge():
        try:
            rclpy.spin(bridge)
        except ExternalShutdownException:
            pass

    ros_thread = threading.Thread(target=spin_bridge, daemon=True)
    ros_thread.start()
    app = QtWidgets.QApplication(sys.argv)
    panel = ControlPanel(bridge)
    panel.show()
    try:
        code = app.exec_()
    finally:
        bridge.set_arm(False)
        bridge.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)
    raise SystemExit(code)
