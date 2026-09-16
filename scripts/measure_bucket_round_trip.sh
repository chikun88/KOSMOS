#!/bin/bash
# usage: _bucket_ab.sh NAME TRACKER(true|false)
# 地点4 <-> 5（固定バケツ2 <-> 固定バケツ3）の往復を計測する。
cd /home/egg8/Desktop/omni_autonomy_compare_20260909_01_startup
source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
source ros2_ws/install/setup.bash >/dev/null 2>&1
NAME=$1; TRACKER=$2
# static_transform_publisher は /opt/ros/jazzy/lib/tf2_ros/ にあるので
# "nav2" にも "install/omni_autonomy_next" にも一致しない。起動ごとに2個
# 残り続け、load average が 20 を超えて計測が全部当てにならなくなった。
pkill -f "ros2 launch omni_autonomy_next" >/dev/null 2>&1
pkill -f "/opt/ros/jazzy/lib/" >/dev/null 2>&1
pkill -f "install/omni_autonomy_next" >/dev/null 2>&1
sleep 1
pkill -9 -f "/opt/ros/jazzy/lib/" >/dev/null 2>&1
pkill -9 -f "install/omni_autonomy_next" >/dev/null 2>&1
sleep 4
nohup setsid ros2 launch omni_autonomy_next system.launch.py \
  demo:=true lidars:=false wheels:=false motors:=false gui:=false rviz:=false \
  initial_pose_id:=1 goal_id:=4 tracker:=$TRACKER \
  sim_command_delay:=0.12 sim_velocity_tau:=0.08 \
  > /tmp/bucket_$NAME.log 2>&1 < /dev/null &
sleep 54
python3 - <<'PY' > /tmp/bucket_${NAME}_result.txt 2>&1
import json, math, sys, time
import rclpy, yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

SHARE = ('/home/egg8/Desktop/omni_autonomy_compare_20260909_01_startup/ros2_ws/install/'
         'omni_autonomy_next/share/omni_autonomy_next/config/field_poses.yaml')
POSES = yaml.safe_load(open(SHARE, encoding='utf-8'))['poses']


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Runner(Node):
    def __init__(self):
        super().__init__('bucket_ab')
        transient = QoSProfile(depth=1)
        transient.reliability = ReliabilityPolicy.RELIABLE
        transient.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.goal_pub = self.create_publisher(
            String, '/navigation/goal_id_request',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE))
        self.pose = None
        self.status = None
        self.create_subscription(
            PoseWithCovarianceStamped, '/localization/pose', self._pose, 5)
        self.create_subscription(
            String, '/navigation/goal_status', self._status, transient)

    def _pose(self, m):
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y,
                     yaw_of(m.pose.pose.orientation))

    def _status(self, m):
        try:
            self.status = json.loads(m.data)
        except ValueError:
            pass

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def goto(self, goal_id, timeout=45.0):
        self.status = None
        self.goal_pub.publish(String(data=str(goal_id)))
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            status = self.status or {}
            # goal_id を突き合わせないと、直前の目標の SUCCEEDED を
            # 拾って「1.5 秒で到達」のような偽の成功になる。
            if str(status.get('goal_id')) != str(goal_id):
                continue
            state = status.get('state')
            if state == 'SUCCEEDED':
                break
            if state in ('ABORTED', 'REJECTED', 'CANCELED'):
                return None, None
        else:
            return None, None
        elapsed = time.monotonic() - started
        target = POSES[goal_id]

        def err():
            return math.hypot(self.pose[0] - target['x'],
                              self.pose[1] - target['y'])

        # 宣言時・整定までの最大（＝接触リスクはここ）・整定後の3つを見る。
        at_declaration = err()
        peak = at_declaration
        end = time.monotonic() + 2.5
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            peak = max(peak, err())
        settled = err()
        yaw_error = abs(math.atan2(
            math.sin(self.pose[2] - target['yaw']),
            math.cos(self.pose[2] - target['yaw'])))
        return elapsed, (at_declaration, peak, settled, yaw_error)


rclpy.init()
node = Runner()
# 起動時目標(4)が終わるまで待つ。走行中に次を投げると preempt になり、
# 最初の区間の計測が成立しない。
deadline = time.monotonic() + 60.0
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.05)
    status = node.status or {}
    if str(status.get('goal_id')) == '4' and status.get('state') == 'SUCCEEDED':
        break
node.spin(2.0)
print('%-8s %-8s %-9s %-9s %-9s %-9s'
      % ('leg', 'time[s]', 'decl[mm]', 'peak[mm]', 'settl[mm]', 'yaw[deg]'))
for goal in (5, 4, 5, 4):
    elapsed, err = node.goto(goal)
    if elapsed is None:
        print('%-8s %s' % ('-> %d' % goal, 'FAILED / TIMEOUT'))
        continue
    print('%-8s %-8.1f %-9.0f %-9.0f %-9.0f %-9.2f'
          % ('-> %d' % goal, elapsed, 1000.0 * err[0], 1000.0 * err[1],
             1000.0 * err[2], math.degrees(err[3])))
node.destroy_node()
rclpy.shutdown()
PY
cat /tmp/bucket_${NAME}_result.txt
# static_transform_publisher は /opt/ros/jazzy/lib/tf2_ros/ にあるので
# "nav2" にも "install/omni_autonomy_next" にも一致しない。起動ごとに2個
# 残り続け、load average が 20 を超えて計測が全部当てにならなくなった。
pkill -f "ros2 launch omni_autonomy_next" >/dev/null 2>&1
pkill -f "/opt/ros/jazzy/lib/" >/dev/null 2>&1
pkill -f "install/omni_autonomy_next" >/dev/null 2>&1
sleep 1
pkill -9 -f "/opt/ros/jazzy/lib/" >/dev/null 2>&1
pkill -9 -f "install/omni_autonomy_next" >/dev/null 2>&1
