"""Optional staged-heading execution for the existing trajectory tracker."""
import math
import time
import numpy as np
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .staged_heading import HeadingStage, TurnResponse, free_rotation_disk, rotation_clearance, wrap
from .reverse_approach import reverse_command, TIMEOUT as REVERSE_TIMEOUT
from .execution_clearance import braking_pose_path, additional_delay_clearance_error


_TRANSIENT_BRAKING_REASONS = frozenset((
    'BRAKING_SWEEP_BLOCKED', 'MEASURED_BRAKING_SWEEP_BLOCKED',
    'CLEARANCE_RECOVERY_STOPPING', 'CLEARANCE_NOT_INCREASING'))


class StagedHeadingMixin:
    def _init_staged_heading(self):
        self.motion_mode = str(self.get_parameter('motion_mode').value)
        if self.motion_mode not in ('simultaneous', 'staged_heading'):
            raise ValueError('motion_mode must be simultaneous or staged_heading')
        self.requested_motion_mode = self.motion_mode
        self.heading_stage = None
        self.stage_goal_enabled = False
        self.stage_revision = 0
        self.stage_blocked = None
        self.stage_continuation = None
        self.stage_live_map = None
        self.stage_tf = Buffer()
        self.stage_listener = TransformListener(self.stage_tf, self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/navigation/motion_mode', self._on_motion_mode, qos)
        self.create_subscription(String, '/navigation/goal_status', self._stage_goal_status, qos)
        self.create_subscription(Bool, '/navigation/cancel_request', self._stage_cancel, 5)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._stage_costmap, qos)

    def _on_motion_mode(self, message):
        if message.data not in ('simultaneous', 'staged_heading'):
            self.get_logger().warning('Rejected unknown motion mode: ' + message.data)
            return
        self.requested_motion_mode = message.data
        self._status('MOTION_MODE_QUEUED', applies='next_goal')

    def _stage_new_goal(self):
        # Called under the tracker lock. Invalidate the worker AND the previous
        # trajectory, including a same-position goal with a different yaw.
        self.reverse_goal = None
        self.reverse_ignore_cad = False
        self.reverse_settled_since = None
        self.motion_mode = self.requested_motion_mode
        self.stage_goal_enabled = True
        self.heading_stage = None
        self.stage_continuation = None
        self.stage_revision += 1
        self.stage_blocked = None
        self.trajectory = None
        self.last_plan = None
        self.pending_plan = None
        self.plan_received_stamp = None
        self.final_approach_until = 0.

    def _stage_cancel(self, message):
        if message.data:
            with self.lock:
                self.reverse_goal = None
                self.reverse_ignore_cad = False
                self.stage_goal_enabled = False
                self.stage_continuation = None
                self.stage_revision += 1
                self.trajectory = None
                self.pending_plan = None
                self.last_plan = None
                self.plan_received_stamp = None
                self.stage_blocked = 'GOAL_CANCELED'
            self._publish(0., 0., 0.)

    def _stage_goal_status(self, message):
        import json
        try:
            data = json.loads(message.data)
            state = data.get('state')
        except (ValueError, AttributeError):
            return
        # /active_goal and /goal_status are separate DDS topics. A previous
        # action's retained/delayed terminal result can arrive after a new goal
        # and must not disable the controller for that new request.
        goal_stamp = (data.get('cancel_goal_stamp', data.get('goal_stamp'))
                      if state == 'PREEMPTING' else data.get('goal_stamp'))
        active_stamp = getattr(self, 'active_goal_stamp', None)
        if active_stamp is not None and goal_stamp != active_stamp:
            return
        if state == 'REVERSE_APPROACH':
            target = data.get('reverse_goal')
            if (not isinstance(target, list) or len(target) != 3
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in target)):
                return
            now = time.monotonic()
            with self.lock:
                if getattr(self, 'reverse_goal', None) is None:
                    self._stage_new_goal()
                    self.reverse_goal = np.array(target, dtype=float)
                    self.active_goal = self.reverse_goal.copy()
                    self.reverse_started = now
                    self.reverse_blocked = None
                    self.heading_stage = HeadingStage(target[2], target[2], phase='TRANSLATE')
                elif not np.allclose(self.reverse_goal, target, atol=1.e-6, rtol=0.):
                    return
                self.reverse_heartbeat = now
                self.reverse_ignore_cad = (data.get('ignore_cad') is True
                                           and data.get('remembered_pose') == 'A')
        elif state in ('CANCELING', 'CANCELED', 'FAILED', 'PREEMPTING', 'SUCCEEDED'):
            self._stage_cancel(Bool(data=True))
        elif state == 'FINAL_APPROACH' and self.stage_goal_enabled:
            self.final_approach_until = time.monotonic()+5.
            self.finished_at = self.terminal_since = None

    def _reverse_tick(self, now, pose, pose_stamp):
        """The bridge has already settled at the 25 cm gate and ended Nav2."""
        reason = getattr(self, 'reverse_blocked', None)
        if now-self.reverse_started > REVERSE_TIMEOUT:
            reason = 'REVERSE_TIMEOUT'
        if reason:
            self._publish(0., 0., 0.)
            self._status(reason)
            return
        if (pose is None or not np.isfinite(pose).all() or now-pose_stamp > .3
                or now-self.reverse_heartbeat > .6):
            self._publish(0., 0., 0.)
            self._status('REVERSE_FEEDBACK_STALE')
            return
        scale = float(np.clip(self.speed_scale, 0., 1.))
        measured = self.velocity.copy()
        command, state = reverse_command(pose, self.reverse_goal, measured, scale,
            float(self.get_parameter('feedback_delay_sec').value))
        if state == 'REVERSE_DEVIATION':
            self.reverse_blocked = state
        # Validate the entire short straight segment with the full CAD body.
        points = np.linspace(pose[:2], self.reverse_goal[:2], 61)
        if not getattr(self, 'reverse_ignore_cad', False) and (self.clearance is None or any(np.min(self.clearance.clearance_over_poses(
                points, np.full(len(points), yaw), cap=.1)) < .026
                for yaw in (pose[2], self.reverse_goal[2]))):
            command, state = (0., 0., 0.), 'REVERSE_CLEARANCE_BLOCKED'
        if (state == 'REVERSING' and not any(command)
                and self.stage_blocked in _TRANSIENT_BRAKING_REASONS):
            # A bare zero cannot erase a failed sweep. Explicitly certify the
            # fresh raw stopping envelope before allowing settlement again.
            self._stage_safe_command(0., 0., 0., recertify_zero=True)
        self._publish(*command)
        self._status(state, cad_override=getattr(self, 'reverse_ignore_cad', False))

    def _stage_costmap(self, message):
        try:
            grid = np.asarray(message.data, dtype=np.int16).reshape(
                message.info.height, message.info.width)
            q = message.info.origin.orientation
            if abs(q.x)+abs(q.y)+abs(q.z) > 1.e-6:
                raise ValueError('rotated occupancy grid origin')
            # Store grid<-map, refreshed with every full local map. Never treat
            # odom coordinates as map coordinates after a localization update.
            if message.header.frame_id == 'map':
                transform = (0., 0., 0.)
            else:
                t = self.stage_tf.lookup_transform(message.header.frame_id, 'map', Time())
                q = t.transform.rotation
                yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                transform = (t.transform.translation.x, t.transform.translation.y, yaw)
            stamp = message.header.stamp.sec + message.header.stamp.nanosec*1.e-9
            age = self.get_clock().now().nanoseconds*1.e-9 - stamp
            if not -.1 <= age <= .8:
                raise ValueError('stale costmap source timestamp')
            self.stage_live_map = (grid, float(message.info.resolution),
                (message.info.origin.position.x, message.info.origin.position.y),
                transform, time.monotonic()-max(age, 0.))
        except Exception:
            self.stage_live_map = None

    def _stage_live_turn_clear(self, position, now):
        snapshot = self.stage_live_map
        if snapshot is None or now-snapshot[-1] > .8:
            return False
        grid, resolution, origin, (x, y, yaw), _ = snapshot
        c, s = math.cos(yaw), math.sin(yaw)
        center = (x+c*position[0]-s*position[1], y+s*position[0]+c*position[1])
        return free_rotation_disk(grid, resolution, origin, center,
                                  self.clearance.radius + .12)

    def _stage_tick(self, now, pose, measured, scale):
        """Return True when this phase owns this control tick."""
        stage = self.heading_stage
        # Transient braking faults may retry with fresh feedback, but they
        # remain arrival evidence until an actual new proof succeeds.
        if (self.stage_blocked
                and self.stage_blocked not in _TRANSIENT_BRAKING_REASONS):
            self._publish(0., 0., 0.)
            self._status('STAGED_BLOCKED', reason=self.stage_blocked)
            return True
        if stage is None:
            self._publish(0., 0., 0.)
            self._status('STAGED_PLANNING')
            return True
        if (stage.phase in ('APPROACH', 'TRANSLATE')
                and self.stage_blocked in _TRANSIENT_BRAKING_REASONS
                and np.linalg.norm(self.command) <= 1.e-9
                and np.linalg.norm(measured[:2]) < .025
                and abs(measured[2]) < .025):
            # Normal terminal/paused paths can publish bare zero. Explicitly
            # recertify the raw stopping envelope before clearing its fault.
            self._stage_safe_command(0., 0., 0., recertify_zero=True)
            if (self.stage_blocked
                    and self.stage_blocked not in _TRANSIENT_BRAKING_REASONS):
                self._publish(0., 0., 0.)
                self._status('STAGED_BLOCKED', reason=self.stage_blocked)
                return True
        if stage.phase == 'TRANSLATE':
            # The yaw servo must remain active to reject disturbances. Zeroing
            # the entire command here also zeroed the only correcting torque.
            # The swept-footprint gate checks the actual yaw and correction.
            return False
        if stage.phase == 'APPROACH':
            # A turn gate is an open region, not a precision docking target.
            # Capture its braking envelope before a positional servo asks the
            # robot to return to a point it just passed. Keep the full stop and
            # live/continuous rotation checks before applying any yaw torque.
            speed = max(float(np.linalg.norm(measured[:2])),
                        float(np.linalg.norm(self.command[:2])))
            delay = float(self.get_parameter('feedback_delay_sec').value)
            stop_distance = speed*delay + speed**2/(2*max(
                getattr(self, 'deceleration', self.acceleration), .1))
            early = np.linalg.norm(stage.gate-pose[:2]) + stop_distance > .03
            if (np.linalg.norm(stage.gate-pose[:2]) > .10
                    or stop_distance > .025
                    or speed >= .08):
                return False
            if (not self._stage_live_turn_clear(pose[:2], now)
                    or rotation_clearance(self.clearance, pose[:2], pose[2], stage.target)
                       < .10 + (.10 if early else stop_distance)):
                return False
            with self.lock:
                # Capture a checked turning region instead of crawling back
                # to its nominal center. The braking envelope fits within
                # 25 mm nominally. Early capture reserves 100 mm of CAD
                # clearance for uncertain braking; rotation keeps 35 mm.
                # Departure is rebuilt from the measured stop after settling.
                stage.gate = pose[:2].copy()
                stage.settle_drift = .10 if early else .035
                stage.phase = 'SETTLE'
                stage.braking_started = now
                self.stage_revision += 1
        drift_limit = stage.settle_drift if stage.phase == 'SETTLE' else .035
        if np.linalg.norm(stage.gate-pose[:2]) > drift_limit:
            self._publish(0., 0., 0.)
            self._status('STAGED_BLOCKED', reason='ROTATION_POSITION_DRIFT')
            return True
        # Finish the planned translation stop before holding position for the
        # turn. The capture check already reserves the delayed braking distance.
        if stage.phase == 'SETTLE':
            commanding_motion = np.linalg.norm(self.command) > 1.e-9
            if commanding_motion:
                # A planned stop may decelerate only while its checked region
                # is still clear. Fault stops bypass the rate limiter.
                if (not self._stage_live_turn_clear(pose[:2], now)
                        or rotation_clearance(self.clearance, pose[:2], pose[2], stage.target) < .10):
                    stage.settled_since = None
                    stage.braking_started = now
                    self._publish(0., 0., 0.)
                    self._status('STAGED_WAITING_CLEARANCE')
                    return True
                # Drain transport from the last moving command, not just from
                # entry into SETTLE; delayed odometry can still report zero.
                stage.braking_started = now
            stopped = (not commanding_motion
                       and np.linalg.norm(measured[:2]) < .02
                       and abs(measured[2]) < .025)
            if not stopped:
                stage.settled_since = None
            elif stage.settled_since is None:
                stage.settled_since = now
            elif (now-stage.settled_since >= .05-1.e-9
                    and (stage.braking_started is None or now-stage.braking_started >=
                         float(self.get_parameter('feedback_delay_sec').value))):
                with self.lock:
                    # Build departure from the observed stop, so there is no
                    # return leg to the old mathematical gate after rotation.
                    stage.gate = pose[:2].copy()
                    stage.phase = 'ROTATE'
                    stage.rotation_started = 0.
                    stage.rotation_finishing = False
                    stage.turn_response = TurnResponse()
                    stage.settled_since = None
                    self.stage_revision += 1
                    self.pending_plan = self.last_plan
                self.plan_event.set()
            if stage.phase == 'SETTLE':
                self._publish(*self._rate_limit(0., 0., 0.))
                self._status('STAGED_SETTLING')
                return True
        self.best_distance_at = now
        self.finished_at = self.terminal_since = None
        stage.rotation_started += getattr(self, 'control_dt', self.period) * scale
        if stage.rotation_started > 20.:
            self.stage_blocked = 'ROTATION_TIMEOUT'
            self._publish(0., 0., 0.)
            self._status('STAGED_BLOCKED', reason=self.stage_blocked)
            return True
        if (not self._stage_live_turn_clear(pose[:2], now)
                or rotation_clearance(self.clearance, pose[:2], pose[2], stage.target) < .10):
            stage.settled_since = None
            self._publish(0., 0., 0.)
            self._status('STAGED_WAITING_CLEARANCE')
            return True
        error = wrap(stage.target-pose[2])
        response = stage.turn_response.update(now, float(pose[2]),
            float(self.command[2]), float(self.get_parameter('feedback_delay_sec').value))
        settled = abs(error) < .008 and abs(measured[2]) < .025 and np.linalg.norm(measured[:2]) < .02
        if settled:
            if stage.settled_since is None:
                stage.settled_since = now
            if now-stage.settled_since >= .05-1.e-9:
                with self.lock:
                    continuation = getattr(self, 'stage_continuation', None)
                    profile_key = (self.speed_limit, self.lateral_limit, self.yaw_limit,
                        self.acceleration, self.lateral_acceleration, self.yaw_acceleration,
                        getattr(self, 'deceleration', self.acceleration))
                    ready = (continuation is not None
                        and continuation[1] == profile_key
                        and continuation[2] == self.stage_revision
                        and np.linalg.norm(continuation[0].points[0]-pose[:2]) <= .015)
                    stage.phase = 'TRANSLATE'
                    self.stage_revision += 1
                    self.trajectory = continuation[0] if ready else None
                    if ready:
                        self.reference_time = 0.
                        self.trajectory_profile_key = profile_key
                        self.plan_stamp = now
                        # The gate's near-zero remaining distance is not a
                        # progress record for the departure. Equivalent replans
                        # can reuse this trajectory forever without resetting
                        # the endpoint, falsely triggering NO_PROGRESS later.
                        self.tracked_endpoint = self.trajectory.points[-1].copy()
                        self.best_distance = math.inf
                        self.best_distance_at = now
                        self.terminal_best_yaw = math.inf
                        self.terminal_best_distance = math.inf
                    self.stage_continuation = None
                    self.pending_plan = self.last_plan
                    self.finished_at = self.terminal_since = None
                self.plan_event.set()
                if ready:
                    return False
            self._publish(0., 0., 0.)
            self._status('STAGED_ROTATION_SETTLED')
            return True
        stage.settled_since = None
        if np.linalg.norm(measured[:2]) >= .03:
            self._publish(0., 0., 0.)
            self._status('STAGED_BLOCKED', reason='TRANSLATION_DURING_ROTATION')
            return True
        predicted_error = wrap(error-measured[2]*float(self.get_parameter('feedback_delay_sec').value))
        # Accelerate the main turn, then latch the low-gain finishing loop.
        # Do not re-enter the fast loop if latency carries the body past the
        # target: re-engagement on every overshoot creates a limit cycle.
        if abs(predicted_error) < .06:
            stage.rotation_finishing = True
        # A stronger finishing servo removes the long low-speed tail. Reduce
        # it by the observed drive response so a high-gain base does not hunt.
        gain = (1.5 if stage.rotation_finishing else 2.5) / response
        target_rate = np.clip(gain*float(self.get_parameter('yaw_gain').value)*predicted_error,
                              -self.yaw_limit*scale, self.yaw_limit*scale)
        # Keep sufficient angle to brake after the measured transport delay.
        delay = float(self.get_parameter('feedback_delay_sec').value)
        braking_rate = math.sqrt((self.yaw_acceleration*delay)**2
            + 2*self.yaw_acceleration*abs(predicted_error))-self.yaw_acceleration*delay
        target_rate = float(np.clip(target_rate, -braking_rate, braking_rate))
        _, _, rate = self._rate_limit(0., 0., float(target_rate))
        rate = self._monitor_safe_yaw_rate(rate, pose[:2], float(pose[2]))
        self._publish(0., 0., rate)
        self._status('STAGED_ROTATING', yaw_error=round(error, 4),
                     turn_response_gain=round(response, 3))
        return True

    def _stage_safe_command(self, vx, vy, wz, *, recertify_zero=False):
        """Check independent delayed braking envelopes, retaining stage floors."""
        entered = time.monotonic()
        if not (vx or vy or wz) and not recertify_zero:
            return 0., 0., 0.
        if (getattr(self, 'stage_blocked', None) is not None
                and self.stage_blocked not in _TRANSIENT_BRAKING_REASONS):
            return 0., 0., 0.
        if (getattr(self, 'reverse_goal', None) is not None
                and getattr(self, 'reverse_ignore_cad', False)):
            return vx, vy, wz
        if self.heading_stage is None or self.clearance is None:
            return 0., 0., 0.
        if self.heading_stage.phase not in ('APPROACH', 'TRANSLATE'):
            return vx, vy, wz
        self.stage_execution_clearance_bound = self.stage_measured_clearance_bound = None
        self.stage_execution_age_error = self.stage_measured_age_error = None
        self.stage_braking_elapsed = None
        self.stage_certificate_context = None
        try:
            pose = np.asarray(self.pose, dtype=float).copy()
            raw = getattr(self, 'raw_velocity', self.velocity)
            measured = np.asarray(raw, dtype=float).copy()
            if (pose.shape != (3,) or measured.shape != (3,)
                    or not np.isfinite(pose).all() or not np.isfinite(measured).all()):
                raise ValueError('invalid staged pose or raw velocity')
            # Real nodes retain raw acquisition stamps; unstamped standalone
            # fixtures intentionally supply their own velocity snapshot.
            stamped = hasattr(self, 'raw_velocity')
            pose_age = 0.
            valid_work_sec = None
            if stamped:
                now = entered
                pose_stamp, velocity_stamp = self.pose_stamp, self.velocity_stamp
                pose_timeout = float(self.get_parameter('pose_timeout_sec').value)
                velocity_timeout = float(self.get_parameter('velocity_timeout_sec').value)
                pose_age, velocity_age = now-pose_stamp, now-velocity_stamp
                if (not math.isfinite(pose_age) or not math.isfinite(velocity_age)
                        or not 0. <= pose_age <= pose_timeout
                        or not 0. <= velocity_age <= velocity_timeout):
                    raise ValueError('staged execution source stale')
                valid_work_sec = min(pose_timeout-pose_age,
                                     velocity_timeout-velocity_age)
            result = StagedHeadingMixin._stage_checked_command(
                self, vx, vy, wz, pose, measured, pose_age, entered,
                valid_work_sec=valid_work_sec)
            if self.stage_certificate_context is not None:
                self.stage_certificate_context['velocity_acquisition_age_sec'] = (
                    velocity_age if stamped else None)
            if stamped and (time.monotonic()-pose_stamp > pose_timeout
                            or time.monotonic()-velocity_stamp > velocity_timeout
                            or getattr(self, 'stopping', False)):
                raise ValueError('staged execution source expired during sweep')
            if not isinstance(result, dict):
                return result
            elapsed = time.monotonic()-entered
            self.stage_braking_elapsed = elapsed
            def age_error(command):
                return additional_delay_clearance_error(
                    command, result['delay'], result['deceleration'],
                    self.clearance.radius, elapsed)
            measured_error = age_error(measured)
            self.stage_measured_age_error = measured_error
            self.stage_measured_clearance_bound = result['measured_min']-measured_error
            if self.stage_measured_clearance_bound < result['floor']:
                self.stage_blocked = 'MEASURED_BRAKING_SWEEP_BLOCKED'
                return 0., 0., 0.
            for _, candidate, nominal_min, endpoint, candidate_trace in sorted(
                    result['candidates'], key=lambda value: value[0], reverse=True):
                error = age_error(candidate)
                if (nominal_min-error >= result['floor']
                        and (not result['recovering']
                             or endpoint-error > result['start_clearance']+1.e-6)):
                    self.stage_execution_age_error = error
                    self.stage_execution_clearance_bound = nominal_min-error
                    if self.stage_certificate_context is not None:
                        self.stage_certificate_context['selected_twist'] = candidate.tolist()
                        self.stage_certificate_context['selected_certificate'] = candidate_trace
                    if self.stage_blocked in _TRANSIENT_BRAKING_REASONS:
                        self.stage_blocked = None
                    return tuple(candidate)
            self.stage_blocked = ('CLEARANCE_NOT_INCREASING' if result['recovering']
                                  else 'BRAKING_SWEEP_BLOCKED')
            return 0., 0., 0.
        except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
            self.stage_blocked = f'INVALID_BRAKING_SWEEP:{error}'
            return 0., 0., 0.

    def _stage_checked_command(self, vx, vy, wz, pose, measured, pose_age, entered,
                               *, valid_work_sec=None):
        if (getattr(self, 'reverse_goal', None) is not None
                and getattr(self, 'reverse_ignore_cad', False)):
            return vx, vy, wz
        if self.heading_stage is None or self.clearance is None:
            return (0., 0., 0.)
        if self.heading_stage.phase not in ('APPROACH', 'TRANSLATE'):
            return vx, vy, wz
        start_clearance = self.clearance.body_clearance(pose[:2],pose[2], cap=.35)
        if not math.isfinite(start_clearance) or start_clearance < 0.:
            raise ValueError('invalid staged initial footprint clearance')
        reversing = getattr(self, 'reverse_goal', None) is not None
        recovering = start_clearance < .025+self.clearance.radius*.02+.005
        if reversing:
            # Saved A has 28 mm of CAD clearance. Retain a 25 mm floor plus
            # continuous-sweep reserve without introducing lateral recovery.
            if start_clearance < .026:
                return 0., 0., 0.
            recovering = False
        if recovering:
            # Keep useful tangential motion while gently restoring clearance.
            # Requiring a full stop for lateral correction creates repeated
            # stop/replan cycles even when measured motion is wall-parallel.
            point, yaw = pose[:2], pose[2]
            gradient = np.array([
                self.clearance.body_clearance(point+offset,yaw)
                - self.clearance.body_clearance(point-offset,yaw)
                for offset in (np.array([.002,0.]),np.array([0.,.002]))])
            norm = float(np.linalg.norm(gradient))
            if norm < 1.e-8:
                self.stage_blocked = 'CLEARANCE_NOT_INCREASING'
                return 0.,0.,0.
            c,s = math.cos(yaw),math.sin(yaw)
            outward = np.array([c*gradient[0]+s*gradient[1],
                                 -s*gradient[0]+c*gradient[1]])/norm
            measured_speed = float(np.linalg.norm(self.velocity[:2]))
            already_outward = (measured_speed <= .08 and
                float(self.velocity[:2] @ outward) >= .8*measured_speed)
            if (start_clearance < .03 or
                    (start_clearance < .035 and measured_speed > .03 and not already_outward)):
                self.stage_blocked = 'CLEARANCE_RECOVERY_STOPPING'
                return 0.,0.,0.
            requested = np.array([vx,vy])
            normal = float(requested @ outward)
            tangent = requested-normal*outward
            if normal < -1.e-6 and np.linalg.norm(tangent) < 1.e-6:
                self.stage_blocked = 'CLEARANCE_NOT_INCREASING'
                return 0.,0.,0.
            original_speed = math.hypot(vx,vy)
            corrected = tangent + min(.035, max(.02,normal))*outward
            norm = float(np.linalg.norm(corrected))
            if norm > original_speed:
                corrected *= original_speed/max(norm,1.e-9)
            vx,vy = corrected
            wz = 0.
        delay = float(self.get_parameter('feedback_delay_sec').value)+pose_age
        deceleration = getattr(self, 'deceleration', self.acceleration)
        angular_deceleration = self.yaw_acceleration
        floor = min(.035,start_clearance-.001) if recovering else .035
        if reversing:
            floor = .025
        context = dict(pose=pose.tolist(), raw_twist=measured.tolist(),
                       requested_twist=[float(vx), float(vy), float(wz)],
                       pose_acquisition_age_sec=pose_age,
                       initial_delay_sec=delay,
                       remaining_source_lifetime_sec=valid_work_sec,
                       floor_m=float(floor), recovery=bool(recovering))
        if (all(math.isfinite(value) for value in (vx, vy, wz, pose_age, delay, floor))
                and (valid_work_sec is None or math.isfinite(valid_work_sec))):
            self.stage_certificate_context = context
        def age_error(command):
            return additional_delay_clearance_error(
                command, delay, deceleration, self.clearance.radius,
                time.monotonic()-entered)
        def sweep(command, trace):
            # Independent envelopes preserve the staged/reverse floors. This
            # is not a proof of an arbitrary queued transition or actuator.
            command = np.asarray(command, dtype=float)
            speed, rate = math.hypot(command[0], command[1]), abs(command[2])
            if (not np.isfinite(command).all() or not math.isfinite(delay) or delay < 0.
                    or not math.isfinite(deceleration) or deceleration <= 0.
                    or not math.isfinite(angular_deceleration) or angular_deceleration <= 0.):
                raise ValueError('invalid staged braking inputs')
            travel = (delay*(speed+self.clearance.radius*rate)
                      + .5*speed*(speed/deceleration)
                      + .5*self.clearance.radius*rate*(rate/angular_deceleration))
            if not math.isfinite(travel) or travel/.005 > 4096:
                raise ValueError('staged braking envelope exceeds work budget')
            # The enclosing-body travel bound can certify open regions with
            # one exact scalar query. Recovery needs the actual endpoint.
            age_reserve = age_error(command)
            target_floor = floor+age_reserve
            # Captured source lifetimes bound the query headroom that could be
            # useful after later command-search work. Acceptance still uses
            # actual elapsed age. Unstamped standalone fixtures have no such
            # lifetime and explicitly retain their current-age query policy.
            query_age_reserve = age_reserve
            if valid_work_sec is not None:
                query_age_reserve = max(age_reserve,
                    additional_delay_clearance_error(
                        command, delay, deceleration, self.clearance.radius,
                        valid_work_sec))
            trace.update(total_combined_travel_m=float(travel),
                         margin_m=float(target_floor),
                         query_headroom_m=float(query_age_reserve-age_reserve),
                         current_clearance_m=float(start_clearance),
                         current_query_cap_m=.35,
                         current_clearance_is_exact=bool(start_clearance < .35))
            if not recovering:
                # An unsaturated initial query is the exact distance at this
                # immutable pose; reuse it for every requested scale.
                current = (start_clearance if start_clearance < .35 else
                           self.clearance.body_clearance(
                               pose[:2], pose[2],
                               cap=floor+query_age_reserve+travel+.01))
                if not math.isfinite(current) or current < 0.:
                    raise ValueError('invalid staged footprint clearance')
                endpoint_bound = current-travel
                nominal_min = endpoint_bound-.00250000001
                trace.update(current_clearance_m=float(current),
                             current_query_cap_m=(.35 if start_clearance < .35 else
                                                  float(floor+query_age_reserve+travel+.01)))
                trace['current_clearance_is_exact'] = bool(
                    current < trace['current_query_cap_m'])
                if nominal_min >= floor+max(query_age_reserve, age_error(command)):
                    trace.update(proof_kind='cheap', complete_bound_m=float(nominal_min),
                                 endpoint_bound_m=float(endpoint_bound),
                                 sampled_query_cap_m=None, rejected_sample_bound_m=None)
                    return True, endpoint_bound, nominal_min
            positions, yaws, error, gaps = braking_pose_path(
                pose, command, delay, deceleration, angular_deceleration,
                self.clearance.radius, spacing=.005, max_intervals=4096)
            reserve = .5*float(np.max(gaps, initial=0.))
            # A failing bound rejects certification without scanning the
            # remaining poses. An accepted candidate must check every chunk.
            endpoint_bound, nominal_min = None, math.inf
            cap = max(floor+query_age_reserve+reserve+.005,
                      start_clearance+.005+query_age_reserve if recovering else 0.)+float(error[-1])
            if not math.isfinite(cap):
                raise ValueError('staged clearance query cap must be finite')
            if not recovering and start_clearance < .35:
                # The full minimum includes the exact initial distance. This
                # limiter preserves that minimum, but is unsuitable for the
                # separate recovery endpoint-improvement proof.
                initial_cap = start_clearance+float(error[-1])
                if initial_cap > 0.:
                    cap = min(cap, initial_cap)
            trace.update(sampled_query_cap_m=float(cap))
            for first in range(0, len(yaws), 24):
                last = first+24
                margins = np.asarray(self.clearance.clearance_over_poses(
                    positions[first:last], yaws[first:last], cap=cap), dtype=float)
                if (margins.shape != yaws[first:last].shape
                        or not np.isfinite(margins).all() or np.any(margins < 0.)):
                    raise ValueError('invalid staged footprint clearances')
                bounds = margins-error[first:last]
                if np.min(bounds) < target_floor+reserve:
                    trace.update(proof_kind='sampled_rejected', complete_bound_m=None,
                                 endpoint_bound_m=None,
                                 rejected_sample_bound_m=float(np.min(bounds)-reserve))
                    return False, None, None
                nominal_min = min(nominal_min, float(np.min(bounds))-reserve)
                endpoint_bound = float(bounds[-1])
            trace.update(proof_kind='sampled', complete_bound_m=float(nominal_min),
                         endpoint_bound_m=endpoint_bound, rejected_sample_bound_m=None)
            return nominal_min-age_error(command) >= floor, endpoint_bound, nominal_min

        # Requested outward motion cannot cancel measured inward momentum.
        measured_trace = {}
        context['measured_certificate'] = measured_trace
        measured_safe, _, measured_min = sweep(measured, measured_trace)
        if not measured_safe:
            self.stage_blocked = 'MEASURED_BRAKING_SWEEP_BLOCKED'
            return 0., 0., 0.
        request = np.array([vx, vy, wz])
        requested_trace = {}
        context['requested_certificate'] = requested_trace
        clear, endpoint, nominal_min = sweep(request, requested_trace)
        candidates = [(1., request.copy(), nominal_min, endpoint, requested_trace)] if clear else []
        if not clear:
            low, high = 0., 1.
            for _ in range(7):
                middle = .5*(low+high)
                candidate = middle*request
                candidate_trace = {}
                candidate_safe, candidate_end, candidate_min = sweep(candidate, candidate_trace)
                if candidate_safe:
                    low = middle
                    candidates.append((middle, candidate, candidate_min, candidate_end, candidate_trace))
                else:
                    high = middle
        # The wrapper checks the complete raw/request minima and recovery's
        # endpoint improvement using the same final age immediately on return.
        return dict(measured_min=measured_min, candidates=candidates,
                    delay=delay, deceleration=deceleration, floor=floor,
                    recovering=recovering, start_clearance=start_clearance)
