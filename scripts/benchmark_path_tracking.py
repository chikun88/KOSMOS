#!/usr/bin/env python3
"""Reproducible production-controller replay in an empty, simulated workspace.

Loads the production numerical code without importing or mocking ROS modules.
The tracker builds a fixed curved route once; a planner heartbeat refreshes its
age without replacing it with a straight line to the goal. Real tracker tick,
RuntimeGuard, bridge calibration and UART saturation execute at their configured
limits. This does not run Nav2, localization, collision monitoring or hardware.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path('ros2_ws/src/omni_autonomy_next')


def polyline_projection(points, positions):
    """Independent exact point-to-segment metric; never uses tracker.project()."""
    points, positions = np.asarray(points, float), np.asarray(positions, float)
    if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 1
            or positions.ndim != 2 or positions.shape[1:] != (2,)
            or not np.isfinite(points).all() or not np.isfinite(positions).all()):
        raise ValueError('positions and nonempty path must be finite (N, 2) arrays')
    if len(points) == 1:
        return np.linalg.norm(positions - points[0], axis=1), np.zeros(len(positions))
    segments = np.diff(points, axis=0)
    lengths2 = np.sum(segments * segments, axis=1)
    lengths = np.sqrt(lengths2)
    arc = np.r_[0., np.cumsum(lengths)]
    distances, progress = [], []
    # Bound peak memory for long dense paths.
    for position in positions:
        fractions = np.divide(np.sum((position-points[:-1])*segments, axis=1),
                              lengths2, out=np.zeros(len(segments)), where=lengths2 > 0.)
        fractions = np.clip(fractions, 0., 1.)
        residual = position - (points[:-1] + fractions[:, None]*segments)
        norms = np.linalg.norm(residual, axis=1)
        index = int(np.argmin(norms))
        distances.append(norms[index])
        progress.append(arc[index]+fractions[index]*lengths[index])
    return np.asarray(distances), np.asarray(progress)


def sustained_arrival(times, positions, velocities, goal, hold_sec=.5):
    """Require every observation through the whole hold interval, including end."""
    times = np.asarray(times, float)
    positions, velocities, goal = map(lambda value: np.asarray(value, float),
                                      (positions, velocities, goal))
    if (times.ndim != 1 or not len(times) or positions.shape != (len(times), 3)
            or velocities.shape != positions.shape or goal.shape != (3,)
            or not all(np.isfinite(v).all() for v in (times, positions, velocities, goal))
            or np.any(np.diff(times) <= 0.) or not math.isfinite(hold_sec) or hold_sec <= 0.):
        raise ValueError('arrival samples require finite, strictly increasing time and poses')
    errors = np.linalg.norm(positions[:, :2]-goal[:2], axis=1)
    yaws = np.abs((positions[:, 2]-goal[2]+math.pi) % (2*math.pi)-math.pi)
    settled = ((errors <= .015) & (yaws <= .015)
               & (np.linalg.norm(velocities[:, :2], axis=1) <= .025)
               & (np.abs(velocities[:, 2]) <= .025))
    started = None
    for index, valid in enumerate(settled):
        started = (index if started is None else started) if valid else None
        if started is not None and times[index]-times[started] >= hold_sec-1.e-10:
            return float(times[started])
    return None



def integrate_plant(pose, velocity, target, dt, tau):
    """Independent first-order plant: exact velocity/yaw, Simpson XY integral.

    Evaluating the world twist at start/midpoint/end removes the centimetre
    bias of endpoint Euler integration at high speed. No production predictor
    is called to generate the simulated ground truth.
    """
    pose, velocity, target = (np.asarray(v, float) for v in (pose, velocity, target))
    difference = velocity-target
    def sample(at):
        response = target+difference*math.exp(-at/tau)
        yaw = pose[2]+target[2]*at+difference[2]*tau*(-math.expm1(-at/tau))
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c*response[0]-s*response[1], s*response[0]+c*response[1]])
    result = pose.copy()
    result[:2] += dt/6.*(sample(0.)+4.*sample(dt/2.)+sample(dt))
    result[2] += target[2]*dt+difference[2]*tau*(-math.expm1(-dt/tau))
    return result, target+difference*math.exp(-dt/tau)


def load_controller(source_root):
    """Remove transport-only imports in a private namespace, never sys.modules mocks."""
    package = source_root / PACKAGE
    sys.path.insert(0, str(package))
    source = package / 'omni_autonomy_next/trajectory_tracker_node.py'
    tree = ast.parse(source.read_text())
    excluded = {'rclpy', 'geometry_msgs', 'nav_msgs', 'std_msgs'}
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for item in tree.body:
        if isinstance(item, ast.Import) and any(a.name.split('.')[0] in excluded for a in item.names):
            continue
        if isinstance(item, ast.ImportFrom) and (
                (item.module or '').split('.')[0] in excluded or item.module == 'staged_heading_node'):
            continue
        if isinstance(item, ast.If) and '__name__' in ast.unparse(item.test):
            continue  # never invoke the executable entrypoint
        body.append(item)
    imported = sys.modules.get('omni_autonomy_next')
    if imported is not None and Path(imported.__file__).resolve().parent != (package / 'omni_autonomy_next').resolve():
        raise RuntimeError('Use separate processes for different source roots; dependencies are already imported')
    module = ModuleType('omni_autonomy_next._offline_tracker')
    module.__package__ = 'omni_autonomy_next'
    module.__file__ = str(source)
    module.Node = type('UninitializedTransport', (), {})
    module.StagedHeadingMixin = type('UnusedStagedTransport', (), {})
    module.Twist = lambda: SimpleNamespace(linear=SimpleNamespace(x=0., y=0.),
                                          angular=SimpleNamespace(z=0.))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), 'exec'),
         vars(module))
    clock = SimpleNamespace(now=0.)
    module.time = SimpleNamespace(monotonic=lambda: clock.now)
    defaults = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'defaults' for t in n.targets))
    bridge_file = package / 'omni_autonomy_next/motor_udp_bridge_node.py'
    bridge_method = next(n for n in ast.walk(ast.parse(bridge_file.read_text()))
                         if isinstance(n, ast.FunctionDef) and n.name == '_velocity_from_latest_twist')
    bridge_namespace = {'math': math, 'ProtocolError': ValueError}
    exec(compile(ast.Module(body=[bridge_method], type_ignores=[]), str(bridge_file), 'exec'), bridge_namespace)
    return module, clock, ast.literal_eval(defaults), bridge_namespace['_velocity_from_latest_twist']


def make_runtime(module, defaults, source_root, points, profile, goal_yaw, geometry='none'):
    from omni_autonomy_next.config import load_robot, calibrated_tracking_parameters
    from omni_autonomy_next.runtime_guard import RuntimeGuard, MotionLimits
    config = source_root / PACKAGE / 'config'
    robot = load_robot(str(config / 'robot.yaml'))
    raw_robot = yaml.safe_load((config / 'robot.yaml').read_text())['robot']
    drive = {**raw_robot['drivetrain'], **robot['drivetrain']}
    profiles = json.loads(yaml.safe_load((config / 'runtime.yaml').read_text())['runtime_guard']['ros__parameters']['profiles_json'])
    if profile not in profiles:
        raise ValueError(f'unknown configured profile: {profile}')
    runtime = yaml.safe_load((config / 'runtime.yaml').read_text())['runtime_guard']['ros__parameters']
    smoother = yaml.safe_load((config / 'nav2_next.yaml').read_text())['velocity_smoother']['ros__parameters']
    params = dict(defaults)
    params.update(calibrated_tracking_parameters(raw_robot, hardware=True, motion_mode='simultaneous'))
    # This is an obstacle-free path geometry benchmark. No clearance model is
    # substituted for the real field and no faster geographic mode is enabled.
    node = object.__new__(module.TrajectoryTracker)
    values = dict(trajectory=None, pose=np.array([*points[0], 0.]), pose_stamp=0.,
        reference_time=0., plan_stamp=0., active_goal=np.array([*points[-1], goal_yaw]),
        lock=threading.Lock(), velocity=np.zeros(3), raw_velocity=np.zeros(3),
        speed_scale=1., period=1./max(20., params['control_rate_hz']), velocity_stamp=0.,
        odometry_paused=False, best_distance=math.inf, best_distance_at=0., terminal_since=None,
        terminal_best_yaw=math.inf, tracked_endpoint=None, command=np.zeros(3), finished_at=None,
        acceleration=float(smoother['max_accel'][0]), default_acceleration=float(smoother['max_accel'][0]),
        deceleration=abs(float(smoother['max_decel'][0])), yaw_acceleration=float(smoother['max_accel'][2]),
        lateral_acceleration=params['lateral_acceleration'], envelope=module.OmniEnvelope(drive),
        default_max_wheel_speed=drive['max_wheel_speed'],
        profile_max_wheel_speeds=drive.get('profile_max_wheel_speeds', {}),
        profile_linear_accelerations=drive.get('profile_linear_accelerations', {}),
        profiles=json.loads(runtime['profiles_json']), plan_build_ms=0., snap_offset=0., snapped=True,
        motion_mode='simultaneous', clearance=None, monitor_horizon=None, fixed_gate_speed=None,
        statuses=[], status_events=[], pending_plan=None, last_plan=None, plan_event=threading.Event(),
        command_pub=SimpleNamespace(publish=lambda message: None),
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        get_logger=lambda: SimpleNamespace(warning=lambda *a, **kw: None),
        _relay_behavior=lambda *args: False, _track_flow=lambda *args: None,
        _flow_gap=lambda: 0., _flow_gain=lambda: 1.)
    node.__dict__.update(values)
    if geometry == 'open_box':
        box = np.array([[-30., -30.], [30., -30.], [30., 30.], [-30., 30.]])
        node.clearance = module.CadClearanceModel(np.stack((box, np.roll(box, -1, axis=0)), axis=1),
                                                  footprint=robot['footprint'])
        node.monitor_horizon = module.load_collision_monitor_horizon(str(config / 'nav2_next.yaml'))
    def record_status(state, **kwargs):
        node.statuses.append(state)
        node.status_events.append((module.time.monotonic(), state))
    node._status = record_status
    node._apply_profile(profile)
    hard = MotionLimits(*(runtime['hard_max_'+key] for key in (
        'linear_speed', 'lateral_speed', 'angular_speed', 'linear_acceleration',
        'angular_acceleration', 'linear_jerk', 'angular_jerk')))
    guard = RuntimeGuard(profiles={k: MotionLimits(**v) for k, v in node.profiles.items()},
        hard_limits=hard, default_profile=profile, command_timeout_sec=runtime['command_timeout_sec'],
        red_zone_speed_scale=runtime['red_zone_speed_scale'], wheel_radius=drive['wheel_radius'],
        wheel_positions=drive['wheel_positions'], wheel_drive_angles_rad=np.radians(drive['wheel_drive_angles_deg']),
        wheel_signs=drive['wheel_signs'], max_wheel_speed=drive['max_wheel_speed'],
        profile_max_wheel_speeds=drive.get('profile_max_wheel_speeds', {}),
        translation_budget_share=drive.get('translation_budget_share', .45))
    bridge = SimpleNamespace(latest_twist=module.Twist(), linear_x_sign=1., linear_y_sign=1., angular_z_sign=1.,
        max_linear_speed=max(hard.linear, hard.lateral), max_angular_speed=hard.angular,
        linear_command_scale=drive['linear_command_scale'], angular_command_scale=drive['angular_command_scale'])
    return node, guard, bridge, params


def paths():
    x = np.linspace(0., 8., 161)
    theta = np.linspace(0., math.pi/2, 161)
    return {
        'straight_12m': np.array([[0., 0.], [12., 0.]]),
        'straight_24m': np.array([[0., 0.], [24., 0.]]),
        'diagonal_12m': np.array([[0., 0.], [12./math.sqrt(2), 12./math.sqrt(2)]]),
        'quarter_circle_r3': np.c_[3.*np.sin(theta), 3.*(1.-np.cos(theta))],
        's_curve_8m': np.c_[x, .7*np.sin(2.*math.pi*x/8.)],
    }


def reference_metrics(module, trajectory):
    # Avoid vertices: the certified position path is piecewise linear. Report
    # feedforward consistency inside each segment, not an artificial derivative
    # across a nondifferentiable corner or a wrapped yaw discontinuity.
    intervals = np.diff(trajectory.time)
    times = np.concatenate([trajectory.time[:-1][intervals > 1.e-5]
                            + fraction*intervals[intervals > 1.e-5] for fraction in (.2, .5, .8)])
    linear, angular, durations = [], [], []
    for at in times:
        delta = min(1.e-6, trajectory.duration*1.e-6)
        before, after = trajectory.sample(at-delta), trajectory.sample(at+delta)
        started = time.perf_counter_ns()
        sample = trajectory.sample(at)
        durations.append((time.perf_counter_ns()-started)*1.e-6)
        linear.append(float(np.linalg.norm((after[0]-before[0])/(2*delta)-sample[1])))
        angular.append(abs(module.wrap(after[2]-before[2])/(2*delta)-sample[3]))
    lengths = np.linalg.norm(np.diff(trajectory.points, axis=0), axis=1)
    acceleration = np.divide(np.abs(np.diff(trajectory.speed**2)), 2.*lengths,
                             out=np.zeros_like(lengths), where=lengths > 1.e-12)
    return dict(linear_derivative_error_max_m_s=max(linear, default=0.),
                planned_discrete_lateral_acceleration_max_m_s2=float(np.max(
                    module.menger_curvature(trajectory.points)*trajectory.speed**2)),
                planned_tangential_acceleration_max_m_s2=float(np.max(acceleration, initial=0.)),
                yaw_derivative_error_max_rad_s=max(angular, default=0.),
                sample_wall_p50_ms=float(np.percentile(durations, 50)) if durations else 0.,
                sample_wall_p99_ms=float(np.percentile(durations, 99)) if durations else 0.,
                planned_peak_speed_m_s=float(np.max(trajectory.speed)),
                planned_duration_s=trajectory.duration, path_length_m=trajectory.length)


def replay(module, clock, defaults, bridge_velocity, root, points, profile, delay, tau, duration, goal_yaw, geometry='none'):
    from omni_autonomy_next.runtime_guard import GuardHealth
    from omni_autonomy_next.robomas_uart import mix_velocity
    node, guard, bridge, params = make_runtime(module, defaults, root, points, profile, goal_yaw, geometry)
    clock.now = 0.
    node._build_trajectory(points.copy(), goal_yaw)
    if node.trajectory is None:
        raise RuntimeError('empty-space benchmark could not build trajectory')
    trajectory = node.trajectory
    reference = reference_metrics(module, trajectory)
    dt = .01
    actual, truth = np.zeros(3), node.pose.copy()
    queue = deque(np.zeros(3) for _ in range(round(delay/dt)))
    # Nominal inverse calibration isolates controller error. Gain uncertainty
    # is an explicit scenario, never described as measured physical behavior.
    gains = np.array([1./bridge.linear_command_scale]*2+[1./bridge.angular_command_scale])
    rows, commands, timings, saturations, wheels, measured_bounds = [], [], [], [], [], []
    for step in range(round(duration/dt)):
        now = step*dt
        clock.now = now
        node.pose = truth.copy()
        node.pose_stamp = node.velocity_stamp = now
        node.raw_velocity = actual.copy()
        node.velocity += (1.-math.exp(-dt/params['velocity_filter_sec']))*(actual-node.velocity)
        node.plan_received_stamp = math.floor(now)  # planner heartbeat, immutable route
        if step % round(node.period/dt) == 0:
            entered = time.perf_counter_ns()
            node._tick()
            timings.append((time.perf_counter_ns()-entered)*1.e-6)
        safe = guard.step(node.command, now_sec=now, command_age_sec=now-node.last_control_tick,
            health=GuardHealth(True, False, True, True, True), profile=profile, user_scale=1., red_zone=False)
        command = np.array(safe.velocity)
        bridge.latest_twist.linear.x, bridge.latest_twist.linear.y = command[:2]
        bridge.latest_twist.angular.z = command[2]
        wire = np.asarray(bridge_velocity(bridge))
        wheel, saturation = mix_velocity(*wire)
        queue.append(wire*saturation*gains)
        truth, actual = integrate_plant(truth, actual, queue.popleft(), dt, tau)
        rows.append([now+dt, *truth, *actual])
        if geometry == 'open_box':
            footprint = node.clearance.footprint
            world_x = truth[0]+footprint[:, 0]*math.cos(truth[2])-footprint[:, 1]*math.sin(truth[2])
            world_y = truth[1]+footprint[:, 0]*math.sin(truth[2])+footprint[:, 1]*math.cos(truth[2])
            measured_bounds.append(float(min(30.-np.max(np.abs(world_x)), 30.-np.max(np.abs(world_y)))))
        commands.append(command)
        saturations.append(saturation)
        wheels.append(max(map(abs, wheel)))
    data, commands = np.asarray(rows), np.asarray(commands)
    cross, progress = polyline_projection(trajectory.points, data[:, 1:3])
    input_cross, _ = polyline_projection(points, data[:, 1:3])
    speed = np.linalg.norm(data[:, 4:6], axis=1)
    moving = speed > .05
    cruise = speed[(progress > .3*trajectory.length) & (progress < .7*trajectory.length)]
    goal_error = np.linalg.norm(data[:, 1:3]-node.active_goal[:2], axis=1)
    arrival = sustained_arrival(data[:, 0], data[:, 1:4], data[:, 4:7], node.active_goal)
    return dict(reference=reference, arrival_s=arrival, simulated_duration_s=float(data[-1, 0]),
        peak_speed_m_s=float(speed.max()), cruise_median_m_s=float(np.median(cruise)) if len(cruise) else None,
        cross_track_peak_m=float(cross.max()), cross_track_p95_m=float(np.percentile(cross[moving], 95)) if moving.any() else 0.,
        cross_track_rms_m=float(np.sqrt(np.mean(cross[moving]**2))) if moving.any() else 0.,
        input_path_distance_peak_m=float(input_cross.max()), final_goal_error_m=float(goal_error[-1]),
        final_yaw_error_rad=abs(module.wrap(float(data[-1, 3])-goal_yaw)),
        final_speed_m_s=float(speed[-1]), tail_1s_goal_error_max_m=float(goal_error[-100:].max()),
        tick_wall_p50_ms=float(np.percentile(timings, 50)), tick_wall_p99_ms=float(np.percentile(timings, 99)),
        tick_wall_max_ms=float(max(timings)), uart_peak_units=int(max(wheels)),
        uart_saturation_min=float(min(saturations)),
        command_acceleration_peak_m_s2=float(np.linalg.norm(np.diff(commands[:, :2], axis=0)/dt, axis=1).max()),
        terminal_states=sorted(set(node.statuses)-{'TRACKING', 'TERMINAL'}),
        controller_failures_before_arrival=sorted({state for at, state in node.status_events
            if state not in ('TRACKING', 'TERMINAL', 'IDLE')
            and (arrival is None or at <= arrival+.5)}),
        predictive_cruise_eligible=bool(getattr(trajectory, 'sprint_cruise_allowed', False)),
        execution_certificate_used=hasattr(node, 'execution_certificate_context'),
        last_execution_failure=getattr(node, 'execution_clearance_reason', None),
        sampled_footprint_clearance_min_m=min(measured_bounds) if measured_bounds else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', nargs='+', choices=['none', 'open_box'], default=['none', 'open_box'])
    parser.add_argument('--profiles', nargs='+', default=['sprint'])
    parser.add_argument('--paths', nargs='+', choices=list(paths()), default=list(paths()))
    parser.add_argument('--delays', type=float, nargs='+', default=[.12, .20, .30])
    parser.add_argument('--tau', type=float, default=.12)
    parser.add_argument('--duration', type=float, default=24.)
    parser.add_argument('--goal-yaw', type=float, default=0.)
    parser.add_argument('--require-arrival', action='store_true')
    args = parser.parse_args()
    if (any(not math.isfinite(x) or x < 0. for x in args.delays)
            or not all(math.isfinite(x) and x > 0. for x in (args.tau, args.duration))
            or args.duration < .51 or not math.isfinite(args.goal_yaw)
            or any(abs(v/.01-round(v/.01)) > 1.e-7 for v in args.delays)):
        parser.error('finite positive tau/duration >= 0.51 s and delays in nonnegative 10 ms increments required')
    root = args.source_root.resolve()
    sources = sorted((root/PACKAGE/'omni_autonomy_next').glob('*.py'))
    sources += [root/PACKAGE/'config'/f for f in ('robot.yaml', 'runtime.yaml', 'nav2_next.yaml')]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    module, clock, defaults, bridge_velocity = load_controller(root)
    cases = []
    for profile in args.profiles:
        for name in args.paths:
            for geometry in args.geometry:
                for delay in args.delays:
                    result = replay(module, clock, defaults, bridge_velocity, root, paths()[name],
                                    profile, delay, args.tau, args.duration, args.goal_yaw, geometry)
                    cases.append(dict(path=name, profile=profile, geometry=geometry, delay_s=delay, tau_s=args.tau,
                                      goal_yaw_rad=args.goal_yaw, result=result))
                    print(f'{profile} {geometry} {name} delay={delay:.2f}: peak={result["peak_speed_m_s"]:.3f} m/s '
                          f'cross={result["cross_track_peak_m"]*1000:.2f} mm arrival={result["arrival_s"]}', flush=True)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))] for p in sources):
        raise RuntimeError('Source changed during benchmark; rerun on a fixed snapshot')
    report = dict(kind='production numerical-controller empty-space replay', schema_version=1,
        source_root=str(root), source_hashes=hashes,
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         platform=platform.platform(), cpu_count=os.cpu_count(),
                         cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        all_arrived=all(c['result']['arrival_s'] is not None
                        and not c['result']['controller_failures_before_arrival'] for c in cases),
        limitations=['No ROS executor, Nav2, localization, sensor noise or live collision monitor.',
                     'none geometry omits CAD; open_box runs real CAD and execution certificates against a synthetic 60 m square.',
                     'Simulation time does not advance during calculations; computation-age expiry and scheduling deadlines are not simulated.',
                     'Delay/first-order response and inverse calibration are simulated assumptions, not physical identification.',
                     'Uses simultaneous mode and a fixed generated path; not staged-heading or all-field acceptance.',
                     'Control continues after arrival for tail measurement; post-arrival watchdog states are reported separately from pre-arrival failures.',
                     'UART saturation is enforced; calibrated response at high speed is unverified.',
                     'Wall-clock timing includes execution certificates only in open_box cases and is not a real-time guarantee.',
                     'Nearest-polyline cross-track does not prove branch order on self-intersecting routes.'],
        arrival_contract=dict(position_m=.015, yaw_rad=.015, linear_speed_m_s=.025,
                              angular_speed_rad_s=.025, continuous_hold_s=.5), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return 1 if args.require_arrival and not report['all_arrived'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
