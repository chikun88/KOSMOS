"""Conservative CAD clearance of an independent delayed braking envelope.

The envelope holds one body-frame translation direction and signed yaw rate
through the delay, then reduces their magnitudes linearly to zero. It does not
model a queued transition between commands or certify an actuator's response.
"""
import math

import numpy as np


_MAX_INTERVALS = 200000
_RUNTIME_MAX_INTERVALS = 2048


def _finite_scalar(value, name, *, positive=False, nonnegative=False):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be finite') from error
    if (not math.isfinite(value) or (positive and value <= 0.)
            or (nonnegative and value < 0.)):
        raise ValueError(f'{name} has an invalid value')
    return value


def additional_delay_clearance_error(command, delay_sec, linear_deceleration,
                                     radius, elapsed_sec):
    """Bound body-point displacement caused by an uncertain extra delay.

    At matching braking progress the longer-delay path is the nominal path
    left-transformed by the command's extra constant-twist prefix. The nominal
    translation arclength is ``S``; every body point's lever arm is at most
    ``S + radius``. The bound also covers the extra prefix against the initial
    body. Curved longer-delay paths need not contain the nominal swept path.
    """
    try:
        command = np.asarray(command, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('command must be a finite planar triple') from error
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError('command must be a finite planar triple')
    delay = _finite_scalar(delay_sec, 'delay', nonnegative=True)
    linear = _finite_scalar(linear_deceleration, 'linear deceleration', positive=True)
    radius = _finite_scalar(radius, 'footprint radius', nonnegative=True)
    elapsed = _finite_scalar(elapsed_sec, 'elapsed time', nonnegative=True)
    speed, rate = math.hypot(command[0], command[1]), abs(float(command[2]))
    arclength = speed*delay + .5*speed*(speed/linear)
    prefix_translation, prefix_angle = speed*elapsed, rate*elapsed
    lever = arclength + radius
    error = prefix_translation + min(2., prefix_angle)*lever
    if not all(math.isfinite(value) for value in (
            speed, arclength, prefix_translation, prefix_angle, lever, error)):
        raise ValueError('additional-delay displacement bound must be finite')
    return error


def braking_pose_path(pose, command, delay_sec, linear_deceleration,
                      angular_deceleration, radius, spacing=.01, *,
                      max_intervals=_MAX_INTERVALS):
    """Return XY samples, exact yaws, cumulative XY error, and interval travel.

    Every interval has at most ``spacing`` of combined body travel: translation
    arclength plus ``radius`` times angular travel. Samples include the delay
    and both stopping boundaries. XY integration uses a speed-weighted mean
    heading, with a cumulative ``ds * dtheta**2 / 8`` error bound. Constant
    twist during the delay instead uses its exact circular arc.

    ``radius`` must enclose the complete footprint about the pose's origin.
    The mathematical bound is conditional on this independent braking model.
    At most ``max_intervals`` intervals and ``max_intervals + 1`` retained
    poses are generated, including all phase boundaries.
    """
    try:
        pose = np.asarray(pose, dtype=float)
        command = np.asarray(command, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('pose and command must be finite planar triples') from error
    if (pose.shape != (3,) or command.shape != (3,)
            or not np.isfinite(pose).all() or not np.isfinite(command).all()):
        raise ValueError('pose and command must be finite planar triples')
    delay = _finite_scalar(delay_sec, 'delay', nonnegative=True)
    deceleration = _finite_scalar(linear_deceleration, 'linear deceleration', positive=True)
    angular = _finite_scalar(angular_deceleration, 'angular deceleration', positive=True)
    radius = _finite_scalar(radius, 'footprint radius', nonnegative=True)
    spacing = _finite_scalar(spacing, 'spacing', positive=True)
    if spacing > .01:
        raise ValueError('combined-travel spacing must not exceed 0.01 m')
    budget = _finite_scalar(max_intervals, 'sample budget', positive=True)
    if budget != int(budget) or budget > _MAX_INTERVALS:
        raise ValueError('sample budget must be an integer no larger than 200000')
    budget = int(budget)
    speed = math.hypot(command[0], command[1])
    rate = abs(float(command[2]))
    linear_duration, angular_duration = speed / deceleration, rate / angular
    linear_stop, angular_stop = delay + linear_duration, delay + angular_duration
    if not all(math.isfinite(v) for v in (speed, linear_stop, angular_stop)):
        raise ValueError('braking horizon must be finite')
    if speed == 0. and rate == 0.:
        return pose[None, :2].copy(), pose[2:3].copy(), np.zeros(1), np.empty(0)
    direction = command[:2] / speed if speed else np.zeros(2)
    sign = math.copysign(1., command[2])
    boundaries = sorted(set((0., delay, linear_stop, angular_stop)))
    positions, yaws, errors, travel = [pose[None, :2].copy()], [pose[2:3].copy()], [np.zeros(1)], []
    current = pose[:2].copy()
    error = 0.
    intervals = 0

    def heading(at):
        braking_time = np.minimum(np.maximum(at - delay, 0.), angular_duration)
        return (float(pose[2]) + command[2] * np.minimum(at, delay)
                + sign * (rate - .5 * angular * braking_time) * braking_time)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        duration = end - start
        if start < delay:
            v0, v1, w0, w1 = speed, 0., float(command[2]), 0.
        else:
            v0 = max(0., speed - deceleration * (start - delay))
            v1 = -deceleration if start < linear_stop else 0.
            w0 = sign * max(0., rate - angular * (start - delay))
            w1 = -sign * angular if start < angular_stop else 0.
        b = v0 + radius * abs(w0)
        k = -v1 + radius * abs(w1)
        amount = (b - .5 * k * duration) * duration
        if not all(math.isfinite(v) for v in (b, k, amount, b*b)) or amount < 0.:
            raise ValueError('combined braking travel must be finite and nonnegative')
        sample_count = amount / spacing
        if not math.isfinite(sample_count) or sample_count > budget - intervals:
            raise ValueError('braking envelope exceeds the bounded sample budget')
        count = max(1, math.ceil(sample_count))
        if count > budget - intervals:
            raise ValueError('braking envelope exceeds the bounded sample budget')
        if amount == 0.:
            offsets = np.array([0., duration])
        else:
            arcs = np.linspace(0., amount, count + 1)
            if k == 0.:
                offsets = arcs / b
            else:
                # Stable inverse of b*u - k*u**2/2, including a stop at u=b/k.
                offsets = 2. * arcs / (b + np.sqrt(np.maximum(0., b*b - 2.*k*arcs)))
            offsets[-1] = duration
        phase_yaw = heading(start)
        left, h = offsets[:-1], np.diff(offsets)
        v = np.maximum(0., v0 + v1*left)
        w = w0 + w1*left
        ds = np.maximum(0., v*h + .5*v1*h*h)
        dtheta = w*h + .5*w1*h*h
        theta = phase_yaw + w0*left + .5*w1*left*left
        if start < delay:
            half = .5*dtheta
            mu = theta + half
            displacement = ds*np.sinc(half/np.pi)
            increments = np.zeros_like(ds)
        else:
            # Group powers with their coefficients to avoid h**4 overflow
            # for a long, very slow finite braking interval.
            v_h, dv_h2 = v*h, v1*h*h
            w_h, dw_h2 = w*h, w1*h*h
            moment = (w_h*(v_h/2. + dv_h2/3.)
                      + .5*dw_h2*(v_h/3. + dv_h2/4.))
            mu = theta + np.divide(moment, ds, out=np.zeros_like(ds), where=ds > 0.)
            displacement = ds
            # The weighted mean cancels the first Taylor term; its variance
            # is at most dtheta**2/4 for monotone yaw. Keep cumulative errors.
            increments = ds*dtheta*dtheta/8.
        c, s = np.cos(mu), np.sin(mu)
        delta = displacement[:, None]*np.column_stack((
            c*direction[0] - s*direction[1], s*direction[0] + c*direction[1]))
        phase_positions = current + np.cumsum(delta, axis=0)
        phase_errors = error + np.cumsum(increments)
        positions.append(phase_positions)
        yaws.append(heading(start + offsets[1:]))
        errors.append(phase_errors)
        travel.append(ds + radius*np.abs(dtheta))
        current, error = phase_positions[-1], float(phase_errors[-1])
        intervals += len(h)
    positions, yaws = np.concatenate(positions), np.concatenate(yaws)
    errors = np.concatenate(errors)
    travel = np.concatenate(travel) if travel else np.empty(0)
    if not (np.isfinite(positions).all() and np.isfinite(yaws).all()
            and np.isfinite(errors).all() and np.isfinite(travel).all()):
        raise ValueError('braking envelope geometry must be finite')
    return positions, yaws, errors, travel


def delayed_braking_clearance(model, pose, command, delay_sec,
                             linear_deceleration, angular_deceleration,
                             margin=.025, spacing=.01):
    """Return a lower footprint-clearance bound, including sweep reserves.

    ``margin`` is the caller's comparison threshold, not an extra subtraction.
    It sets the query's saturation cap. ``model`` supplies a circumscribed
    ``radius`` and exact, cap-saturated ``clearance_over_poses`` distances.
    """
    margin = _finite_scalar(margin, 'margin', nonnegative=True)
    spacing = _finite_scalar(spacing, 'spacing', positive=True)
    try:
        radius = _finite_scalar(model.radius, 'footprint radius', nonnegative=True)
    except AttributeError as error:
        raise ValueError('model must supply a circumscribed footprint radius') from error
    points, yaws, error, travel = braking_pose_path(
        pose, command, delay_sec, linear_deceleration, angular_deceleration, radius, spacing)
    reserve = radius * .02 + max(.005, .5 * float(np.max(travel, initial=0.)))
    cap = margin + reserve + float(error[-1]) + spacing
    clearances = np.asarray(model.clearance_over_poses(points, yaws, cap=cap), dtype=float)
    if (clearances.shape != yaws.shape or not np.isfinite(clearances).all()
            or np.any(clearances < 0.)):
        raise ValueError('model must return finite nonnegative footprint clearances')
    return float(np.min(clearances - error)) - reserve


def delayed_braking_certificate(model, pose, command, delay_sec,
                                linear_deceleration, angular_deceleration,
                                margin=.025, spacing=.01, *, query_headroom_m=0.,
                                diagnostics=None):
    """Return ``(safe, complete_lower_bound, rejected_sample_bound)``.

    A cheap acceptance certificate uses present body clearance minus the entire
    possible body travel. Otherwise inspect the same braking path in 24-pose
    chunks. A below-margin sample immediately rejects certification; its bound
    is not a complete-path minimum and does not prove physical collision.
    Runtime work is limited to 2048 intervals (2049 poses including all phase
    boundaries); the total-travel budget is checked before any geometry query.
    ``query_headroom_m`` raises the geometry query caps so complete certificates
    retain additional clearance. A cheap bound must retain that headroom;
    otherwise the sampled path may provide a tighter complete bound. Sampled
    acceptance still uses ``margin``. Headroom does not model additional delay.
    If supplied, ``diagnostics`` is a dict cleared before input validation and
    populated on each normal tuple return with scalar proof context. A rejected
    sample is labelled separately from a complete bound; validation exceptions
    leave the dict empty so a previous proof cannot describe the failed call.
    """
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise ValueError('diagnostics must be a dict')
        diagnostics.clear()
    try:
        pose = np.asarray(pose, dtype=float)
        command = np.asarray(command, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('pose and command must be finite planar triples') from error
    if (pose.shape != (3,) or command.shape != (3,)
            or not np.isfinite(pose).all() or not np.isfinite(command).all()):
        raise ValueError('pose and command must be finite planar triples')
    delay = _finite_scalar(delay_sec, 'delay', nonnegative=True)
    linear = _finite_scalar(linear_deceleration, 'linear deceleration', positive=True)
    angular = _finite_scalar(angular_deceleration, 'angular deceleration', positive=True)
    margin = _finite_scalar(margin, 'margin', nonnegative=True)
    spacing = _finite_scalar(spacing, 'spacing', positive=True)
    headroom = _finite_scalar(query_headroom_m, 'query headroom', nonnegative=True)
    if spacing > .01:
        raise ValueError('combined-travel spacing must not exceed 0.01 m')
    try:
        radius = _finite_scalar(model.radius, 'footprint radius', nonnegative=True)
    except AttributeError as error:
        raise ValueError('model must supply a circumscribed footprint radius') from error
    speed, rate = math.hypot(command[0], command[1]), abs(float(command[2]))
    total_travel = (delay * (speed + radius*rate)
                    + .5*speed*(speed/linear) + .5*radius*rate*(rate/angular))
    reserve = radius*.02 + .005
    current_cap = margin + reserve + total_travel + spacing + headroom
    if not all(math.isfinite(value) for value in (
            speed, delay+speed/linear, delay+rate/angular, total_travel, current_cap)):
        raise ValueError('combined braking travel and clearance cap must be finite')
    sample_count = total_travel / spacing
    if not math.isfinite(sample_count) or sample_count > _RUNTIME_MAX_INTERVALS:
        raise ValueError('braking envelope exceeds the bounded sample budget')
    current = np.asarray(model.body_clearance(pose[:2], pose[2], cap=current_cap), dtype=float)
    if current.shape != () or not np.isfinite(current) or current < 0.:
        raise ValueError('model must return finite nonnegative footprint clearance')
    cheap_bound = float(current) - total_travel - reserve
    context = None
    if diagnostics is not None:
        context = {
            'current_clearance_m': float(current),
            'current_clearance_is_exact': bool(float(current) < current_cap),
            'total_combined_travel_m': total_travel,
            'margin_m': margin,
            'query_headroom_m': headroom,
            'current_query_cap_m': current_cap,
            'sampled_query_cap_m': None,
            'complete_bound_m': None,
            'rejected_sample_bound_m': None,
        }
    if cheap_bound >= margin + headroom:
        if context is not None:
            context.update(proof_kind='cheap', complete_bound_m=cheap_bound)
            diagnostics.update(context)
        return True, cheap_bound, None

    points, yaws, error, travel = braking_pose_path(
        pose, command, delay, linear, angular, radius, spacing,
        max_intervals=_RUNTIME_MAX_INTERVALS)
    # Retain even the rounding excess of the full float API's measured gap.
    reserve = radius*.02 + max(.005, .5*float(np.max(travel, initial=0.)))
    cap = margin + reserve + float(error[-1]) + spacing + headroom
    if not math.isfinite(cap):
        raise ValueError('clearance cap must be finite')
    if float(current) < current_cap:
        # An unsaturated current query is exact. The whole-path minimum cannot
        # exceed its initial sample, so farther samples may be capped at c0+Emax:
        # after subtracting E_i they remain at least c0 and cannot lower that min.
        initial_cap = float(current) + float(error[-1])
        if initial_cap > 0.:
            cap = min(cap, initial_cap)
    if context is not None:
        context['sampled_query_cap_m'] = cap
    complete_bound = math.inf
    for start in range(0, len(yaws), 24):
        stop = start + 24
        clearances = np.asarray(model.clearance_over_poses(
            points[start:stop], yaws[start:stop], cap=cap), dtype=float)
        if (clearances.shape != yaws[start:stop].shape
                or not np.isfinite(clearances).all() or np.any(clearances < 0.)):
            raise ValueError('model must return finite nonnegative footprint clearances')
        sample_bounds = clearances - error[start:stop] - reserve
        violated = sample_bounds[sample_bounds < margin]
        if len(violated):
            rejected_bound = float(violated[0])
            if context is not None:
                context.update(proof_kind='sampled_rejected',
                               rejected_sample_bound_m=rejected_bound)
                diagnostics.update(context)
            return False, None, rejected_bound
        complete_bound = min(complete_bound, float(np.min(sample_bounds)))
    if context is not None:
        context.update(proof_kind='sampled', complete_bound_m=complete_bound)
        diagnostics.update(context)
    return True, complete_bound, None
