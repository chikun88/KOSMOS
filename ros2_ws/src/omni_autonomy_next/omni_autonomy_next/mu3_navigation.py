"""MU3 saved-pose command state machine; no ROS or hardware dependencies."""

from .field_side import LEFT, RIGHT


# Legacy slots 1..7 / 9..15 remain unchanged. New points use 17..22 /
# 25..30; 0,8,16,24 are not destinations. Field and point are atomic.
SIDE_SLOT_OFFSET = 8
MAX_SAVED_POSES = 13


def encode_slot(index, side):
    """One-based configured point index -> backward-compatible wire slot."""
    if not 1 <= index <= MAX_SAVED_POSES or side not in (LEFT, RIGHT):
        raise ValueError('invalid saved point or field')
    return (index if index <= 7 else index + 9) + (8 if side == RIGHT else 0)


def decode_slot(slot, names):
    """Return (pose name, field side), rejecting every reserved/unknown slot."""
    slot = int(slot)
    if not 1 <= slot <= 30 or slot in (8, 16, 23, 24):
        return None, None
    side = RIGHT if slot & 8 else LEFT
    local = slot & 7
    index = local + (7 if slot >= 16 else 0)
    if not 1 <= index <= min(len(names), MAX_SAVED_POSES):
        return None, None
    return names[index - 1], side


class RemoteNavigation:
    def __init__(self, names, emit):
        self.names = tuple(names)
        self.emit = emit
        self.side = LEFT
        self.sequence = None
        self.pi_sequence = None
        self.received = float('-inf')
        self.safety_received = float('-inf')
        self.safety = {}
        self.poses = {}
        self.input = {}
        self.ready = False
        self.owned = False
        self.phase = 'IDLE'
        self.target = None
        self.started = 0.0
        self.saw_navigation = False

    def safety_update(self, state, now):
        self.safety = state
        self.safety_received = now

    def healthy(self, now):
        return not self.health_problem(now)

    def health_problem(self, now):
        p, s = self.input, self.safety
        checks = (
            (now - self.received >= .3, 'TELEMETRY_STALE'),
            (now - self.safety_received >= 1.0, 'SAFETY_STATE_STALE'),
            (not p.get('mu3_alive'), 'MU3_LOST'),
            (not p.get('link_alive'), 'MOTOR_LINK_LOST'),
            (not p.get('uart_open'), 'MOTOR_UART_CLOSED'),
            (p.get('estop_active') or s.get('emergency_stop'), 'EMERGENCY_STOP'),
            (p.get('fault_latched'), 'MOTOR_FAULT_LATCHED'),
            (not s.get('tracking_effective', False), 'LOCALIZATION_UNHEALTHY'),
            (s.get('require_rl_policy') and not s.get('rl_effective'), 'RL_UNHEALTHY'),
        )
        return next((reason for failed, reason in checks if failed), '')

    def receive(self, message, now, subscribers_ready=True):
        pi_seq = int(message['pi_seq'])
        if self.pi_sequence is not None:
            delta = (pi_seq - self.pi_sequence) & 65535
            if not 0 < delta < 32768:
                return  # Duplicate/out-of-order UDP must not replay commands.
        self.pi_sequence = pi_seq
        self.input = message
        self.received = now
        slot, sequence = int(message['slot']), int(message['sequence'])
        if not message.get('mu3_alive'):
            self.stop('MU3_LOST')
            self.ready = False
            self.sequence = sequence
            return
        if slot == 0:
            self.stop('RELEASED')
            self.ready = True
            self.sequence = sequence
            return
        if self.sequence is not None:
            delta = (sequence - self.sequence) & 65535
            if not 0 < delta < 32768:
                return
        self.sequence = sequence
        if not self.ready:
            self.emit('status', 'RELEASE_REQUIRED')
            return
        name, side = decode_slot(slot, self.names)
        if name is None or name not in self.poses:
            self.stop('UNKNOWN_SAVED_POSE')
            self.emit('status', 'UNKNOWN_SAVED_POSE')
            return
        if not subscribers_ready or not self.healthy(now):
            self.stop('NOT_READY')
            reason = ('CONTROL_SUBSCRIBERS_UNAVAILABLE' if not subscribers_ready
                      else self.health_problem(now))
            self.emit('status', 'NOT_READY:' + reason)
            return
        self.owned = True
        self.emit('cancel', True)
        self.emit('arm', False)
        self.target = name
        self.side = side
        self.phase = 'DISARMING'
        self.started = now
        self.saw_navigation = False
        self.emit('status', f'PREPARING:{self.target}@{self.side}')

    def tick(self, now):
        if now - self.received >= .3:
            self.stop('TELEMETRY_LOST')
            self.ready = False
            self.pi_sequence = None
            return
        if not self.owned:
            return
        if not self.healthy(now):
            self.stop('HEALTH_LOST')
            self.ready = False
            return
        if self.phase == 'DISARMING' and now - self.started >= .2:
            if self.input.get('auto_engaged'):
                if now - self.started > 2.0:
                    self.stop('DISARM_TIMEOUT')
                return
            self.emit('arm', True)
            self.phase = 'ARMING'
        elif self.phase == 'ARMING':
            if self.input.get('auto_engaged') and self.safety.get('armed'):
                self.emit('goal', (self.target, self.side))
                self.emit('status', f'GOAL_SENT:{self.target}@{self.side}')
                self.phase = 'NAVIGATING'
                self.started = now
            elif now - self.started > 2.0:
                self.stop('ARM_TIMEOUT')
        elif self.phase == 'NAVIGATING':
            if not self.input.get('auto_engaged') or not self.safety.get('armed'):
                self.stop('DISARMED')
            elif not self.saw_navigation and now - self.started > 3.0:
                self.stop('GOAL_ACK_TIMEOUT')

    def navigation_update(self, status):
        if self.phase != 'NAVIGATING' or status.get('remembered_pose') != self.target:
            return
        state = str(status.get('state', ''))
        self.saw_navigation = True
        if state in {'SUCCEEDED', 'FAILED', 'REJECTED', 'ABORTED', 'CANCELED',
                     'UNKNOWN_REMEMBERED_POSE', 'INVALID_REMEMBERED_POSE'}:
            self.stop(state)

    def stop(self, reason):
        if self.owned:
            # A completed goal has already stopped; cancel would overwrite its
            # retained SUCCEEDED result with an unrelated CANCELED status.
            if reason != 'SUCCEEDED':
                self.emit('cancel', True)
            self.emit('arm', False)
            self.emit('status', reason)
        self.owned = False
        self.phase = 'IDLE'
        self.target = None
