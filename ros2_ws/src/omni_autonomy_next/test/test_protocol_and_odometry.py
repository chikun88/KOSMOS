import numpy as np

from omni_autonomy_next.geometry import compose_pose
from omni_autonomy_next.measurement_wheel_odometry import (
    counter_delta,
    meters_per_count_from_radius,
)
from omni_autonomy_next.wall_localizer_node import localizer_tf_chain
from omni_autonomy_next.motor_udp_protocol import (
    ProtocolError,
    decode_v3_command,
    encode_v3_command,
)


def test_counter_wrap_is_shortest_signed_delta():
    maximum = (1 << 32) - 1
    np.testing.assert_array_equal(
        counter_delta([2, maximum - 2], [maximum - 2, 2], counter_bits=32),
        [5.0, -5.0],
    )


def test_encoder_scale_is_circumference_per_count():
    # 25.4 mm radius = the 50.8 mm outside diameter of the passive omni wheel.
    scale = meters_per_count_from_radius([2048.0] * 4, 0.0254)
    np.testing.assert_allclose(scale, [2.0 * np.pi * 0.0254 / 2048.0] * 4)


FRAMES = {'map_frame': 'map', 'odom_frame': 'odom', 'base_frame': 'base_link'}


def _resolved_base_pose(chain):
    """Walk map -> ... -> base_link, or fail if the chain does not connect."""
    children = {parent: (child, pose) for parent, child, pose in chain}
    pose = np.zeros(3)
    frame = 'map'
    while frame != 'base_link':
        assert frame in children, f'{frame} has no child transform'
        frame, step = children[frame]
        pose = compose_pose(pose, step)
    return pose


def test_wheel_odometry_owns_odom_to_base_link_while_it_is_alive():
    pose = np.array([-1.8, 4.75, -1.5708])
    wheel_pose = np.array([0.4, -0.2, 0.3])
    chain = localizer_tf_chain(pose, wheel_pose, True, **FRAMES)
    assert [(parent, child) for parent, child, _ in chain] == [('map', 'odom')]
    # measurement_wheel supplies the missing link, so the full chain still
    # resolves to the estimated pose.
    np.testing.assert_allclose(
        _resolved_base_pose(chain + [('odom', 'base_link', wheel_pose)]),
        pose,
        atol=1.0e-9,
    )


def test_missing_measurement_wheel_keeps_base_link_attached_to_map():
    # The counter board was never opened, so nobody else publishes
    # odom->base_link. Without the substitute RViz loses the robot markers and
    # both LiDAR point clouds even though the field CAD still renders.
    pose = np.array([-1.8, 4.75, -1.5708])
    chain = localizer_tf_chain(pose, None, False, **FRAMES)
    assert [(parent, child) for parent, child, _ in chain] == [
        ('map', 'odom'), ('odom', 'base_link'),
    ]
    np.testing.assert_allclose(_resolved_base_pose(chain), pose, atol=1.0e-9)


def test_wheel_dropout_moves_base_link_without_jumping_the_odom_frame():
    pose = np.array([2.0, -1.0, 0.7])
    wheel_pose = np.array([0.4, -0.2, 0.3])
    alive = localizer_tf_chain(pose, wheel_pose, True, **FRAMES)
    dropped = localizer_tf_chain(pose, wheel_pose, False, **FRAMES)
    np.testing.assert_allclose(alive[0][2], dropped[0][2], atol=1.0e-12)
    np.testing.assert_allclose(_resolved_base_pose(dropped), pose, atol=1.0e-9)


def test_v3_velocity_protocol_round_trip_and_crc():
    frame = encode_v3_command(
        seq=65530, t_tx_us=0x12345678,
        vx_mps=0.713, vy_mps=-0.427, wz_radps=1.234,
        auto_request=True, estop=False,
    )
    decoded = decode_v3_command(frame)
    assert decoded[:3] == (713, -427, 1234)
    assert decoded[4] & 0x01
    assert decoded[5] == 65530
    assert decoded[6] == 0x12345678
    damaged = bytearray(frame)
    damaged[8] ^= 0x40
    try:
        decode_v3_command(bytes(damaged))
    except ProtocolError:
        pass
    else:
        raise AssertionError('CRC damage must be rejected')
