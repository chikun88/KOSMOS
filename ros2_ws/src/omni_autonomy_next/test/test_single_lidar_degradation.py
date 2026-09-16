"""One dead LiDAR must degrade autonomy, not disable it.

The rear A2M8 went silent on the deployed robot.  Two independent places used
to turn that into a full stop: the localizer refused to match unless every
configured LiDAR was fresh, and Nav2's collision monitor brakes to zero
whenever a configured observation source has no data.  These tests pin both
behaviours plus the port identification that made the outage look like a
missing device node.
"""

import re
import sys
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from omni_autonomy_next.lidar_ports import resolve_lidar_ports  # noqa: E402
from omni_autonomy_next.scan_freshness import (  # noqa: E402
    scan_generations_changed,
)
from omni_autonomy_next.scan_source_supervisor_node import (  # noqa: E402
    sources_to_change,
)

FRONT_SERIAL = 'B5E89AF2C1EA98D4BEEB9CF03A3B3517'
LIDARS = [
    {
        'name': 'front',
        'device_serial': 'B5E89A',
        'serial_port': '/dev/serial/by-path/does-not-exist-front',
    },
    {
        'name': 'rear',
        'device_serial': '849F9A',
        'serial_port': '/dev/serial/by-path/does-not-exist-rear',
    },
]


def test_lidars_are_identified_by_serial_number_not_by_hub_port():
    # The rear cable moved from hub port 2.3 to 2.2, so the by-path device in
    # robot.yaml disappeared while the LiDAR itself was unchanged.
    serials = {'/dev/ttyUSB5': '849F9A0011223344556677880B4C3517',
               '/dev/ttyUSB9': FRONT_SERIAL}
    ports, responsive, _ = resolve_lidar_ports(
        LIDARS, probe=serials.get, ports=sorted(serials),
    )
    assert ports == {'front': '/dev/ttyUSB9', 'rear': '/dev/ttyUSB5'}
    assert responsive == {'front', 'rear'}


def test_a_silent_lidar_still_gets_its_port_but_is_not_called_responsive():
    serials = {'/dev/ttyUSB0': None, '/dev/ttyUSB2': FRONT_SERIAL}
    ports, responsive, notes = resolve_lidar_ports(
        LIDARS, probe=serials.get, ports=sorted(serials),
    )
    # Handing the leftover device to the silent LiDAR lets sllidar_node report
    # the real fault instead of a missing /dev entry.
    assert ports == {'front': '/dev/ttyUSB2', 'rear': '/dev/ttyUSB0'}
    assert responsive == {'front'}
    assert any('elimination' in note for note in notes)


def test_a_failed_probe_is_not_mistaken_for_two_dead_lidars():
    # No pyserial, no permission, or the stack already holds the ports open.
    # Reporting both dead would switch off every collision-monitor source.
    _, responsive, _ = resolve_lidar_ports(
        LIDARS, probe=lambda port: None, ports=['/dev/ttyUSB0', '/dev/ttyUSB2'],
    )
    assert responsive == {'front', 'rear'}


def test_generations_are_keyed_by_lidar_so_a_swap_is_never_read_as_unchanged():
    # With bare counters, dropping to the rear LiDAR alone at the same
    # generation number looked unchanged and froze matching at a stale pose.
    assert scan_generations_changed((('rear', 7),), (('front', 7),))
    assert not scan_generations_changed((('front', 7),), (('front', 7),))


def test_a_silent_source_is_disabled_and_re_enabled_when_it_returns():
    enabled = {'scan_front': True, 'scan_rear': True}
    assert sources_to_change(enabled, {'scan_front'}) == {'scan_rear': False}
    enabled['scan_rear'] = False
    assert sources_to_change(enabled, {'scan_front'}) == {}
    # A re-plugged LiDAR must rejoin the safety layer without a restart.
    assert sources_to_change(enabled, {'scan_front', 'scan_rear'}) == {
        'scan_rear': True
    }


def test_losing_every_lidar_leaves_the_monitor_stopping_the_robot():
    # Disabling the last source would let the robot drive with no obstacle input
    # at all, which is worse than the stop the monitor would otherwise impose.
    enabled = {'scan_front': True, 'scan_rear': True}
    assert sources_to_change(enabled, set()) == {}


def test_the_supervisor_waits_longer_than_the_monitors_own_source_timeout():
    nav2 = yaml.safe_load(
        (PACKAGE / 'config' / 'nav2_next.yaml').read_text(encoding='utf-8')
    )
    monitor = nav2['collision_monitor']['ros__parameters']
    supervisor = (
        PACKAGE / 'omni_autonomy_next' / 'scan_source_supervisor_node.py'
    ).read_text(encoding='utf-8')
    default = float(
        re.search(
            r"declare_parameter\('scan_timeout_sec', ([0-9.]+)\)", supervisor
        ).group(1)
    )
    for source in monitor['observation_sources']:
        assert default > float(monitor[source]['source_timeout'])


def test_the_supervisor_is_launched_with_the_stack():
    system = (PACKAGE / 'launch' / 'system.launch.py').read_text(encoding='utf-8')
    assert "executable='scan_source_supervisor'" in system
    entry_points = (PACKAGE / 'setup.py').read_text(encoding='utf-8')
    assert 'scan_source_supervisor = ' in entry_points


def test_localizer_matches_on_whatever_lidars_are_live():
    source = (PACKAGE / 'omni_autonomy_next' / 'wall_localizer_node.py').read_text(
        encoding='utf-8'
    )
    assert 'min_active_lidars' in source
    # The old code returned on the first stale LiDAR, which stopped the robot
    # for as long as one unit was broken.
    assert 'WAITING_FOR_BOTH_SCANS' not in source
