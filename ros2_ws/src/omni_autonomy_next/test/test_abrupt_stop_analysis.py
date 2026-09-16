import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('stop_analysis', ROOT / 'scripts/inspect_abrupt_stops.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_publisher_clock_pulses_exclude_idle_and_delayed_receipts(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'start_ros_ns': 1_000_000_000, 'session_id': 'test',
        'settings': {'operation_mode': 'hardware'}}))
    (tmp_path / 'recording-summary.json').write_text('{"closed":true}')
    rows = []
    for t, speed in [(1., .6), (1.05, 0.), (1.10, .2),
                     (2., .5), (2.05, 0.), (3., 0.)]:
        rows.append(dict(topic='/cmd_vel_safe', published_unix_ns=int(t * 1e9),
                         monotonic_ns=100, received_ros_ns=8_000_000_000,
                         received_unix_ns=8_000_000_000,
                         value={'linear': {'x': speed, 'y': 0.}, 'angular': {'z': 0.}}))
    # Recorder callback order may differ from publisher order.
    (tmp_path / 'samples-0000.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in reversed(rows)) + '\n{"partial":')
    result = analysis.inspect(tmp_path)
    assert result['malformed_rows'] == 1
    assert len(result['abrupt_stop_pulses']) == 1
    event = result['abrupt_stop_pulses'][0]
    assert event['at_sec'] == .05
    assert event['zero_duration_sec'] == .05
    assert event['previous_command'] == [.6, 0., 0.]
    assert event['wheel_speed_after_min'] is None
    assert result['raw_prefixes']['samples-0000.jsonl']['sha256']
    for row in rows:
        row.pop('published_unix_ns')
    (tmp_path / 'samples-0000.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    result = analysis.inspect(tmp_path)
    assert result['abrupt_stop_pulses'] == []
    assert result['missing_publisher_clock']['/cmd_vel_safe'] == 6
