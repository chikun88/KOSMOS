"""Installed and source configurations resolve the same hash-verified CAD."""
from pathlib import Path
import shutil

import pytest

from omni_autonomy_next.config import ConfigError, load_field, load_robot
from omni_autonomy_next.cad_import import convert_stl


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('installed', [False, True])
def test_cad_loads_outside_repository(tmp_path, monkeypatch, installed):
    package = PACKAGE
    if installed:
        package = tmp_path / 'share' / 'omni_autonomy_next'
        shutil.copytree(PACKAGE / 'config', package / 'config')
        shutil.copytree(PACKAGE / 'comparison_assets', package / 'comparison_assets')
    unrelated = tmp_path / 'other'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    robot = load_robot(str(package / 'config' / 'robot.yaml'))
    assert Path(robot['cad_model_file']).is_file()
    for name in ('field_cad.yaml', 'field_planning.yaml'):
        field = load_field(str(package / 'config' / name))
        assert Path(field['source_stl']).is_file()
        assert Path(field['source_layout']).is_file()


def test_installed_cad_changes_are_rejected(tmp_path):
    package = tmp_path / 'share' / 'omni_autonomy_next'
    shutil.copytree(PACKAGE / 'config', package / 'config')
    shutil.copytree(PACKAGE / 'comparison_assets', package / 'comparison_assets')
    (package / 'comparison_assets' / 'field.stl').write_bytes(b'changed CAD')
    with pytest.raises(ConfigError, match='Field CAD changed'):
        load_field(str(package / 'config' / 'field_planning.yaml'))


def test_relative_cad_import_loads_from_a_different_output_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(PACKAGE)
    output = tmp_path / 'generated' / 'field.yaml'
    converted = convert_stl(Path('comparison_assets/field.stl'), output,
                            z_mm=130., simplify_tolerance_mm=.05)
    monkeypatch.chdir(tmp_path)
    loaded = load_field(str(output))
    assert Path(loaded['source_stl']) == (PACKAGE/'comparison_assets/field.stl').resolve()
    assert loaded['source_stl_sha256'] == converted['field']['source_stl_sha256']
    assert len(loaded['walls']) > 1
