import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('launcher', Path(__file__).parents[1] / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_changes_to_configuration_or_source_require_rebuild(tmp_path):
    source, output = tmp_path / 'src', tmp_path / 'build'
    source.mkdir()
    config = source / 'robot.yaml'
    config.write_text('speed: 1\n')
    assert not launcher.build_matches_sources(source, output)
    launcher.remember_built_sources(source, output, launcher.source_digest(source))
    assert launcher.build_matches_sources(source, output)
    config.write_text('speed: 2\n')
    assert not launcher.build_matches_sources(source, output)


def test_build_products_and_bytecode_do_not_invalidate_build(tmp_path):
    source, output = tmp_path / 'src', tmp_path / 'src' / 'build'
    source.mkdir()
    (source / 'node.py').write_text('pass\n')
    launcher.remember_built_sources(source, output, launcher.source_digest(source))
    (output / 'binary').write_bytes(b'build output')
    (source / '__pycache__').mkdir()
    (source / '__pycache__' / 'node.pyc').write_bytes(b'bytecode')
    assert launcher.build_matches_sources(source, output)


def test_edit_during_build_is_not_marked_successful(tmp_path):
    source, output = tmp_path / 'src', tmp_path / 'build'
    source.mkdir()
    (source / 'node.py').write_text('pass\n')
    expected = launcher.source_digest(source)
    (source / 'node.py').write_text('print(1)\n')
    with pytest.raises(RuntimeError, match='ソースが変更'):
        launcher.remember_built_sources(source, output, expected)
    assert not (output / '.source.sha256').exists()


def test_linked_source_package_changes_require_rebuild(tmp_path):
    source, external = tmp_path / 'src', tmp_path / 'external'
    source.mkdir()
    external.mkdir()
    node = external / 'node.py'
    node.write_text('pass\n')
    (source / 'package').symlink_to(external, target_is_directory=True)
    before = launcher.source_digest(source)
    node.write_text('print(1)\n')
    assert launcher.source_digest(source) != before


def test_cyclic_source_links_cannot_certify_a_build(tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'cycle').symlink_to(source, target_is_directory=True)
    with pytest.raises(RuntimeError, match='循環'):
        launcher.source_digest(source)
