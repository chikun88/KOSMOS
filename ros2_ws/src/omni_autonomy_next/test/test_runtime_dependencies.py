"""A current Python wrapper must not mask the native TF deadlock version."""
import pytest

from omni_autonomy_next.runtime_dependencies import require_fixed_tf2


def installed_packages(tmp_path, tf2='0.36.23', tf2_ros='0.36.23'):
    for package, version in (('tf2', tf2), ('tf2_ros', tf2_ros)):
        share = tmp_path / package
        share.mkdir()
        (share / 'package.xml').write_text(
            f'<package><name>{package}</name><version>{version}</version></package>')
    return lambda package: str(tmp_path / package)


def test_new_wrapper_does_not_hide_old_native_tf_core(tmp_path):
    with pytest.raises(RuntimeError, match=r'tf2 0\.36\.22.*deadlock'):
        require_fixed_tf2(installed_packages(tmp_path, tf2='0.36.22'))


def test_old_wrapper_is_rejected_even_with_fixed_core(tmp_path):
    with pytest.raises(RuntimeError, match=r'tf2_ros 0\.36\.22'):
        require_fixed_tf2(installed_packages(tmp_path, tf2_ros='0.36.22'))


@pytest.mark.parametrize('version', ['0.36.23', '0.36.24', '0.37.0', '1.0.0'])
def test_fixed_or_newer_dependencies_are_accepted(tmp_path, version):
    assert require_fixed_tf2(installed_packages(tmp_path, version, version)) == {
        'tf2': version, 'tf2_ros': version}


@pytest.mark.parametrize('version', ['', 'unknown', '0.36', '0.36.23foo'])
def test_unverifiable_dependency_is_not_treated_as_fixed(tmp_path, version):
    with pytest.raises(RuntimeError, match='Invalid installed tf2 version'):
        require_fixed_tf2(installed_packages(tmp_path, tf2=version))


def test_selected_overlay_without_manifest_cannot_pass(tmp_path):
    with pytest.raises(RuntimeError, match='Cannot verify installed tf2'):
        require_fixed_tf2(lambda package: str(tmp_path / package))
