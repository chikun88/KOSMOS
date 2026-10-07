"""Reject ROS dependencies with known control-path deadlocks before startup."""
from pathlib import Path
import re
from xml.etree import ElementTree


TF2_MINIMUM_VERSION = (0, 36, 23)


def require_fixed_tf2(share_lookup=None):
    """Check the actual selected ROS overlay, including tf2's native core.

    Jazzy geometry2 0.36.23 fixes a BufferCore / waitForTransform lock inversion.
    A current tf2_ros package alone does not upgrade an older installed tf2 core.
    """
    if share_lookup is None:
        from ament_index_python.packages import get_package_share_directory
        share_lookup = get_package_share_directory
    versions = {}
    for package in ('tf2', 'tf2_ros'):
        try:
            manifest = Path(share_lookup(package)) / 'package.xml'
            version = ElementTree.parse(manifest).getroot().findtext('version', '').strip()
        except Exception as error:
            raise RuntimeError(f'Cannot verify installed {package}: {error}') from error
        if not re.fullmatch(r'\d+\.\d+\.\d+', version):
            raise RuntimeError(f'Invalid installed {package} version: {version!r}')
        if tuple(map(int, version.split('.'))) < TF2_MINIMUM_VERSION:
            raise RuntimeError(
                f'{package} {version} has an unsupported TF dependency version. '
                'ROS Jazzy requires tf2 and tf2_ros >= 0.36.23 to avoid a TF '
                'deadlock. Run: sudo apt-get update && sudo apt-get install '
                'ros-jazzy-tf2 ros-jazzy-tf2-ros')
        versions[package] = version
    return versions


if __name__ == '__main__':
    print(f'ROS TF dependencies: {require_fixed_tf2()}')
