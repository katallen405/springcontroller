import glob
import os

from setuptools import setup

package_name = "springcontroller"


def _tree(src_root):
    """data_files entries installing src_root/** under share/<pkg>/src_root,
    keeping its subdirectory layout (package:// mesh URIs depend on it)."""
    entries = []
    for dirpath, _, filenames in os.walk(src_root):
        if filenames:
            entries.append((os.path.join("share", package_name, dirpath),
                            [os.path.join(dirpath, f) for f in filenames]))
    return entries

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Globbed rather than listed by name: add a new launch/*.launch.py
        # or config/*.yaml file and it's installed automatically, no
        # setup.py edit needed -- an explicit list risks a config file
        # silently missing from the install.
        ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
        # Meshes for flat_urdf_files/gen2_robotiq_flat.urdf, referenced as
        # package://springcontroller/meshes/... (see
        # tools/make_gen2_robotiq_urdf.sh).
        *_tree("meshes"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kat Allen",
    maintainer_email="kat.allen@tufts.edu",
    description="Virtual spring impedance controller for robot arms.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "virtual_spring_node = springcontroller.virtual_spring_node:main",
            "torque_relay = springcontroller.torque_relay:main",
            "press_to_pin = springcontroller.press_to_pin:main",
        ],

    },
)
