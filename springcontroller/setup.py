import glob

from setuptools import setup

package_name = "springcontroller"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Globbed rather than listed by name: add a new launch/*.launch.py
        # or config/*.yaml file and it's installed automatically, no
        # setup.py edit needed. (A file that previously went missing here
        # -- config/springs.yaml -- would install silently broken with the
        # old explicit-list approach; see git history if curious.)
        ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
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
