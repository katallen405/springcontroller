import glob

from setuptools import setup

package_name = "springcontroller_ui"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Globbed rather than listed by name, matching springcontroller's own
        # setup.py convention -- add a new launch/config/web file and it's
        # installed automatically.
        ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
        ("share/" + package_name + "/web", glob.glob("web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kat Allen",
    maintainer_email="kat.allen@tufts.edu",
    description="Browser-based control panel for the Gen3 spring-controller study.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "study_control_panel_node = springcontroller_ui.orchestration_node:main",
        ],
    },
)
