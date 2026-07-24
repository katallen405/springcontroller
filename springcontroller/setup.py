from setuptools import setup

package_name = "springcontroller"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/virtual_spring.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/torque_relay.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/gen3_spring.launch.py"]),
        ("share/" + package_name + "/config", ["config/kinova_springs.yaml"]),
        ("share/" + package_name + "/config", ["config/2DoF_springs.yaml"]),
        ("share/" + package_name + "/config", ["config/gen3_springs.yaml"]),
        ("share/" + package_name + "/launch", ["launch/press_to_pin.launch.py"]),
        ("share/" + package_name + "/config", ["config/gen2_kinova_springs.yaml"]),
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
            "equilibrium_mover = springcontroller.equilibrium_mover:main",
            "press_to_pin = springcontroller.press_to_pin:main",
        ],

    },
)
