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
        ("share/" + package_name + "/config", ["config/kinova_springs.yaml"]),
        ("share/" + package_name + "/config", ["config/2DoF_springs.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Virtual spring impedance controller for robot arms.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "virtual_spring_node = springcontroller.virtual_spring_node:main",
        ],
    },
)
