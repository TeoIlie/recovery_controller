from setuptools import setup
import os
from glob import glob

package_name = "recovery_controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Teodor Ilie",
    maintainer_email="teodor.ilie@queensu.com",
    description="Recovery policy deployment for sim-to-real transfer on F1TENTH",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "recovery_node = recovery_controller.recovery_node:main",
        ],
    },
)
