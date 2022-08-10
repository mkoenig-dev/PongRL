#!/usr/bin/env python

"""The setup script."""

import pkg_resources
from setuptools import find_packages, setup

with open("requirements.txt") as req_file:
    requirements = [
        str(requirement) for requirement in pkg_resources.parse_requirements(req_file)
    ]

test_requirements = [
    "pytest>=3",
]

description = "Game environment to train a DQN on pong."

setup(
    author="Marcel König",
    author_email="marcel.koenig97@web.de",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description=description,
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords="pong",
    name="pong",
    packages=find_packages(include=["pong", "pong.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Kinguuusama/PongRL",
    version="0.1.0",
    zip_safe=False,
)
