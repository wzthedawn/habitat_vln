#!/usr/bin/env python3
"""Setup script for Multi-Agent VLN Navigation System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="habitat_vln",
    version="1.0.0",
    description="Multi-Agent VLN Navigation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VLN Research Team",
    author_email="",
    url="https://github.com/example/habitat_vln",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "vln-train=scripts.train:main",
            "vln-eval=scripts.evaluate:main",
            "vln-infer=scripts.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)