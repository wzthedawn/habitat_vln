#!/usr/bin/env python3
"""Setup script for Pipeline VLN Navigation System."""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

requirements_path = Path(__file__).parent / "envs" / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="habitat_vln",
    version="2.0.0",
    description="Pipeline VLN Navigation System with Multi-Tier Model Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VLN Research Team",
    url="https://github.com/wzthedawn/habitat_vln",
    packages=find_packages(exclude=["tests", "src", "docs", "logs", "results"]),
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "vln-eval=run_vln_experiment:main",
            "vln-server=vllm_server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
