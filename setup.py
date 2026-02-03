#!/usr/bin/env python3
"""
SloughGPT Setup Script
Proper Python package setup for distribution
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "SloughGPT - Custom GPT system with continuous learning and cognitive capabilities"

# Read requirements file
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return ["torch", "numpy", "transformers", "fastapi", "uvicorn"]

setup(
    name="sloughgpt",
    version="1.0.0",
    author="SloughGPT Team",
    author_email="team@sloughgpt.ai",
    description="Custom GPT system with continuous learning and cognitive capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sloughgpt/sloughgpt",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="artificial intelligence, gpt, machine learning, neural network, cognitive system, continuous learning",
    entry_points={
        "console_scripts": [
            "sloughgpt=sloughgpt.cli:main",
        ],
        "gui_scripts": [
            "sloughgpt-web=sloughgpt.api:main",
        ],
    },
    package_data={
        "sloughgpt": [
            "config/*.json",
            "models/*.bin",
            "static/*",
        ],
    },
    zip_safe=False,
)