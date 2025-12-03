"""Setup script for PyAirline RM."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyairline-rm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Airline Revenue Management Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyairline_rm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "viz": [
            "plotly>=5.17.0",
            "dash>=2.14.0",
            "dash-bootstrap-components>=1.5.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "lightgbm>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyairline-rm=pyairline_rm.main:main",
        ],
    },
)
