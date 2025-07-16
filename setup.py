"""
Setup script for Web Graph Generator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="web-graph-analysis",
    version="1.0.0",
    author="Maxwell Morgan",
    author_email="maxwell.morgan@detroitmi.gov",
    description="Perform analysis of website information graphs and generate visualizations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CityOfDetroit/web-graph-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
        ],
        "advanced": [
            "plotly>=5.11.0",
            "dash>=2.6.0",
            "pandas>=1.5.0",
            "scipy>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "web-graph-analysis=web_graph_generator.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)