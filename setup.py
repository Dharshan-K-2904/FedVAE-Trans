#!/usr/bin/env python
"""Setup configuration for FedVAE-Trans package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fedvae-trans",
    version="1.0.0",
    author="Dharshan K, Aparajitha Chandan, Yeswant Padavala",
    author_email="dharshan.ksk@pes.edu",
    description="Communication-Efficient Federated Variational Transformer for Log Anomaly Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PES-University/FedVAE-Trans",
    project_urls={
        "Bug Tracker": "https://github.com/PES-University/FedVAE-Trans/issues",
        "Documentation": "https://github.com/PES-University/FedVAE-Trans#readme",
        "Source Code": "https://github.com/PES-University/FedVAE-Trans",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "federated-learning",
        "variational-autoencoder",
        "transformer",
        "anomaly-detection",
        "privacy-preserving",
        "distributed-learning",
        "deep-learning",
    ],
    zip_safe=False,
)
