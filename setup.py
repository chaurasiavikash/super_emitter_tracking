from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="super-emitter-tracking",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Super-Emitter Tracking and Temporal Analysis System for TROPOMI methane data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/super-emitter-tracking",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.7.0", "flake8>=6.0.0"],
        "viz": ["plotly>=5.15.0", "dash>=2.11.0"],
        "ml": ["lightgbm>=4.0.0", "catboost>=1.2.0", "optuna>=3.3.0"],
    },
    entry_points={
        "console_scripts": [
            "super-emitter-tracker=main:main",
        ],
    },
)
