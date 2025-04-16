from setuptools import setup, find_packages

setup(
    name="crypto_predictor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "predictor=predictor.cli:app",
        ],
    },
)
