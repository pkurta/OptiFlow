from setuptools import setup, find_packages


setup(
    name="optiflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyQt5==5.15.10",
        "numpy==1.26.4",
        "matplotlib==3.8.4",
        "scipy==1.11.4",
    ],
    entry_points={
        "console_scripts": [
            "optiflow-app=optiflow.app:run_app",
        ]
    },
)

