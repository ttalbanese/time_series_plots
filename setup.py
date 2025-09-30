from setuptools import find_packages, setup

setup(
    name="time_series_plots",
    packages=find_packages(exclude=["time_series_plots_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
