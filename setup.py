from setuptools import setup, find_packages

setup(
    name="adoption_rate_analytics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "pyodbc",
        "statsmodels"
    ],
    python_requires=">=3.8",
) 