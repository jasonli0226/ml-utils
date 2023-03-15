from setuptools import setup, find_packages

setup(
    name="example-py-package",
    version="0.1",
    packages=find_packages(),
    description="An example package",
    author="Jason Li",
    install_requires=["tensorflow", "matplotlib", "numpy"],
)
