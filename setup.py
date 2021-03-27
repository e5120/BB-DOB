from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="BB-DOB",
    version="1.0.0",
    description="The Black-Box Discrete Optimization Benchmarks",
    author="sho shimazu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['bbdob'],
    url="https://github.com/e5120/BB-DOB",
    license="MIT",
)
