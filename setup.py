from setuptools import setup

setup(
    name="hdfisher",
    version="1.0",
    description="Fisher Forecasting for CMB-HD",
    url="https://github.com/CMB-HD/hdlike",
    author="CMB-HD Collaboration",
    python_requires=">=3",
    #install_requires=["numpy"],
    packages=["hdfisher"],
    include_package_data=True,
)