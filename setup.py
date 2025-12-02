from setuptools import setup, find_packages

name = "skhippr"
version = "2.0"
author = "Fabia Bayer"
author_email = "skhippr@inm.uni-stuttgart.de"
url = "SKHiPPR <https://github.com/f-bayer/SKHiPPR>"
description = "Stability using Koopman-Hill Projection for Periodic Solutions and Resonance Curves"
long_description = ""
license = "LICENSE"

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    install_requires=["numpy>=1.26.4", "scipy>=1.13.0", "matplotlib >=3.10.1"],
    packages=find_packages(),
    python_requires=">=3.10",
)
