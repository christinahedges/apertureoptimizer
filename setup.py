#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/apertureoptimizer*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('version.py').read())

setup(name='apertureoptimizer',
      version=__version__,
      description="Tool for optimizing Kepler, K2 and TESS apertures, given a planet signal.",
      long_description=open('README.md').read(),
      author='KeplerGO',
      author_email='keplergo@mail.arc.nasa.gov',
      license='MIT',
      install_requires=['numpy>=1.11', 'astropy>=1.3',
                        'matplotlib>=1.5.3', 'tqdm', 'lightkurve'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov', 'pytest-remotedata'],
      include_package_data=True,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
