# install using 'pip install -e .'
import os
from setuptools import setup

setup(name='GDN',
      packages=['gdn'],
      package_dir={'gdn': 'gdn'},
      install_requires=['torch',],
      version='0.0.1')

