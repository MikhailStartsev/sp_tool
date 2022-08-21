#!/usr/bin/env python

from distutils.core import setup
from __init__ import __version__


setup(name='sp_tool',
      version=__version__,
      description='A tool for labelling smooth pursuit (SP) eye movements (along with fixations and saccades) '
                  'based on clustering the gaze points of multiple observers in time-X-Y space',
      author='Mikhail Startsev',
      author_email='mikhail.startsev@tum.de',
      url='http://dl.acm.org/citation.cfm?id=2857521',
      keywords='smooth pursuit eye tracking DBSCAN',
      install_requires=['liac-arff', 'numpy', 'scikit-learn', 'jellyfish'],
      package_dir={'sp_tool': ''},
      packages=['sp_tool.examples', 'sp_tool'])
