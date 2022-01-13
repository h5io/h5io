#! /usr/bin/env python
#

# Copyright (C) 2011-2014 Alexandre Gramfort
# <alexandre.gramfort@telecom-paristech.fr>

import os
import pathlib

from setuptools import setup


descr = """Python Objects Onto HDF5"""

DISTNAME = 'h5io'
DESCRIPTION = descr
MAINTAINER = 'Eric Larson'
MAINTAINER_EMAIL = 'larson.eric.d@gmail.com'
URL = 'http://h5io.github.io'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/h5io/h5io'

version = None
with open(pathlib.Path('h5io') / '_version.py', 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
    else:
        raise RuntimeError('Could not determine version')


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=version,
          download_url=DOWNLOAD_URL,
          python_requires='>=3.7',
          install_requires=['numpy'],
          long_description=open('README.rst').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['h5io', 'h5io.tests'],
          package_data={},
          scripts=[])
