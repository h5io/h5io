#! /usr/bin/env python
#

# Copyright (C) 2011-2014 Alexandre Gramfort
# <alexandre.gramfort@telecom-paristech.fr>

import os

import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup
import versioneer


descr = """Python Objects Onto HDF5"""

DISTNAME = 'h5io'
DESCRIPTION = descr
MAINTAINER = 'Eric Larson'
MAINTAINER_EMAIL = 'larson.eric.d@gmail.com'
URL = 'http://h5io.github.io'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/h5io/h5io'


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
          version=versioneer.get_version(),
          download_url=DOWNLOAD_URL,
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
          scripts=[],
          cmdclass=versioneer.get_cmdclass())
