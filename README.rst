.. -*- mode: rst -*-

|Travis|_ |Appveyor|_ |Coveralls|_

.. |Travis| image:: https://api.travis-ci.org/h5io/h5io.png?branch=master
.. _Travis: https://travis-ci.org/h5io/h5io

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/puwaarmllxq5wfvm/branch/master?svg=true
.. _Appveyor: https://ci.appveyor.com/project/Eric89GXL/h5io/branch/master

.. |Coveralls| image:: https://coveralls.io/repos/h5io/h5io/badge.png?branch=master
.. _Coveralls: https://coveralls.io/r/h5io/h5io?branch=master

`h5io <http://h5io.github.io>`_
=======================================================

Python Objects Onto HDF5 (h5io) is a package designed to
facilitate saving standard Python objects into the forward-compatible
HDF5 format.

Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using git, simply type::

    git clone git://github.com/h5io/h5io.git

If you don't have git installed, you can download a zip or tarball
of the latest code: https://github.com/h5io/h5io/archives/master

Install h5io
^^^^^^^^^^^^

As any Python packages, to install h5io, go in the source code directory
and do::

    python setup.py install

or if you don't have admin access to your python setup (permission denied
when install) use::

    python setup.py install --user

You can also install the latest release version with pip::

    pip install h5io --upgrade

or for the latest development version (the most up to date)::

    pip install -e git+https://github.com/h5io/h5io#egg=h5io-dev --user

Dependencies
^^^^^^^^^^^^

The required dependencies to build the software are ``h5py``, ``numpy``,
and ``scipy``. Eventually the ``scipy`` requirement could be relaxed if
users need it.

Licensing
^^^^^^^^^

h5io is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011, authors of h5io
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of MNE-Python authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**
