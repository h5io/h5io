# -*- coding: utf-8 -*-
from os import path as op
from nose.tools import assert_raises, assert_true, assert_equal

import numpy as np
try:
    from scipy import sparse
except ImportError:
    sparse = None

from h5io import write_hdf5, read_hdf5, _TempDir, object_diff


def test_hdf5():
    """Test HDF5 IO
    """
    tempdir = _TempDir()
    test_file = op.join(tempdir, 'test.hdf5')
    sp = np.eye(3) if sparse is None else sparse.eye(3, 3, format='csc')
    sp[2, 2] = 2
    x = dict(a=dict(b=np.zeros(3)), c=np.zeros(2, np.complex128),
             d=[dict(e=(1, -2., 'hello', u'goodbyeu\u2764')), None], f=sp)
    write_hdf5(test_file, 1)
    assert_equal(read_hdf5(test_file), 1)
    assert_raises(IOError, write_hdf5, test_file, x)  # file exists
    write_hdf5(test_file, x, overwrite=True)
    assert_raises(IOError, read_hdf5, test_file + 'FOO')  # not found
    xx = read_hdf5(test_file)
    assert_true(object_diff(x, xx) == '')  # no assert_equal, ugly output

    # bad title
    assert_raises(ValueError, read_hdf5, test_file, title='nonexist')
    assert_raises(ValueError, write_hdf5, test_file, x, overwrite=True,
                  title=1)
    assert_raises(ValueError, read_hdf5, test_file, title=1)
    # unsupported objects
    assert_raises(TypeError, write_hdf5, test_file, {1: 'foo'},
                  overwrite=True)
    assert_raises(TypeError, write_hdf5, test_file, object, overwrite=True)


def test_object_diff():
    """Test object diff calculation
    """
    assert_true('type' in object_diff(1, 1.))
    assert_true('missing' in object_diff({1: 1}, {}))
    assert_true('missing' in object_diff({}, {1: 1}))
    assert_true('length' in object_diff([], [1]))
    assert_true('value' in object_diff('a', 'b'))
    assert_true('None' in object_diff(None, 'b'))
    assert_true('array mismatch' in object_diff(np.array([1]), np.array([2])))
    if sparse is not None:
        a = sparse.coo_matrix([[1]])
        b = sparse.coo_matrix([[1, 2]])
        assert_true('shape mismatch' in object_diff(a, b))
        c = sparse.coo_matrix([[1, 1]])
        assert_true('1 element' in object_diff(b, c))
    assert_raises(RuntimeError, object_diff, object, object)
