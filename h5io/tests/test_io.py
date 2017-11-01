# -*- coding: utf-8 -*-
from os import path as op
from nose.tools import assert_raises, assert_true, assert_equal

import numpy as np
try:
    from scipy import sparse
except ImportError:
    sparse = None

try:
    from pandas import DataFrame, Series
except ImportError:
    DataFrame = Series = None

from h5io import (write_hdf5, read_hdf5,
                  _TempDir, object_diff, list_file_contents)


def test_hdf5():
    """Test HDF5 IO
    """
    tempdir = _TempDir()
    test_file = op.join(tempdir, 'test.hdf5')
    sp = np.eye(3) if sparse is None else sparse.eye(3, 3, format='csc')
    sp_csr = np.eye(3) if sparse is None else sparse.eye(3, 3, format='csr')
    df = np.eye(3) if isinstance(DataFrame, type(None)) else DataFrame(
        np.eye(3))
    sr = np.eye(3) if isinstance(Series, type(None)) else Series(
        np.random.randn(3))
    sp[2, 2] = 2
    sp_csr[2, 2] = 2
    x = dict(a=dict(b=np.zeros(3)), c=np.zeros(2, np.complex128),
             d=[dict(e=(1, -2., 'hello', u'goodbyeu\u2764')), None], f=sp,
             g=dict(dfa=df, srb=sr), h=sp_csr, i=sr, j='hi')
    write_hdf5(test_file, 1)
    assert_equal(read_hdf5(test_file), 1)
    assert_raises(IOError, write_hdf5, test_file, x)  # file exists
    write_hdf5(test_file, x, overwrite=True)
    assert_raises(IOError, read_hdf5, test_file + 'FOO')  # not found
    xx = read_hdf5(test_file)
    assert_true(object_diff(x, xx) == '')  # no assert_equal, ugly output
    list_file_contents(test_file)  # Testing the h5 listing
    assert_raises(TypeError, list_file_contents, sp)  # Only string works
    write_hdf5(test_file, np.bool_(True), overwrite=True)
    assert_equal(read_hdf5(test_file), np.bool_(True))

    # bad title
    assert_raises(ValueError, read_hdf5, test_file, title='nonexist')
    assert_raises(ValueError, write_hdf5, test_file, x, overwrite=True,
                  title=1)
    assert_raises(ValueError, read_hdf5, test_file, title=1)
    # unsupported objects
    assert_raises(TypeError, write_hdf5, test_file, {1: 'foo'},
                  overwrite=True)
    assert_raises(TypeError, write_hdf5, test_file, object, overwrite=True)
    # special_chars
    spec_dict = {'first/second': 'third'}
    assert_raises(ValueError, write_hdf5, test_file, spec_dict, overwrite=True)
    assert_raises(ValueError, write_hdf5, test_file, spec_dict, overwrite=True,
                  slash='brains')
    write_hdf5(test_file, spec_dict, overwrite=True, slash='replace')
    assert_equal(
        read_hdf5(test_file, slash='replace').keys(), spec_dict.keys())
    in_keys = list(read_hdf5(test_file, slash='ignore').keys())
    assert_true('{FWDSLASH}' in in_keys[0])
    assert_raises(ValueError, read_hdf5, test_file, slash='brains')
    # Testing that title slashes aren't replaced
    write_hdf5(
        test_file, spec_dict, title='one/two', overwrite=True, slash='replace')
    assert_equal(read_hdf5(test_file, title='one/two', slash='replace').keys(),
                 spec_dict.keys())

    write_hdf5(test_file, 1, title='first', overwrite=True)
    write_hdf5(test_file, 2, title='second', overwrite='update')
    assert_equal(read_hdf5(test_file, title='first'), 1)
    assert_equal(read_hdf5(test_file, title='second'), 2)
    assert_raises(IOError, write_hdf5, test_file, 3, title='second')
    write_hdf5(test_file, 3, title='second', overwrite='update')
    assert_equal(read_hdf5(test_file, title='second'), 3)

    write_hdf5(test_file, 5, title='second', overwrite='update', compression=5)
    assert_equal(read_hdf5(test_file, title='second'), 5)


def test_path_support():
    tempdir = _TempDir()
    test_file = op.join(tempdir, 'test.hdf5')
    write_hdf5(test_file, 1, title='first')
    write_hdf5(test_file, 2, title='second/third', overwrite='update')
    assert_raises(ValueError, read_hdf5, test_file, title='second')
    assert_equal(read_hdf5(test_file, 'first'), 1)
    assert_equal(read_hdf5(test_file, 'second/third'), 2)


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
    if not isinstance(DataFrame, type(None)):
        for ob_type in (DataFrame, Series):
            a = ob_type([1])
            b = ob_type([1, 2])
            assert_true('shape mismatch' in object_diff(a, b))
            c = ob_type([1, 3])
            assert_true('1 element' in object_diff(b, c))
    assert_raises(RuntimeError, object_diff, object, object)


def test_numpy_values():
    tempdir = _TempDir()
    test_file = op.join(tempdir, 'test.hdf5')
    for cast in [np.int8, np.int16, np.int32, np.int64, np.bool_,
                 np.float16, np.float32, np.float64]:
        value = cast(1)
        write_hdf5(test_file, value, title='first', overwrite='update')
        assert_equal(read_hdf5(test_file, 'first'), value)
