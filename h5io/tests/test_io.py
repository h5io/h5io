# -*- coding: utf-8 -*-
import datetime
from os import path as op
import pytest

import numpy as np
from numpy.testing import assert_equal
try:
    from scipy import sparse
except ImportError:
    sparse = None

try:
    from pandas import DataFrame, Series
except ImportError:
    DataFrame = Series = None

from h5io import (write_hdf5, read_hdf5,
                  object_diff, list_file_contents)


def test_hdf5(tmpdir):
    """Test HDF5 IO."""
    tempdir = str(tmpdir)
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
    pytest.raises(IOError, write_hdf5, test_file, x)  # file exists
    write_hdf5(test_file, x, overwrite=True)
    pytest.raises(IOError, read_hdf5, test_file + 'FOO')  # not found
    xx = read_hdf5(test_file)
    assert (object_diff(x, xx) == '')  # no assert_equal, ugly output
    list_file_contents(test_file)  # Testing the h5 listing
    pytest.raises(TypeError, list_file_contents, sp)  # Only string works
    write_hdf5(test_file, np.bool_(True), overwrite=True)
    assert_equal(read_hdf5(test_file), np.bool_(True))

    # bad title
    pytest.raises(ValueError, read_hdf5, test_file, title='nonexist')
    pytest.raises(ValueError, write_hdf5, test_file, x, overwrite=True,
                  title=1)
    pytest.raises(ValueError, read_hdf5, test_file, title=1)
    # unsupported objects
    pytest.raises(TypeError, write_hdf5, test_file, {1: 'foo'},
                  overwrite=True)
    pytest.raises(TypeError, write_hdf5, test_file, object, overwrite=True)
    # special_chars
    spec_dict = {'first/second': 'third'}
    pytest.raises(ValueError, write_hdf5, test_file, spec_dict, overwrite=True)
    pytest.raises(ValueError, write_hdf5, test_file, spec_dict, overwrite=True,
                  slash='brains')
    write_hdf5(test_file, spec_dict, overwrite=True, slash='replace')
    assert_equal(
        read_hdf5(test_file, slash='replace').keys(), spec_dict.keys())
    in_keys = list(read_hdf5(test_file, slash='ignore').keys())
    assert ('{FWDSLASH}' in in_keys[0])
    pytest.raises(ValueError, read_hdf5, test_file, slash='brains')
    # Testing that title slashes aren't replaced
    write_hdf5(
        test_file, spec_dict, title='one/two', overwrite=True, slash='replace')
    assert_equal(read_hdf5(test_file, title='one/two', slash='replace').keys(),
                 spec_dict.keys())

    write_hdf5(test_file, 1, title='first', overwrite=True)
    write_hdf5(test_file, 2, title='second', overwrite='update')
    assert_equal(read_hdf5(test_file, title='first'), 1)
    assert_equal(read_hdf5(test_file, title='second'), 2)
    pytest.raises(IOError, write_hdf5, test_file, 3, title='second')
    write_hdf5(test_file, 3, title='second', overwrite='update')
    assert_equal(read_hdf5(test_file, title='second'), 3)

    write_hdf5(test_file, 5, title='second', overwrite='update', compression=5)
    assert_equal(read_hdf5(test_file, title='second'), 5)


def test_hdf5_use_json(tmpdir):
    """Test HDF5 IO."""
    tempdir = str(tmpdir)
    test_file = op.join(tempdir, 'test.hdf5')
    splash_dict = {'first/second': {'one/more': 'value'}}
    pytest.raises(ValueError, write_hdf5, test_file, splash_dict,
                  overwrite=True, slash='error', use_json=True)
    spec_dict = {'first/second': 'third'}
    write_hdf5(test_file, spec_dict, overwrite=True, slash='replace',
               use_json=True)
    assert_equal(
        read_hdf5(test_file, slash='replace').keys(), spec_dict.keys())
    in_keys = list(read_hdf5(test_file, slash='ignore').keys())
    assert ('{FWDSLASH}' in in_keys[0])
    comp_dict = {'first': [1, 2], 'second': 'str', 'third': {'a': 1}}
    write_hdf5(test_file, comp_dict, overwrite=True, use_json=True)
    assert_equal(
        sorted(read_hdf5(test_file, slash='replace').keys()),
        sorted(comp_dict.keys()))
    numpy_dict = {'first': np.array([1])}
    write_hdf5(test_file, numpy_dict, overwrite=True, use_json=True)
    assert_equal(list(read_hdf5(test_file, slash='replace').values())[0],
                 list(numpy_dict.values())[0])
    pytest.raises(ValueError, read_hdf5, test_file, slash='brains')
    # Testing that title slashes aren't replaced
    write_hdf5(test_file, spec_dict, title='one/two', overwrite=True,
               slash='replace', use_json=True)
    assert_equal(read_hdf5(test_file, title='one/two', slash='replace').keys(),
                 spec_dict.keys())


def test_path_support(tmpdir):
    tempdir = str(tmpdir)
    test_file = op.join(tempdir, 'test.hdf5')
    write_hdf5(test_file, 1, title='first')
    write_hdf5(test_file, 2, title='second/third', overwrite='update')
    pytest.raises(ValueError, read_hdf5, test_file, title='second')
    assert_equal(read_hdf5(test_file, 'first'), 1)
    assert_equal(read_hdf5(test_file, 'second/third'), 2)


def test_object_diff():
    """Test object diff calculation."""
    assert ('type' in object_diff(1, 1.))
    assert ('missing' in object_diff({1: 1}, {}))
    assert ('missing' in object_diff({}, {1: 1}))
    assert ('length' in object_diff([], [1]))
    assert ('value' in object_diff('a', 'b'))
    assert ('None' in object_diff(None, 'b'))
    assert ('array mismatch' in object_diff(np.array([1]), np.array([2])))
    if sparse is not None:
        a = sparse.coo_matrix([[1]])
        b = sparse.coo_matrix([[1, 2]])
        assert ('shape mismatch' in object_diff(a, b))
        c = sparse.coo_matrix([[1, 1]])
        assert ('1 element' in object_diff(b, c))
    if not isinstance(DataFrame, type(None)):
        for ob_type in (DataFrame, Series):
            a = ob_type([1])
            b = ob_type([1, 2])
            assert ('shape mismatch' in object_diff(a, b))
            c = ob_type([1, 3])
            assert ('1 element' in object_diff(b, c))
    pytest.raises(RuntimeError, object_diff, object, object)


def test_numpy_values(tmpdir):
    """Test NumPy values."""
    test_file = op.join(str(tmpdir), 'test.hdf5')
    for cast in [np.int8, np.int16, np.int32, np.int64, np.bool_,
                 np.float16, np.float32, np.float64]:
        value = cast(1)
        write_hdf5(test_file, value, title='first', overwrite='update')
        assert_equal(read_hdf5(test_file, 'first'), value)


def test_multi_dim_array(tmpdir):
    """Test multidimensional arrays."""
    rng = np.random.RandomState(0)
    traj = np.array([rng.randn(2, 1), rng.randn(3, 1)], dtype=object)
    test_file = op.join(str(tmpdir), 'test.hdf5')
    write_hdf5(test_file, traj, title='first', overwrite='update')
    for traj_read, traj_sub in zip(read_hdf5(test_file, 'first'), traj):
        assert (np.equal(traj_read, traj_sub).all())
    traj_no_structure = np.array(
        [rng.randn(2, 1, 1), rng.randn(3, 1, 2)], dtype=object)
    pytest.raises(ValueError, write_hdf5, test_file, traj_no_structure,
                  title='second', overwrite='update')


class XT(datetime.tzinfo):

    def utcoffset(self, dt):
        return datetime.timedelta(hours=-5)  # Eastern on standard time

    def tzname(self, dt):
        return "UTC-05:00"

    def dst(self, dt):
        return None


def test_datetime(tmpdir):
    """Test datetime.datetime support."""
    fname = op.join(str(tmpdir), 'test.hdf5')
    # Naive
    y, m, d, h, m, s, mu = range(1, 8)
    dt = datetime.datetime(y, m, d, h, m, s, mu)
    for key in ('year', 'month', 'day', 'hour', 'minute', 'second',
                'microsecond'):
        val = locals()[key[:1] if key != 'microsecond' else 'mu']
        assert val == getattr(dt, key)
    assert dt.year == y
    assert dt.month == m
    write_hdf5(fname, dt)
    dt2 = read_hdf5(fname)
    assert isinstance(dt2, datetime.datetime)
    assert dt == dt2
    assert dt2.tzinfo is None
    # Aware
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    write_hdf5(fname, dt, overwrite=True)
    dt2 = read_hdf5(fname)
    assert isinstance(dt2, datetime.datetime)
    assert dt == dt2
    assert dt2.tzinfo is datetime.timezone.utc
    # Custom
    dt = dt.replace(tzinfo=XT())
    write_hdf5(fname, dt, overwrite=True)
    dt2 = read_hdf5(fname)
    assert isinstance(dt2, datetime.datetime)
    assert dt == dt2
    assert dt2.tzinfo is not None
    assert dt2.tzinfo is not datetime.timezone.utc
    for key in ('utcoffset', 'tzname', 'dst'):
        v1 = getattr(dt2.tzinfo, key)(None)
        v2 = getattr(dt.tzinfo, key)(None)
        assert v1 == v2


@pytest.mark.parametrize('name', (None, 'foo'))
def test_timezone(name, tmpdir):
    """Test datetime.timezone support."""
    fname = op.join(str(tmpdir), 'test.hdf5')
    kwargs = dict()
    if name is not None:
        kwargs['name'] = name
    x = datetime.timezone(datetime.timedelta(hours=-7), **kwargs)
    write_hdf5(fname, x)
    y = read_hdf5(fname)
    assert isinstance(y, datetime.timezone)
    assert y == x
    if name is not None:
        assert y.tzname(None) == name
