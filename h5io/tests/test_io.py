"""Tests."""

import datetime
import sys
from abc import ABCMeta
from io import UnsupportedOperation
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_equal

try:
    from scipy import sparse
except ImportError:
    sparse = None

try:
    from pandas import DataFrame, Series
except ImportError:
    DataFrame = Series = None

import h5py

from h5io import list_file_contents, object_diff, read_hdf5, write_hdf5


def test_hdf5(tmp_path):
    """Test HDF5 IO."""
    pytest.importorskip("tables")
    test_file = str(tmp_path / "test.hdf5")
    sp = np.eye(3) if sparse is None else sparse.eye(3, 3, format="csc")
    sp_csr = np.eye(3) if sparse is None else sparse.eye(3, 3, format="csr")
    df = np.eye(3) if isinstance(DataFrame, type(None)) else DataFrame(np.eye(3))
    sr = np.eye(3) if isinstance(Series, type(None)) else Series(np.random.randn(3))
    sp[2, 2] = 2
    sp_csr[2, 2] = 2
    x = dict(
        a=dict(b=np.zeros(3)),
        c=np.zeros(2, np.complex128),
        d=[dict(e=(1, -2.0, "hello", "goodbyeu\u2764")), None],
        f=sp,
        g=dict(dfa=df, srb=sr),
        h=sp_csr,
        i=sr,
        j="hi",
    )
    write_hdf5(test_file, 1)
    assert_equal(read_hdf5(test_file), 1)
    pytest.raises(IOError, write_hdf5, test_file, x)  # file exists
    write_hdf5(Path(test_file), x, overwrite=True)
    pytest.raises(IOError, read_hdf5, test_file + "FOO")  # not found
    xx = read_hdf5(Path(test_file))
    assert object_diff(x, xx) == ""  # no assert_equal, ugly output
    list_file_contents(test_file)  # Testing the h5 listing
    pytest.raises(TypeError, list_file_contents, sp)  # Only string works
    write_hdf5(test_file, np.bool_(True), overwrite=True)
    assert_equal(read_hdf5(test_file), np.bool_(True))

    # bad title
    pytest.raises(ValueError, read_hdf5, test_file, title="nonexist")
    pytest.raises(ValueError, write_hdf5, test_file, x, overwrite=True, title=1)
    pytest.raises(ValueError, read_hdf5, test_file, title=1)
    # unsupported objects
    pytest.raises(TypeError, write_hdf5, test_file, {1: "foo"}, overwrite=True)
    # special_chars
    spec_dict = {"first/second": "third"}
    pytest.raises(ValueError, write_hdf5, test_file, spec_dict, overwrite=True)
    pytest.raises(
        ValueError, write_hdf5, test_file, spec_dict, overwrite=True, slash="brains"
    )
    write_hdf5(test_file, spec_dict, overwrite=True, slash="replace")
    assert_equal(read_hdf5(test_file, slash="replace").keys(), spec_dict.keys())
    in_keys = list(read_hdf5(test_file, slash="ignore").keys())
    assert "{FWDSLASH}" in in_keys[0]
    pytest.raises(ValueError, read_hdf5, test_file, slash="brains")
    # Testing that title slashes aren't replaced
    write_hdf5(test_file, spec_dict, title="one/two", overwrite=True, slash="replace")
    assert_equal(
        read_hdf5(test_file, title="one/two", slash="replace").keys(), spec_dict.keys()
    )

    write_hdf5(test_file, 1, title="first", overwrite=True)
    write_hdf5(test_file, 2, title="second", overwrite="update")
    assert_equal(read_hdf5(test_file, title="first"), 1)
    assert_equal(read_hdf5(test_file, title="second"), 2)
    pytest.raises(IOError, write_hdf5, test_file, 3, title="second")
    write_hdf5(test_file, 3, title="second", overwrite="update")
    assert_equal(read_hdf5(test_file, title="second"), 3)

    write_hdf5(test_file, 5, title="second", overwrite="update", compression=5)
    assert_equal(read_hdf5(test_file, title="second"), 5)


def test_h5_file_object(tmp_path):
    """Test file object support."""
    test_file_path = tmp_path / "test1.hdf5"
    # test that wrong object type raises error
    pytest.raises(ValueError, write_hdf5, fname=33, data=1)
    # test that reading/writing are unaffected
    with h5py.File(test_file_path, "a") as test_file_obj:
        data = {"a": 42}
        write_hdf5(test_file_obj, data)
        assert_equal(read_hdf5(test_file_obj), data)
    # test that wrong mode raises error
    with h5py.File(test_file_path, "r") as test_file_obj:
        assert test_file_obj.mode == "r"
        with pytest.raises(UnsupportedOperation):
            write_hdf5(test_file_obj, data=1)
    # at least on some OSes (e.g., macOS) opening with mode='w' leads to
    # test_file_obj.mode == 'r+', so let's skip this for now
    # with h5py.File(test_file_path, 'w') as test_file_obj:
    #     print(test_file_obj.mode)
    #     with pytest.raises(UnsupportedOperation):
    #         read_hdf5(test_file_obj)


def test_hdf5_use_json(tmp_path):
    """Test HDF5 IO."""
    test_file = tmp_path / "test.hdf5"
    splash_dict = {"first/second": {"one/more": "value"}}
    pytest.raises(
        ValueError,
        write_hdf5,
        test_file,
        splash_dict,
        overwrite=True,
        slash="error",
        use_json=True,
    )
    spec_dict = {"first/second": "third"}
    write_hdf5(test_file, spec_dict, overwrite=True, slash="replace", use_json=True)
    assert_equal(read_hdf5(test_file, slash="replace").keys(), spec_dict.keys())
    in_keys = list(read_hdf5(test_file, slash="ignore").keys())
    assert "{FWDSLASH}" in in_keys[0]
    comp_dict = {"first": [1, 2], "second": "str", "third": {"a": 1}}
    write_hdf5(test_file, comp_dict, overwrite=True, use_json=True)
    assert_equal(
        sorted(read_hdf5(test_file, slash="replace").keys()), sorted(comp_dict.keys())
    )
    numpy_dict = {"first": np.array([1])}
    write_hdf5(test_file, numpy_dict, overwrite=True, use_json=True)
    assert_equal(
        list(read_hdf5(test_file, slash="replace").values())[0],
        list(numpy_dict.values())[0],
    )
    pytest.raises(ValueError, read_hdf5, test_file, slash="brains")
    # Testing that title slashes aren't replaced
    write_hdf5(
        test_file,
        spec_dict,
        title="one/two",
        overwrite=True,
        slash="replace",
        use_json=True,
    )
    assert_equal(
        read_hdf5(test_file, title="one/two", slash="replace").keys(), spec_dict.keys()
    )


def test_path_support(tmp_path):
    """Test Path support."""
    test_file = tmp_path / "test.hdf5"
    write_hdf5(test_file, 1, title="first")
    write_hdf5(test_file, 2, title="second/third", overwrite="update")
    pytest.raises(ValueError, read_hdf5, test_file, title="second")
    assert_equal(read_hdf5(test_file, "first"), 1)
    assert_equal(read_hdf5(test_file, "second/third"), 2)


def test_object_diff():
    """Test object diff calculation."""
    assert "type" in object_diff(1, 1.0)
    assert "missing" in object_diff({1: 1}, {})
    assert "missing" in object_diff({}, {1: 1})
    assert "length" in object_diff([], [1])
    assert "value" in object_diff("a", "b")
    assert "None" in object_diff(None, "b")
    assert "array mismatch" in object_diff(np.array([1]), np.array([2]))
    if sparse is not None:
        a = sparse.coo_matrix([[1]])
        b = sparse.coo_matrix([[1, 2]])
        assert "shape mismatch" in object_diff(a, b)
        c = sparse.coo_matrix([[1, 1]])
        assert "1 element" in object_diff(b, c)
    if not isinstance(DataFrame, type(None)):
        for ob_type in (DataFrame, Series):
            a = ob_type([1])
            b = ob_type([1, 2])
            assert "shape mismatch" in object_diff(a, b)
            c = ob_type([1, 3])
            assert "1 element" in object_diff(b, c)
    pytest.raises(RuntimeError, object_diff, object, object)


def test_numpy_values(tmp_path):
    """Test NumPy values."""
    test_file = tmp_path / "test.hdf5"
    for cast in [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.bool_,
        np.float16,
        np.float32,
        np.float64,
        np.void,
    ]:
        value = cast(1)
        write_hdf5(test_file, value, title="first", overwrite="update")
        assert_equal(read_hdf5(test_file, "first"), value)


def test_multi_dim_array(tmp_path):
    """Test multidimensional arrays."""
    rng = np.random.RandomState(0)
    traj = np.array([rng.randn(2, 1), rng.randn(3, 1)], dtype=object)
    test_file = tmp_path / "test.hdf5"
    write_hdf5(test_file, traj, title="first", overwrite="update")
    for traj_read, traj_sub in zip(read_hdf5(test_file, "first"), traj):
        assert np.equal(traj_read, traj_sub).all()
    traj_no_structure = np.array([rng.randn(2, 1, 1), rng.randn(3, 1, 2)], dtype=object)
    pytest.raises(
        ValueError,
        write_hdf5,
        test_file,
        traj_no_structure,
        title="second",
        overwrite="update",
    )


class _XT(datetime.tzinfo):
    def utcoffset(self, dt):
        return datetime.timedelta(hours=-5)  # Eastern on standard time

    def tzname(self, dt):
        return "UTC-05:00"

    def dst(self, dt):
        return None


def test_datetime(tmp_path):
    """Test datetime.datetime support."""
    fname = tmp_path / "test.hdf5"
    # Naive
    y, m, d, h, m, s, mu = range(1, 8)
    dt = datetime.datetime(y, m, d, h, m, s, mu)
    for key in ("year", "month", "day", "hour", "minute", "second", "microsecond"):
        val = locals()[key[:1] if key != "microsecond" else "mu"]
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
    dt = dt.replace(tzinfo=_XT())
    write_hdf5(fname, dt, overwrite=True)
    dt2 = read_hdf5(fname)
    assert isinstance(dt2, datetime.datetime)
    assert dt == dt2
    assert dt2.tzinfo is not None
    assert dt2.tzinfo is not datetime.timezone.utc
    for key in ("utcoffset", "tzname", "dst"):
        v1 = getattr(dt2.tzinfo, key)(None)
        v2 = getattr(dt.tzinfo, key)(None)
        assert v1 == v2


@pytest.mark.parametrize("name", (None, "foo"))
def test_timezone(name, tmp_path):
    """Test datetime.timezone support."""
    fname = tmp_path / "test.hdf5"
    kwargs = dict()
    if name is not None:
        kwargs["name"] = name
    x = datetime.timezone(datetime.timedelta(hours=-7), **kwargs)
    write_hdf5(fname, x)
    y = read_hdf5(fname)
    assert isinstance(y, datetime.timezone)
    assert y == x
    if name is not None:
        assert y.tzname(None) == name


class SaveableClass:
    """A toy class to test saving and loading classes."""

    @classmethod
    def foo(cls):
        """Just a thing on our class to easily see we're getting what we ask for."""
        return 42


def test_class_storage(tmp_path):
    """Test storing a class."""
    test_file = tmp_path / "test.hdf5"

    # Built-in
    write_hdf5(
        fname=test_file,
        data=int,
        title="myclass",
        overwrite="update",
        use_json=False,
        use_state=False,
    )
    loaded_class = read_hdf5(fname=test_file, title="myclass")
    assert loaded_class is int

    # Std. Lib / importable class
    write_hdf5(
        fname=test_file,
        data=datetime.datetime,
        title="myclass",
        overwrite="update",
        use_json=False,
        use_state=False,
    )
    loaded_class = read_hdf5(fname=test_file, title="myclass")
    assert loaded_class is datetime.datetime

    # Local custom class
    write_hdf5(
        fname=test_file,
        data=SaveableClass,
        title="myclass",
        overwrite="update",
        use_json=False,
        use_state=False,
    )
    loaded_class = read_hdf5(fname=test_file, title="myclass")

    assert loaded_class is SaveableClass
    assert loaded_class.foo() == SaveableClass.foo()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires python3.11 or higher")
def test_state_with_numpy_poly1d(tmp_path):
    """The np.poly1d() object has an instance __dict__."""
    test_file = tmp_path / "test.hdf5"
    polyfit_obj = np.poly1d(np.polyfit([1, 2], [1, 2], 1))
    pytest.raises(
        TypeError,
        write_hdf5,
        test_file,
        polyfit_obj,
        title="polyfit",
        overwrite="update",
    )
    write_hdf5(
        fname=test_file,
        data=polyfit_obj,
        title="polyfit",
        overwrite="update",
        use_json=False,
        use_state=True,
    )
    polyfit_obj_reload = read_hdf5(
        fname=test_file,
        title="polyfit",
        slash="ignore",
    )
    assert polyfit_obj == polyfit_obj_reload
    write_hdf5(
        fname=test_file,
        data=polyfit_obj,
        title="polyfit_json",
        overwrite="update",
        use_json=True,
        use_state=True,
    )
    polyfit_obj_reload = read_hdf5(
        fname=test_file,
        title="polyfit_json",
        slash="ignore",
    )
    assert polyfit_obj == polyfit_obj_reload


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires python3.11 or higher")
def test_state_with_pathlib(tmp_path):
    """The Path object has __slots__ and no instance __dict__."""
    test_file = tmp_path / "test.hdf5"
    path_obj = Path(tmp_path)
    pytest.raises(
        TypeError,
        write_hdf5,
        test_file,
        path_obj,
        title="pathobj",
        overwrite="update",
    )
    write_hdf5(
        fname=test_file,
        data=path_obj,
        title="pathobj",
        overwrite="update",
        use_json=False,
        use_state=True,
    )
    path_obj_reload = read_hdf5(
        fname=test_file,
        title="pathobj",
        slash="ignore",
    )
    assert path_obj == path_obj_reload
    write_hdf5(
        fname=test_file,
        data=path_obj,
        title="pathobj_json",
        overwrite="update",
        use_json=True,
        use_state=True,
    )
    path_obj_reload = read_hdf5(
        fname=test_file,
        title="pathobj_json",
        slash="ignore",
    )
    assert path_obj == path_obj_reload


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires python3.11 or higher")
def test_state_with_dynamic_class(tmp_path):
    """A dynamically defined class cannot be stored in HDF5."""
    test_file = tmp_path / "test.hdf5"

    class MyClass:
        a = 1

    my_obj = MyClass()
    pytest.raises(
        TypeError,
        write_hdf5,
        test_file,
        my_obj,
        title="myobj",
        overwrite="update",
    )
    write_hdf5(
        fname=test_file,
        data=my_obj,
        title="myobj",
        overwrite="update",
        use_json=False,
        use_state=True,
    )
    pytest.raises(
        AttributeError,
        read_hdf5,
        fname=test_file,
        title="myobj",
        slash="ignore",
    )

    for nested in [[my_obj], (my_obj,), {"my_obj": my_obj}]:
        pytest.raises(
            TypeError,
            write_hdf5,
            test_file,
            nested,
            title="nested",
            overwrite="update",
        )
        write_hdf5(
            fname=test_file,
            data=nested,
            title="nested",
            overwrite="update",
            use_json=False,
            use_state=True,
        )


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="only python3.11 or lower")
def test_state_python_version_error(tmp_path):
    """Raise RuntimeError when use_state is used with an invalid python version."""
    test_file = tmp_path / "test.hdf5"
    path_obj = Path(tmp_path)
    pytest.raises(
        RuntimeError,
        write_hdf5,
        test_file,
        path_obj,
        title="pathobj",
        overwrite="update",
        use_state=True,
    )


class StringReduceMismatch:
    """A class to test the case of `__reduce__` returning a string."""

    def __reduce__(self):
        """
        Return a string associated with a local object to return.

        Pickle requires that the object found by importing a returned reduce string
        be the _same object_ getting reduced. We follow this requirement and expect
        instances of this class to not store.
        """
        return "what_gets_loaded"  # The associated global variable


what_gets_loaded = {"the_point_is": "this is not the same object"}


class Singleton(ABCMeta):
    """A singleton metaclass for testing when `__reduce__` returns a string."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """If there is an existing instance, return that instead of a new one."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonReduce(metaclass=Singleton):
    """A singleton class for testing when `__reduce__` returns a string."""

    def __reduce__(self):
        """Return a global."""
        return "MY_SINGLETON"


MY_SINGLETON = SingletonReduce()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires python3.11 or higher")
def test_state_with_singleton(tmp_path):
    """When __reduce__ returns a string, load the identical object."""
    test_file = tmp_path / "test.hdf5"

    string_reduce_mismatch_instance = StringReduceMismatch()
    assert string_reduce_mismatch_instance is not what_gets_loaded  # Sanity check
    pytest.raises(
        ValueError,
        write_hdf5,
        fname=test_file,
        data=string_reduce_mismatch_instance,
        overwrite=True,
        use_state=True,
    )  # Should not be allowed to save when the __reduce__ string does not give
    # the same object that's being reduced

    singleton_string_reduce = SingletonReduce()
    write_hdf5(
        fname=test_file,
        data=singleton_string_reduce,
        overwrite=True,
        use_json=False,
        use_state=True,
    )

    reloaded = read_hdf5(fname=test_file)
    assert reloaded is MY_SINGLETON  # The _exact_ object specified in __reduce__
