# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import copyreg
import datetime
import importlib
import inspect
import json
import sys
from inspect import isclass
from io import UnsupportedOperation
from os import path as op
from pathlib import PurePath

import numpy as np

_path_like = (str, PurePath)

special_chars = {"{FWDSLASH}": "/"}
tab_str = "----"

_SPARSE_KINDS = ("csc_matrix", "csr_matrix", "csc_array", "csr_array")


def _import_sparse():
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    return sparse


##############################################################################
# WRITING


def _check_h5py():
    """Check if h5py is installed."""
    try:
        import h5py
    except ImportError:
        raise ImportError("the h5py module is required to use HDF5 I/O")
    return h5py


def _create_titled_group(root, key, title):
    """Create a titled group in h5py."""
    out = root.create_group(key)
    out.attrs["TITLE"] = title
    return out


def _create_titled_dataset(root, key, title, data, comp_kw=None):
    """Create a titled dataset in h5py."""
    comp_kw = {} if comp_kw is None else comp_kw
    out = root.create_dataset(key, data=data, **comp_kw)
    out.attrs["TITLE"] = title
    return out


def _create_pandas_dataset(fname, root, key, title, data):
    h5py = _check_h5py()
    rootpath = "/".join([root, key])
    if isinstance(fname, h5py.File):
        # pandas requires full control over the HDF5 file.
        # The HDF5 file is closed and re-opened.
        file_name = fname.filename
        fname.close()
        # handover control to pandas to write dataset
        data.to_hdf(file_name, key=rootpath)
        # Re-open HDF5 file - requires access to internal variable _id
        fname._id = h5py.File(name=file_name, mode="a").id
        # Continue by setting the TITLE attribute
        fname[rootpath].attrs["TITLE"] = "pd_dataframe"
    else:
        data.to_hdf(fname, key=rootpath)
        with h5py.File(fname, mode="a") as fid:
            fid[rootpath].attrs["TITLE"] = "pd_dataframe"


def write_hdf5(
    fname,
    data,
    overwrite=False,
    compression=4,
    title="h5io",
    slash="error",
    use_json=False,
    use_state=False,
):
    """Write python object to HDF5 format using h5py.

    Parameters
    ----------
    fname : str
        Filename to use.
    data : object
        Object to write. Can be of any of these types:

            {ndarray, dict, list, tuple, int, float, str, datetime, date, timezone}

        Note that dict objects must only have ``str`` keys. It is recommended
        to use ndarrays where possible, as it is handled most efficiently.
    overwrite : True | False | 'update'
        If True, overwrite file (if it exists). If 'update', appends the title
        to the file (or replace value if title exists).
    compression : int
        Compression level to use (0-9) to compress data using gzip.
    title : str
        The top-level directory name to use. Typically it is useful to make
        this your package name, e.g. ``'mnepython'``.
    slash : 'error' | 'replace'
        Whether to replace forward-slashes ('/') in any key found nested within
        keys in data. This does not apply to the top level name (title).
        If 'error', '/' is not allowed in any lower-level keys.
    use_json : bool
        To accelerate the read and write performance of small dictionaries and
        lists they can be combined to JSON objects and stored as strings.
    use_state: bool
        To store objects of unsupported types the `__getstate__()` and other dunder
        methods (`__reduce__()`, `__getnewargs__()`  etc.) are used to  retrieve a
        dictionary which defines the state of the object and store the content of this
        dictionary in the HDF5 file. The interface strives to behave like `pickle`,
        including the same fragility to differences in the environment between save-
        and load-time -- i.e. if modules and classes are not consistent between saving
        and loading the object may fail to load, or load yet still be different from
        the saved object. Since this flag also -- beneficially -- makes it easy to save
        and load your own custom classes, extra care must be taken when the class being
        saved is in the module `__main__`, e.g. when working inside a jupyter notebook.
        In this case, care should be taken that any relevant classes are re-declared
        prior to loading. (Requires python >=3.11)
    """
    h5py = _check_h5py()
    if use_state and sys.version_info < (3, 11):
        raise RuntimeError(
            "The use_state parameter requires Python >= 3.11, as Python 3.11 "
            "added the default implementation of the __getstate__() method in "
            "the object class."
        )
    if isinstance(fname, _path_like):
        mode = "w"
        if op.isfile(fname):
            if isinstance(overwrite, str):
                if overwrite != "update":
                    raise ValueError('overwrite must be "update" or a bool')
                mode = "a"
            elif not overwrite:
                raise IOError(
                    'file "%s" exists, use overwrite=True to overwrite' % fname
                )
    elif isinstance(fname, h5py.File):
        if fname.mode == "r":
            raise UnsupportedOperation("not writable")
    else:
        raise ValueError(f"fname must be str or h5py.File, got {type(fname)}")
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    comp_kw = dict()
    if compression > 0:
        comp_kw = dict(compression="gzip", compression_opts=compression)

    def _write(fid, cleanup_data):
        if title in fid:
            del fid[title]
        _triage_write(
            title,
            data,
            fid,
            comp_kw,
            str(type(data)),
            cleanup_data,
            slash=slash,
            title=title,
            use_json=use_json,
            use_state=use_state,
        )

    cleanup_data = []
    if isinstance(fname, h5py.File):
        _write(fname, cleanup_data)
    else:
        with h5py.File(fname, mode=mode) as fid:
            _write(fid, cleanup_data)

    # Will not be empty if any extra data to be written
    for data in cleanup_data:
        # In case different extra I/O needs different inputs
        title = list(data.keys())[0]
        if title in ["pd_dataframe", "pd_series"]:
            rootname, key, value = data[title]
            _create_pandas_dataset(fname, rootname, key, title, value)


def _triage_write(
    key,
    value,
    root,
    comp_kw,
    where,
    cleanup_data,
    slash="error",
    title=None,
    use_json=False,
    use_state=False,
):
    sparse = _import_sparse()
    if key != title and "/" in key:
        if slash == "error":
            raise ValueError(
                'Found a key with "/", this is not allowed if slash == error'
            )
        elif slash == "replace":
            # Auto-replace keys with proper values
            for key_spec, val_spec in special_chars.items():
                key = key.replace(val_spec, key_spec)
        else:
            raise ValueError("slash must be one of ['error', 'replace'")

    if (
        use_json
        and isinstance(value, (list, dict))
        and _json_compatible(value, slash=slash)
    ):
        value = np.frombuffer(json.dumps(value).encode("utf-8"), np.uint8)
        _create_titled_dataset(root, key, "json", value, comp_kw)
    elif isinstance(value, dict):
        sub_root = _create_titled_group(root, key, "dict")
        for key, sub_value in value.items():
            if not isinstance(key, str):
                raise TypeError("All dict keys must be strings")
            _triage_write(
                "key_{0}".format(key),
                sub_value,
                sub_root,
                comp_kw,
                where + '["%s"]' % key,
                cleanup_data=cleanup_data,
                slash=slash,
                use_state=use_state,
            )
    elif isinstance(value, (list, tuple)):
        title = "list" if isinstance(value, list) else "tuple"
        sub_root = _create_titled_group(root, key, title)
        for vi, sub_value in enumerate(value):
            _triage_write(
                "idx_{0}".format(vi),
                sub_value,
                sub_root,
                comp_kw,
                where + "[%s]" % vi,
                cleanup_data=cleanup_data,
                slash=slash,
                use_state=use_state,
            )
    elif isinstance(value, type(None)):
        _create_titled_dataset(root, key, "None", [False])
    elif isclass(value):
        class_str = value.__module__ + "." + value.__name__
        _create_titled_dataset(
            root, key, "class", np.frombuffer(class_str.encode("utf-8"), np.uint8)
        )
    elif isinstance(value, (int, float)):
        if isinstance(value, int):
            title = "int"
        else:  # isinstance(value, float):
            title = "float"
        _create_titled_dataset(root, key, title, np.atleast_1d(value))
    elif isinstance(value, datetime.datetime):
        title = "datetime"
        value = np.frombuffer(value.isoformat().encode("utf-8"), np.uint8)
        _create_titled_dataset(root, key, title, value)
    elif isinstance(value, datetime.date):
        title = "date"
        value = np.frombuffer(value.isoformat().encode("utf-8"), np.uint8)
        _create_titled_dataset(root, key, title, value)
    elif isinstance(value, datetime.timezone):
        title = "timezone"  # the __repr__ is complete
        value = np.frombuffer(repr(value).encode("utf-8"), np.uint8)
        _create_titled_dataset(root, key, title, value)
    elif isinstance(value, (np.integer, np.floating, np.bool_)):
        title = "np_{0}".format(value.__class__.__name__)
        _create_titled_dataset(root, key, title, np.atleast_1d(value))
    elif isinstance(value, str):
        if isinstance(value, str):  # unicode
            value = np.frombuffer(value.encode("utf-8"), np.uint8)
            title = "unicode"
        else:
            value = np.frombuffer(value.encode("ASCII"), np.uint8)
            title = "ascii"
        _create_titled_dataset(root, key, title, value, comp_kw)
    elif isinstance(value, np.ndarray):
        if not (
            value.dtype == np.dtype("object")
            and len(set([sub.dtype for sub in value])) == 1
        ):
            _create_titled_dataset(root, key, "ndarray", value)
        else:
            ma_index, ma_data = multiarray_dump(value)
            sub_root = _create_titled_group(root, key, "multiarray")
            _create_titled_dataset(sub_root, "index", "ndarray", ma_index)
            _create_titled_dataset(sub_root, "data", "ndarray", ma_data)
    elif isinstance(value, np.void):
        # Based on https://docs.h5py.org/en/stable/strings.html#how-to-store-raw-binary-data
        _create_titled_dataset(root, key, "void", value)
    elif sparse is not None and any(
        isinstance(value, getattr(sparse, kind, type(None))) for kind in _SPARSE_KINDS
    ):
        for kind in _SPARSE_KINDS:
            if isinstance(value, getattr(sparse, kind)):
                break
        sub_root = _create_titled_group(root, key, kind)
        _triage_write(
            "data",
            value.data,
            sub_root,
            comp_kw,
            f"{where}.{kind}_data",
            cleanup_data=cleanup_data,
            slash=slash,
            use_state=use_state,
        )
        _triage_write(
            "indices",
            value.indices,
            sub_root,
            comp_kw,
            f"{where}.{kind}_indices",
            cleanup_data=cleanup_data,
            slash=slash,
            use_state=use_state,
        )
        _triage_write(
            "indptr",
            value.indptr,
            sub_root,
            comp_kw,
            f"{where}.{kind}_indptr",
            cleanup_data=cleanup_data,
            slash=slash,
            use_state=use_state,
        )
        _triage_write(
            "shape",
            value.shape,
            sub_root,
            comp_kw,
            f"{where}.{kind}_shape",
            cleanup_data=cleanup_data,
            slash=slash,
            use_state=use_state,
        )
    else:
        try:
            from pandas import DataFrame, Series
        except ImportError:
            pass
        else:
            if isinstance(value, (DataFrame, Series)):
                if isinstance(value, DataFrame):
                    title = "pd_dataframe"
                else:
                    title = "pd_series"
                rootname = root.name
                cleanup_data.append({title: (rootname, key, value)})
                return

        if use_state:
            class_type = value.__class__.__module__ + "." + value.__class__.__name__
            reduced = value.__reduce__()
            # Some objects reduce to simply the reconstructor function and its
            # arguments, without any state
            if isinstance(reduced, str):
                # https://docs.python.org/3/library/pickle.html#object.__reduce__
                # > If a string is returned, the string should be interpreted as the
                # > name of a global variable. It should be the object’s local name
                # > relative to its module; the pickle module searches the module
                # > namespace to determine the object’s module. This behaviour is
                # > typically useful for singletons.
                # In this case, override the class type to get the global, and manually
                # set the reconstructor variable so this doesn't look "custom"
                class_type = value.__class__.__module__ + "." + reduced

                reconstructor = copyreg._reconstructor
                state = value.__getstate__()
                additional = []
            elif len(reduced) == 2:
                # some objects do not return their internal state via
                # __reduce__, but can be reconstructed anyway by assigned the
                # return value from __getstate__ to __dict__, so we call it
                # here again anyway
                reconstructor, state, additional = reduced[0], value.__getstate__(), []
            else:
                reconstructor, _, state, *additional = reduced
            # For plain objects defining a simple __getstate__ python uses a
            # default reconstruction function defined in the copyreg module, if
            # an object wants to be reconstructed in any other way, we don't
            # know how to save this function in a file, so raise an error here
            # to avoid failure on reading from HDF5 files.
            # The same reasoning applies to objects returning more than 3
            # values from __reduce__.  This requests for additional logic on
            # reconstruction of the object (documented in the pickle module)
            # that we don't implement currently in the _triage_read function
            is_custom = (
                reconstructor is not type(value)
                and reconstructor.__module__ != "copyreg"
            )
            if is_custom or len(additional) != 0:
                raise TypeError(
                    f"Can't write {repr(value)} at location {key}:\n"
                    f"Class {class_type} defines custom reconstructor."
                )

            sub_root = _create_titled_group(root, key, class_type)

            # Based on https://docs.python.org/3/library/pickle.html#object.__getstate__
            # Requires python >= 3.11 as python 3.11 added the default implementation
            # of the __getstate__() method in the object class.
            if state is None:
                # For a class that has no instance __dict__ and no __slots__,
                # the default state is None.
                _guard_string_reductions(reduced, value, class_type, {})
                return
            elif isinstance(state, dict):
                # For a class that has an instance __dict__ and no __slots__,
                # the default state is self.__dict__.
                state_dict = state
            elif isinstance(state, tuple) and isinstance(state[0], dict):
                # For a class that has an instance __dict__ and __slots__, the
                # default state is a tuple consisting of two dictionaries:
                # self.__dict__, and a dictionary mapping slot names to slot
                # values. Only slots that have a value are included in the latter.
                state_dict = state[0]
            elif (
                isinstance(state, tuple)
                and state[0] is None
                and isinstance(state[1], dict)
            ):
                # For a class that has __slots__ and no instance __dict__, the
                # default state is a tuple whose first item is None and whose
                # second item is a dictionary mapping slot names to slot values
                # described in the previous bullet.
                state_dict = state[1]
            else:
                # When the __getstate__() was overwritten and no dict is
                # returned raise a TypeError
                raise TypeError("__getstate__() did not return a state dictionary.")

            _guard_string_reductions(reduced, value, class_type, state_dict)

            for key, value in state_dict.items():
                _triage_write(
                    key,
                    value,
                    sub_root,
                    comp_kw,
                    where,
                    cleanup_data=cleanup_data,
                    slash=slash,
                    use_json=use_json,
                    use_state=use_state,
                )
            return

        err_str = "unsupported type %s (in %s)" % (type(value), where)
        raise TypeError(err_str)


##############################################################################
# READING


def read_hdf5(fname, title="h5io", slash="ignore"):
    """Read python object from HDF5 format using h5py.

    Parameters
    ----------
    fname : str
        File to load.
    title : str
        The top-level directory name to use. Typically it is useful to make
        this your package name, e.g. ``'mnepython'``.
    slash : 'ignore' | 'replace'
        Whether to replace the string {FWDSLASH} with the value /. This does
        not apply to the top level name (title). If 'ignore', nothing will be
        replaced.

    Returns
    -------
    data : object
        The loaded data. Can be of any type supported by ``write_hdf5``.
    """
    h5py = _check_h5py()
    if isinstance(fname, _path_like):
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
    elif isinstance(fname, h5py.File):
        if fname.mode == "w":
            raise UnsupportedOperation('file must not be opened be opened with "w"')
    else:
        raise ValueError(f"fname must be str or h5py.File, got {type(fname)}")
    if not isinstance(title, str):
        raise ValueError("title must be a string")

    def _read(fid):
        if title not in fid:
            raise ValueError('no "%s" data found' % title)
        if isinstance(fid[title], h5py.Group):
            if "TITLE" not in fid[title].attrs:
                raise ValueError('no "%s" data found' % title)
        return _triage_read(fid[title], slash=slash)

    if isinstance(fname, h5py.File):
        return _read(fname)
    else:
        with h5py.File(fname, mode="r") as fid:
            return _read(fid)


def _triage_read(node, slash="ignore"):
    if slash not in ["ignore", "replace"]:
        raise ValueError("slash must be one of 'replace', 'ignore'")
    h5py = _check_h5py()
    sparse = _import_sparse()
    type_str = node.attrs["TITLE"]
    if isinstance(type_str, bytes):
        type_str = type_str.decode()
    if isinstance(node, h5py.Group):
        if type_str == "dict":
            data = dict()
            for key, subnode in node.items():
                if slash == "replace":
                    for key_spec, val_spec in special_chars.items():
                        key = key.replace(key_spec, val_spec)
                data[key[4:]] = _triage_read(subnode, slash=slash)
        elif type_str in ["list", "tuple"]:
            data = list()
            ii = 0
            while True:
                subnode = node.get("idx_{0}".format(ii), None)
                if subnode is None:
                    break
                data.append(_triage_read(subnode, slash=slash))
                ii += 1
            assert len(data) == ii
            data = tuple(data) if type_str == "tuple" else data
            return data
        elif type_str in _SPARSE_KINDS:
            if sparse is None:
                raise RuntimeError("scipy must be installed to read this data")
            klass = getattr(sparse, type_str)
            shape = None
            if "shape" in node:  # backward compat with old versions that didn't write
                shape = _triage_read(node["shape"], slash=slash)
            data = klass(
                (
                    _triage_read(node["data"], slash=slash),
                    _triage_read(node["indices"], slash=slash),
                    _triage_read(node["indptr"], slash=slash),
                ),
                shape=shape,
            )
        elif type_str in ["pd_dataframe", "pd_series"]:
            from pandas import HDFStore, read_hdf

            rootname = node.name
            filename = node.file.filename
            with HDFStore(filename, "r") as tmpf:
                data = read_hdf(tmpf, rootname)
        elif type_str == "multiarray":
            ma_index = _triage_read(node.get("index", None), slash=slash)
            ma_data = _triage_read(node.get("data", None), slash=slash)
            data = multiarray_load(ma_index, ma_data)
        elif sys.version_info >= (3, 11):
            # Requires python >= 3.11 as python 3.11 added the default implementation
            # of the __getstate__() method in the object class.
            # Based on https://docs.python.org/3/library/pickle.html#object.__getstate__
            return _setstate(
                obj_class=_import_class(class_type=type_str),
                state_dict={
                    n: _triage_read(node[n], slash="ignore") for n in list(node.keys())
                },
            )
        else:
            raise NotImplementedError("Unknown group type: {0}".format(type_str))
    elif type_str == "ndarray":
        data = np.array(node)
    elif type_str == "void":
        # Based on https://docs.h5py.org/en/stable/strings.html#how-to-store-raw-binary-data
        data = np.void(node)
    elif type_str in ("int", "float"):
        cast = int if type_str == "int" else float
        data = cast(np.array(node)[0])
    elif type_str == "datetime":
        data = str(np.array(node).tobytes().decode("utf-8"))
        data = datetime.datetime.fromisoformat(data)
    elif type_str == "date":
        data = str(np.array(node).tobytes().decode("utf-8"))
        data = datetime.date.fromisoformat(data)
    elif type_str == "timezone":
        data = eval(
            str(np.array(node).tobytes().decode("utf-8")), {"datetime": datetime}
        )
    elif type_str.startswith("np_"):
        np_type = type_str.split("_")[1]
        cast = getattr(np, np_type) if np_type != "bool" else bool
        data = np.array(node)[0].astype(cast)
    elif type_str in ("unicode", "ascii", "str"):  # 'str' for backward compat
        decoder = "utf-8" if type_str == "unicode" else "ASCII"
        data = str(np.array(node).tobytes().decode(decoder))
    elif type_str == "json":
        node_unicode = str(np.array(node).tobytes().decode("utf-8"))
        data = json.loads(node_unicode)
    elif type_str == "None":
        data = None
    elif type_str == "class":
        class_str = str(np.array(node).tobytes().decode("utf-8"))
        data = _import_class(class_str)
    else:
        raise TypeError("Unknown node type: {0}".format(type_str))
    return data


# ############################################################################
# UTILITIES


def _sort_keys(x):
    """Sort and return keys of dict."""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


def object_diff(a, b, pre=""):
    """Compute all differences between two python variables.

    Parameters
    ----------
    a : object
        Currently supported: dict, list, tuple, ndarray, int, str, bytes,
        float.
    b : object
        Must be same type as x1.
    pre : str
        String to prepend to each line.

    Returns
    -------
    diffs : str
        A string representation of the differences.
    """
    sparse = _import_sparse()
    try:
        from pandas import DataFrame, Series
    except ImportError:
        DataFrame = Series = type(None)

    out = ""
    if type(a) is not type(b):
        out += pre + " type mismatch (%s, %s)\n" % (type(a), type(b))
    elif isinstance(a, dict):
        k1s = _sort_keys(a)
        k2s = _sort_keys(b)
        m1 = set(k2s) - set(k1s)
        if len(m1):
            out += pre + " x1 missing keys %s\n" % (m1)
        for key in k1s:
            if key not in k2s:
                out += pre + " x2 missing key %s\n" % key
            else:
                out += object_diff(a[key], b[key], pre + "d1[%s]" % repr(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out += pre + " length mismatch (%s, %s)\n" % (len(a), len(b))
        else:
            for xx1, xx2 in zip(a, b):
                out += object_diff(xx1, xx2, pre="")
    elif isinstance(a, (str, int, float, bytes)):
        if a != b:
            out += pre + " value mismatch (%s, %s)\n" % (a, b)
    elif a is None:
        pass  # b must be None due to our type checking
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            out += pre + " array mismatch\n"
    elif sparse is not None and sparse.isspmatrix(a):
        # sparsity and sparse type of b vs a already checked above by type()
        if b.shape != a.shape:
            out += pre + (
                " sparse matrix a and b shape mismatch(%s vs %s)" % (a.shape, b.shape)
            )
        else:
            c = a - b
            c.eliminate_zeros()
            if c.nnz > 0:
                out += pre + (" sparse matrix a and b differ on %s elements" % c.nnz)
    elif isinstance(a, (DataFrame, Series)):
        if b.shape != a.shape:
            out += pre + (
                " pandas values a and b shape mismatch(%s vs %s)" % (a.shape, b.shape)
            )
        else:
            c = a.values - b.values
            nzeros = np.sum(c != 0)
            if nzeros > 0:
                out += pre + (" pandas values a and b differ on %s elements" % nzeros)
    else:
        raise RuntimeError(pre + ": unsupported type %s (%s)" % (type(a), a))
    return out


def _list_file_contents(h5file):
    if "h5io" not in h5file.keys():
        raise ValueError("h5file must contain h5io data")

    # Set up useful variables for later
    h5file = h5file["h5io"]
    root_title = h5file.attrs["TITLE"]
    n_space = (
        np.max([(len(key), len(val.attrs["TITLE"])) for key, val in h5file.items()]) + 2
    )

    # Create print strings
    strs = ["Root type: %s | Items: %s\n" % (root_title, len(h5file))]
    for key, data in h5file.items():
        type_str = data.attrs["TITLE"]
        str_format = "%%-%ss" % n_space
        if type_str == "ndarray":
            desc = "Shape: %s"
            desc_val = data.shape
        elif type_str in ["pd_dataframe", "pd_series"]:
            desc = "Shape: %s"
            desc_val = data["values"].shape
        elif type_str in ("unicode", "ascii", "str"):
            desc = "Text: %s"
            decoder = "utf-8" if type_str == "unicode" else "ASCII"
            data = str(np.array(data).tobytes().decode(decoder))
            desc_val = data[:10] + "..." if len(data) > 10 else data
        else:
            desc = "Items: %s"
            desc_val = len(data)
        this_str = ("%%s Key: %s | Type: %s | " + desc) % (
            str_format,
            str_format,
            str_format,
        )
        this_str = this_str % (tab_str, key, type_str, desc_val)
        strs.append(this_str)
    out_str = "\n".join(strs)
    print(out_str)


def list_file_contents(h5file):
    """List the contents of an h5io file.

    This will list the root and one-level-deep contents of the file.

    Parameters
    ----------
    h5file : str
        The path to an h5io hdf5 file.
    """
    h5py = _check_h5py()
    err = "h5file must be an h5py File object, not {0}"
    if isinstance(h5file, str):
        with h5py.File(h5file, "r") as f:
            _list_file_contents(f)
    else:
        if not isinstance(h5file, h5py.File):
            raise TypeError(err.format(type(h5file)))
        _list_file_contents(h5file)


def _json_compatible(obj, slash="error"):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return True
    elif isinstance(obj, list):
        return all([_json_compatible(item) for item in obj])
    elif isinstance(obj, dict):
        _check_keys_in_dict(obj, slash=slash)
        return all([_json_compatible(item) for item in obj.values()])
    else:
        return False


def _check_keys_in_dict(obj, slash="error"):
    repl = list()
    for key in obj.keys():
        if "/" in key:
            key_prev = key
            if slash == "error":
                raise ValueError(
                    'Found a key with "/", this is not allowed if slash == error'
                )
            elif slash == "replace":
                # Auto-replace keys with proper values
                for key_spec, val_spec in special_chars.items():
                    key = key.replace(val_spec, key_spec)
                repl.append((key, key_prev))
            else:
                raise ValueError("slash must be one of ['error', 'replace'")
    for key, key_prev in repl:
        obj[key] = obj.pop(key_prev)


##############################################################################
# Arrays with mixed dimensions
def _validate_object_array(array):
    if not (
        array.dtype == np.dtype("object")
        and len(set([sub.dtype for sub in array])) == 1
    ):
        raise TypeError("unsupported array type")


def _shape_list(array):
    return [np.shape(sub) for sub in array]


def _validate_sub_shapes(shape_lst):
    if not all([shape_lst[0][1:] == t[1:] for t in shape_lst]):
        raise ValueError("shape does not match!")


def _array_index(shape_lst):
    return [t[0] for t in shape_lst]


def _index_sum(index_lst):
    index_sum_lst = []
    for step in index_lst:
        if index_sum_lst != []:
            index_sum_lst.append(index_sum_lst[-1] + step)
        else:
            index_sum_lst.append(step)
    return index_sum_lst


def _merge_array(array):
    merged_lst = []
    for sub in array:
        merged_lst += sub.tolist()
    return np.array(merged_lst)


def multiarray_dump(array):
    _validate_object_array(array)
    shape_lst = _shape_list(array)
    _validate_sub_shapes(shape_lst=shape_lst)
    index_sum = _index_sum(index_lst=_array_index(shape_lst=shape_lst))
    return index_sum, _merge_array(array=array)


def multiarray_load(index, array_merged):
    array_restore = []
    i_prev = 0
    for i in index[:-1]:
        array_restore.append(array_merged[i_prev:i])
        i_prev = i
    array_restore.append(array_merged[i_prev:])
    return np.array(array_restore, dtype=object)


def _import_class(class_type):
    module_path, class_name = class_type.rsplit(".", maxsplit=1)
    return getattr(
        importlib.import_module(module_path),
        class_name,
    )


def _setstate(obj_class, state_dict):
    got_a_class = inspect.isclass(obj_class)
    if hasattr(obj_class, "__getnewargs_ex__"):
        if got_a_class:
            args, kwargs = obj_class.__getnewargs_ex__(obj_class)
        else:  # Self is first argument
            args, kwargs = obj_class.__getnewargs_ex__()
    elif hasattr(obj_class, "__getnewargs__"):
        if got_a_class:
            args = obj_class.__getnewargs__(obj_class)
        else:  # Self is first argument
            args = obj_class.__getnewargs__()
        kwargs = {}
    else:
        args = ()
        kwargs = {}
    if got_a_class:
        obj = obj_class.__new__(obj_class, *args, **kwargs)
    else:
        # We got an object which is not a class - it could be a singleton-like object -
        # we just return the object without initialisation.
        obj = obj_class
    if hasattr(obj, "__setstate__"):
        obj.__setstate__(state_dict)
    elif hasattr(obj, "__dict__"):
        obj.__dict__ = state_dict
    elif hasattr(obj, "__slots__"):
        for k, v in state_dict.items():
            setattr(obj, k, v)
    elif len(state_dict) != 0:
        raise TypeError(
            "Unexpected state signature, h5io is unable to restore the object."
        )
    return obj


def _guard_string_reductions(reduced, value, class_type, state_dict):
    # use_state adheres to pickling behaviour throughout, and pickle throws a
    # `PicklingError` when `__reduce__` returns a string but the reduced object is not
    # the object being pickled, so we do the same
    if isinstance(reduced, str):
        reduced_obj = _setstate(
            _import_class(class_type=class_type), state_dict=state_dict
        )
        if reduced_obj is not value:
            raise ValueError(
                f"Can't write {value}: it's not the same object as {reduced}"
            )
