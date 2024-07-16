"""Python Objects Onto HDF5."""

from importlib.metadata import version, PackageNotFoundError

from ._h5io import (
    read_hdf5,
    write_hdf5,
    object_diff,
    list_file_contents,
)  # noqa, analysis:ignore

try:
    __version__ = version("h5io")
except PackageNotFoundError:
    # package is not installed
    pass
del version, PackageNotFoundError
