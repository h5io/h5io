# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

h5io (`h5io` on PyPI) saves standard Python objects (dicts, lists, tuples, NumPy arrays,
scalars, strings, dates, SciPy sparse matrices, pandas objects, and optionally arbitrary
objects via their pickle protocol) to HDF5 files using h5py, and reads them back. It is used
by downstream projects such as MNE-Python to write files that people expect to read for years.
It is a small project maintained by volunteers: one library module (`h5io/_h5io.py`) and one
test module (`h5io/tests/test_io.py`).

## Working with humans (read first)

Every change here is reviewed by a volunteer maintainer and then maintained by humans for years.
The human you are working with is the author of anything that leaves their machine, and they are
the one who talks to the maintainers.

- **The human is the sole author.** Do not add `Co-Authored-By:` trailers for Claude / Copilot /
  Cursor / any other tool, and do not make a bot the commit author. The human opening the PR
  takes full responsibility for it and must be able to explain and defend every line on request.
  Disclose which tools were used, and for what, in the PR description instead.
- **Do not speak for the human in public.** Do not post agent-written text to issues, PRs, review
  comments, or discussions, and do not open, comment on, or close them on the human's behalf.
  Draft for the human if they ask; they decide whether to post and rewrite it in their own words.
- **Do not commit or push unless asked**, and never force-push, rewrite published history, or
  create tags or releases.
- **Agree on the design before writing code.** For anything beyond a local fix, describe the
  approach to the human in a few sentences and confirm it is the smallest one that works. If the
  change touches the on-disk format or the `allow_pickle` security boundary (see Architecture),
  say so explicitly; those are the changes a maintainer will look at hardest.
- **Report what actually happened.** If tests failed, were skipped, or could not be run on the
  relevant Python version (for example, the `use_state` paths need Python >= 3.11), say so
  plainly rather than implying it works.

## Keep code and tests brief

A big diff is a cost, not an accomplishment. The whole library is about 1100 lines of Python; a
change that adds a large fraction of that needs a very good reason.

- **Stop and re-plan at roughly 50 new lines of implementation.** Growth past that, or reaching
  for a new module, a new dependency, or a new on-disk type, is a signal that the design is too
  elaborate. Stop, tell the human, and offer the smaller version.
- **YAGNI.** No configuration knobs, fallbacks, extension points, or "robustness" that nothing in
  the change needs. New public API (a function or a keyword argument to `write_hdf5` /
  `read_hdf5`) needs a demonstrated user need, because it is documented and supported forever.
- **DRY.** Reuse what exists: `_triage_write` / `_triage_read` recursion,
  `_create_titled_group`, `_create_titled_dataset`, `_check_allow_pickle`, `_import_class`,
  `_setstate`. Add a branch to the existing triage functions rather than a parallel code path.
- **Keep tests just as brief.** Add tests to `h5io/tests/test_io.py`, preferring a few compact
  round-trip assertions (write, read back, compare, often with `object_diff`) and
  `pytest.mark.parametrize` over new helpers or fixtures. Classes used with `use_state=True` must
  be defined at module level so they can be imported on read. Throwaway verification scripts go
  in a scratch directory, not the repo.
- **Do not add or bump dependencies** (runtime or test) without asking first. The only required
  runtime dependencies are NumPy and h5py; SciPy, pandas, and PyTables are optional.

## Common commands

Development install and tests (pytest options, including coverage, live in `pyproject.toml`):
```bash
python -m pip install -e .
python -m pip install pytest pytest-cov scipy pandas tables
pytest h5io
```

Lint and format (ruff, via pre-commit; this is also run by pre-commit.ci on PRs):
```bash
pre-commit run --all-files
```

Before handing work back, at minimum run `pytest h5io` and `pre-commit run --all-files`, and
say which Python version(s) you tested on. Version-dependent code (`use_state`, anything
touching `__reduce__` / `__getstate__`, or standard-library classes like `pathlib.Path` whose
internals change between Python releases) should be checked on more than one Python version
when possible.

## Boundaries

Ask the human first before:
- adding or bumping a dependency, or changing the minimum Python version (`requires-python`);
- editing CI / release configuration (`.github/`, `codecov.yml`, `.pre-commit-config.yaml`,
  `[build-system]` in `pyproject.toml`);
- adding, renaming, or removing public API, including keyword arguments;
- adding a new on-disk type tag, or changing how an existing type is written (see File format).

Never:
- add AI co-authorship trailers or post agent-written text to issues or PRs (see above);
- import modules, resolve class names, or construct objects named in a file without going
  through `_check_allow_pickle` first, or use `eval` / `exec` / `pickle.loads` on file contents;
- change the meaning of an existing type tag or break reading of files written by earlier
  releases;
- silence a failing test, or mark it skip/xfail, instead of fixing the cause;
- commit build or test products (`*.egg-info`, `build/`, `dist/`, `.coverage`,
  `junit-results.xml`) or HDF5 files other than deliberate test fixtures;
- adapt code whose license is not compatible with this project's BSD-3-Clause license or
  cannot be determined.

## Architecture

### Files
- `h5io/__init__.py`: re-exports the public API: `write_hdf5`, `read_hdf5`, `object_diff`,
  `list_file_contents`. The version comes from `setuptools_scm` via package metadata.
- `h5io/_h5io.py`: everything else. `write_hdf5` validates arguments and opens the file, then
  `_triage_write` recursively dispatches on the Python type of each value. `read_hdf5` opens the
  file and `_triage_read` recursively dispatches on the stored type tag. pandas objects are
  written in a second pass through `pandas.HDFStore` (`cleanup_data`).
- `h5io/tests/test_io.py`: the whole test suite.

### File format
Every group or dataset that h5io writes carries a `TITLE` attribute naming its type (`dict`,
`list`, `tuple`, `ndarray`, `ndarray_str`, `int`, `float`, `unicode`, `ascii`, `json`,
`datetime`, `date`, `timezone`, `multiarray`, `csc_matrix`, `pd_dataframe`, `class`, ...).
`_triage_read` dispatches on that tag, and any other tag in a group is treated as the dotted
name of a class written with `use_state=True`. Files are expected to stay readable across h5io,
NumPy, h5py, and Python versions, so:
- reading files written by older releases must keep working; keep the old read path when the
  write path changes;
- a change that makes new files unreadable by older h5io releases is a compatibility break
  that the human must decide on and that belongs in the PR description;
- the dotted class name stored for `use_state` objects comes from `type(obj).__module__`,
  which can differ between Python versions (e.g., `pathlib._local.PosixPath` on 3.13).

### `use_state` and `allow_pickle`
With `use_state=True`, objects of unsupported types are stored following pickle's protocol:
`__reduce__()` decides how they are rebuilt, either from constructor arguments or from the state
returned by `__getstate__()`. Reconstruction (`_setstate`, `_import_class`) imports and calls
code named in the file, as do the `class` and pandas read paths, so all of them are gated behind
`read_hdf5(..., allow_pickle=True)` via `_check_allow_pickle`. With the default
`allow_pickle=False`, `read_hdf5` is intended to be safe on untrusted files (see `SECURITY.md`);
any change that lets file contents choose code to run without that check is a security bug.

## Changelog

There is no changelog file. Release notes are generated by GitHub from merged PR titles (see
`.github/release.yml`), so the PR title is the user-facing changelog entry. Use the existing
prefix convention (`FIX:`, `ENH:`, `MAINT:`, `DOC:`) followed by a short description of the
effect on users rather than the implementation. When a change adds public API, add
`.. versionadded::` / `.. versionchanged::` to the docstring.

## Code conventions

- Python: formatted and linted with ruff (`ruff format` and rules `E`, `F`, `W`, `D`, `I`; see
  `pyproject.toml`), with ruff's default 88-character lines and double quotes.
- Docstrings follow the NumPy convention (`Parameters`, `Returns`, `Notes`); every public and
  test function needs one, and test-only magic methods may use `# noqa: D105`.
- The library supports Python >= 3.8, so do not use newer syntax or standard-library features
  in `h5io/_h5io.py` (no `match`, no `X | Y` annotations, no `functools.cache`, ...) unless
  guarded. Features that need newer Python, like `use_state` (3.11), check `sys.version_info`
  and the corresponding tests use `pytest.mark.skipif`.
- Private names start with an underscore; optional dependencies (SciPy, pandas) are imported
  lazily inside functions so that `import h5io` needs only NumPy and h5py.
- Match the surrounding style rather than reformatting unrelated code; unrelated changes make
  review harder.
- Code adapted from an outside source must be under a BSD-3-Clause-compatible license (MIT,
  BSD, ISC, Apache-2.0, public domain, ...); attribute it in a comment directly above the
  adapted code, naming the source and its license.
