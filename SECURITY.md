# Security Policy

## Supported Versions

Only the most recent release of h5io is officially supported. The unreleased
development version on the `main` branch is also supported.

| Version | Supported                |
| ------- | ------------------------ |
| main    | :heavy_check_mark: (dev) |
| 0.2.6   | :heavy_check_mark:       |
| < 0.2.6 | :x:                      |

## Reading untrusted files

`h5io.read_hdf5` is intended to be safe to use on untrusted files with its
default arguments. Passing `allow_pickle=True` lets the file name classes to
import and objects to reconstruct (and lets pandas unpickle data), which can
execute arbitrary code, similar to `pickle.load` or
`numpy.load(..., allow_pickle=True)`. Only use `allow_pickle=True` for files
from trusted sources. Code execution that requires `allow_pickle=True` is not
considered a vulnerability in h5io.

## Reporting a Vulnerability

If you think you have found a security vulnerability in h5io,
**please do not report it as a GitHub issue**, in order to keep the
vulnerability confidential. Instead, please report it privately using
[GitHub's private vulnerability reporting](https://github.com/h5io/h5io/security/advisories/new)
and include a description and proof-of-concept that is
[short and self-contained](http://www.sscce.org/).

Generally you will receive a response within one week. h5io does not award
bounties for security vulnerabilities.
