# Dependencies and licenses

The Noscari/AXIOM source is MIT licensed; see LICENSE. That license does not
relicense dependencies, research papers, or external model outputs. No external
model weights, third-party source trees, dependency wheels or native libraries
are bundled in our project wheel/source archive.

The clean Windows/Python 3.14.6 environment contains 24 runtime/transitive/build
and test distributions. Exact versions, downloaded wheel SHA-256 values, upstream
URLs and 152 license/attribution file hashes are in
`experiments/release-a5/dependency-inventory.json`. Every distribution has license
files. `pip check` passed; none of the installed artifacts was marked yanked in
the installation report. This is an inventory of the tested distribution artifacts,
not a universal cross-platform lock or a vulnerability database audit.

| Distribution(s) | Declared license / reviewed file |
|---|---|
| annotated-types, build, filelock, iniconfig, pluggy, pydantic, pydantic_core, pyproject_hooks, pytest, setuptools, typing-inspection, wheel | MIT; setuptools also carries vendor notices |
| colorama, Jinja2, MarkupSafe, mpmath, networkx, fsspec, sympy | BSD family; exact installed license text governs, including bundled components |
| Pygments | BSD-2-Clause and shipped authors/attribution file |
| packaging | Apache-2.0 OR BSD-2-Clause |
| typing_extensions | PSF-2.0 |
| NumPy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0, plus the actual wheel's bundled notices |
| PyTorch | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |

Do not flatten NumPy or PyTorch's multi-license distribution into the project's
MIT label. If you redistribute dependencies, retain their full license and notice
files and review the actual target-platform artifacts. This alpha distributes
requirements and an inventory, not a copied dependency environment.

Dependency artifacts were resolved from `files.pythonhosted.org`. Version ranges
in pyproject.toml are for portability; requirements-executed.txt records the
reference environment. The public installation inventory omits the local wheel
file URL and machine-specific paths. The unredacted pip report stays in local
scratch, outside the release. pip documents the report as evidence of what was
installed, not as an installable lock file:
[pip installation report](https://pip.pypa.io/en/stable/reference/installation-report/).
