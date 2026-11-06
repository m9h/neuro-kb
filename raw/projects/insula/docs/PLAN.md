# Insula: plan
Hosting: <https://github.com/m9h/insula> (decided 2026-10-21).

Goal: the BrainVISA core (soma, AIMS, Anatomist) as a codebase that builds
with stock CMake and Qt 6, packages cleanly as RPMs, and has no legacy layers
to carry. Every stage ends with the Fedora packages (`brainvisa-*`, later
`insula-*`) building from this repository with their `%check` green. Nothing is
pushed anywhere yet; hosting is undecided.

## 1. Provenance (stage 1, done)

Eight upstream repositories were cloned with their full history, checked out
at the commit the Fedora packages pin, the Fedora fix commits were fast-forwarded
on top, each was rewritten into a subdirectory with `git filter-repo`, and the
results were merged with unrelated histories (14,930 commits in total).

| Directory | Pinned commit | Fix commits on top |
|---|---|---|
| soma-base | 857c294 | 0 |
| soma-io | 7bf1670 | 4 (DCMTK 3.7 `decodedBitsAllocated`, MINC `<unistd.h>`, others) |
| aims-free | 9ea2121 | 5 (GCC 15/16, NumPy 2, Python 3.15, ObjectWrapper instantiations) |
| aims-gpl | 333d6d9 | 1 (SIP 6.17 import order) |
| anatomist-free | 07b99e3 | 2 (Qt 6 `add_definitions`, QtVersion) |
| anatomist-gpl | 6a0570d | 2 (SIP 6.17 import order, `execute_process`) |
| brainvisa-cmake | 850b6f3 | 6 (tomllib, FindSIP x2, FindLibKDTree, FindQtVersion, moc predefs) |
| brainvisa-share | e86e904 | 0 |

The fix commits are authored by Morgan Hough, and upstream's are untouched.

**Check against the packages.** The published `brainvisa-cmake`, `brainvisa-soma`
and `brainvisa-anatomist` source RPMs were unpacked, their `%prep` repeated
(the pinned tarballs plus the patch files each spec applies, then removing
the bundled `kdtree++`), and compared with the directories here: no file differs. The only
differences are the `ecat/` removal described below, the bundled `kdtree++`
(which the RPM deletes and Insula still carries until stage 5), and two symlinks
(`changelog.md`, `aimsdata/changelog.html`) that a tarball extract shows as
non-files.

**ECAT.** `aims-free/ecat/` is the University of Louvain ECAT library
(non-commercial, no modified redistribution). It is removed from the tree and
from every commit, so it is not in the history either; the rewrite changed the
`aims-free` commit hashes. BrainVISA's own CeCILL-B code around it (`ecat+/`,
`aimsdata/src/aimsecat/`) stays. A grep for the library's copyright text finds
nothing in any other directory. Building needs `-DBUILD_ECAT=OFF`, as the RPM
does today.

Not done in stage 1: a build of soma-io from this tree. The RPM builds
brainvisa-share, soma-base, soma-io, aims-free and aims-gpl as one combined
CMake project against the installed `brainvisa-cmake`, and a mock chroot with
its build requirements would not fit the 30-minute limit set for this stage.
The source comparison above stands in for it.

## 2. The staged plan

1. **Import** (done). Exit: tree equals what the RPMs build, plus licence
   hygiene.
2. **Qt 6 only, with AUTOMOC/AUTOUIC/AUTORCC.** Delete the Qt 3, 4 and 5 code
   paths (section 5). Replace `BRAINVISA_ADD_MOC_FILES` (13 call sites),
   `BRAINVISA_PYUIC` and the translation macros with Qt's own. This also
   retires the moc predefs workaround. Exit: Anatomist and AIMS GUI build and
   their `%check` passes.
3. **Stock CMake.** One top-level `CMakeLists.txt`, `CMakePresets.json`,
   `GNUInstallDirs`, `install(TARGETS ... EXPORT ...)`, `FILE_SET HEADERS`,
   namespaced exported targets with package config files, `ctest`, one place
   for the version (`project(VERSION)`), and a real `SOVERSION` policy (section
   4). Convert the 248 `.pro` source lists to CMake (section 3). Convert
   bottom-up: soma-base, soma-io, aims-free, aims-gpl, anatomist-free,
   anatomist-gpl, each ending green. Retire `bv_maker` and the `bv_*` scripts.
4. **Bindings.** Keep SIP 6; build with `sip-build` from `pyproject.toml`
   instead of the `bv_sip4make` wrapper. (132 `.sip` files.) Keep the import
   names `soma`, `aims`, `anatomist`.
5. **Unbundle and audit.** Use system kdtree++, nifticlib, gifticlib and
   CiftiLib; audit `brainvisa-share` file by file; settle the Fraqtive files
   (rewrite or relicense with the author's agreement); free ECAT reader.

RPMs become `insula-*` with `Obsoletes` and `Provides` for the `brainvisa-*`
names. The upstreamable subset (the 20 Fedora fix commits) stays cleanly
separable so it can still go to CEA.

## 3. Inventory of `brainvisa-cmake`

`cmake/brainvisa-cmake-config.cmake` is 3,118 lines; `cmake/` also holds
`packaging-deb.cmake`, `brainvisa-find-components.cmake`, 88 `Find*.cmake`
modules in `cmake/modules`, and toolchain files. 51 macros and functions are
defined. "Uses" counts lines mentioning the name in the seven other
directories, so it includes comments.

| Macro | Does | Uses | Stock CMake instead |
|---|---|---|---|
| BRAINVISA_DEPENDENCY | Records a run or dev dependency, with version ranges, for the BrainVISA packager | 128 | `find_package(... VERSION)`; the RPM spec |
| BRAINVISA_INSTALL | `install()` that insists on `COMPONENT` | 103 | `install()` |
| BRAINVISA_ADD_EXECUTABLE | `add_executable` plus install, help generation, scripts | 90 | `add_executable` + `install(TARGETS)` |
| BRAINVISA_COPY_AND_INSTALL_HEADERS | Symlinks or copies headers into `build/include/<dir>/` and installs them | 53 | `target_sources(FILE_SET HEADERS)` |
| BRAINVISA_GET_FILE_LIST_FROM_PRO | Reads source lists out of qmake `.pro` files | 41 | explicit source lists |
| BRAINVISA_COPY_DIRECTORY | Copies or symlinks a directory into the build tree | 19 | `file(COPY)` / install |
| BRAINVISA_FIND_PACKAGE | `find_package` that explains that a component must be configured first | 15 | `find_package` |
| BRAINVISA_ADD_MOC_FILES | Greps headers for `Q_OBJECT` and runs moc on each | 13 | `AUTOMOC` |
| BRAINVISA_GENERATE_DOXYGEN_DOC | Doxygen target | 12 | `doxygen_add_docs` |
| BRAINVISA_COPY_PYTHON_DIRECTORY | Copies or symlinks a Python tree into the build tree | 9 | install, `PYTHONPATH` in tests |
| BRAINVISA_GENERATE_SPHINX_DOC | Sphinx target | 8 | `add_custom_target` |
| BRAINVISA_PROJECT | Reads project info, calls `project()`, sets version variables | 7 | `project(VERSION)` |
| BRAINVISA_CREATE_CMAKE_CONFIG_FILES | Writes `<name>Config.cmake` and version files under `share/<proj>-X.Y/cmake` | 7 | `install(EXPORT)`, `configure_package_config_file` |
| BRAINVISA_COPY_FILES | Copies files into the build tree | 6 | `configure_file`, `file(COPY)` |
| BRAINVISA_ADD_SIP_PYTHON_MODULE | Builds a SIP module via `bv_sip4make` | 4 | `sip-build` |
| BRAINVISA_GENERATE_EPYDOC_DOC | Epydoc (Python 2-era) target | 3 | drop; Sphinx |
| BRAINVISA_VERSION_CONVERT | Version string conversion for packaging | 2 | drop |
| BRAINVISA_ADD_PYTRANSLATION | `.ts` to `.qm` for PyQt code | 2 | `qt_add_translations` |
| BRAINVISA_INSTALL_DIRECTORY | Recursive install following symlinks | 1 | `install(DIRECTORY)` |
| BRAINVISA_GENERATE_TARGET_NAME | Unique target names | 1 | none needed |
| BRAINVISA_ADD_TRANSLATION | `.ts` to `.qm` | 1 | `qt_add_translations` |

Defined but with no call outside `brainvisa-cmake` (30 macros; several are
called by the ones above, the rest are dead): `ADD_TEST`, `ADD_COMPONENT`,
`ADD_COMPONENT_GROUP`, `CREATE_MAIN_COMPONENTS`, `ADD_COMMAND_HELP`,
`GENERATE_COMMANDS_HELP`, `GENERATE_COMMANDS_HELP_INDEX`, `READ_PROJECT_INFO`,
`SET_PROJECT_VERSION`, `FIND_COMPONENT_DIRECTORIES`, `FIND_COMPONENT_SOURCES`,
`PYUIC`, `INSTALL_RUNTIME_LIBRARIES`, `RESOLVE_SYMBOL_LIBRARIES`,
`FIND_FSENTRY`, `REAL_PATHS`, `SYSTEM_PATH_TO_LIST`, `TARGET_SYSTEM_COMMAND`,
`TEMPORARY_FILE_NAME`, `GET_SPACED_QUOTED_LIST`, `GLOB_AND_FILTER`, and the
packaging set (`CREATE_RUN_PACKAGE`, `CREATE_DEV_PACKAGE`,
`CREATE_DEVDOC_PACKAGE`, `CREATE_DOC_PACKAGE`, `CREATE_SRC_PACKAGE`,
`PACKAGING_DEPENDENCY`, `PACKAGING_INITIALIZE_THIRDPARTY_COMPONENT`,
`DEBIAN_ARCHITECTURE`, `VALID_DEBIAN_PACKAGE_NAME`). They go in stage 3.

### Find modules

88 modules; the other seven directories refer to 18 of them by name (a
case-insensitive search for `find_package`, `BRAINVISA_FIND_PACKAGE` or
`BRAINVISA_DEPENDENCY` with the module name, so it may over-count):
Alsa, Blitz, DCMTK, Dot, Epydoc, Glib, LibKDTree, MINC, Numpy, OpenJpeg,
OpenSlide, PNG12, python, PythonInterp, QtVersion, Qwt, SIP, Sphinx.
The other 70 are bundling helpers (Cairo, Harfbuzz, GStreamer, XCB, PCRE, ICU,
OpenSSL, FFI and so on) that exist so a BrainVISA package can ship system
libraries, and similar. I have not checked each one. They go with `BRAINVISA_INSTALL_RUNTIME_LIBRARIES`.

| Module | Replace with |
|---|---|
| Alsa, DCMTK, PNG12, Numpy, PythonInterp, python | CMake's `FindALSA`, `FindDCMTK`, `FindPNG`, `Python::NumPy`, `find_package(Python)` |
| Glib, Blitz, OpenSlide, LibKDTree, MINC, OpenJpeg | `pkg_check_modules(... IMPORTED_TARGET)` or the package's own config |
| QtVersion | `find_package(Qt6)` |
| SIP | `sip-build`; no module |
| Qwt | the distribution's Qwt (Qt 6) config or `pkg_check_modules` |
| Dot, Sphinx, Epydoc | `find_program`; `Doxygen`'s `dot` component; drop Epydoc |

## 4. Conventions the macros encode (things to decide, not just delete)

- **Soname.** Libraries get `VERSION` set to the package version and no
  `SOVERSION` (for example `set_property(TARGET anatomist PROPERTY VERSION
  ${${PROJECT_NAME}_VERSION})`). CMake then uses the full version as the soname,
  so every release changes the soname. Insula needs an explicit `SOVERSION`
  policy (suggestion: major.minor of the ABI, bumped deliberately).
- **Build tree as a working install.** Headers are symlinked into
  `build/include/<dir>/` and Python trees into `build/<python dir>`, so code
  runs from the build tree without installing. Replace with target include
  directories and a test environment that sets `PYTHONPATH` and
  `LD_LIBRARY_PATH`; the RPM `%check` already runs against installed files.
- **Python install dir.** `PYTHON_INSTALL_DIRECTORY` places Python trees; the
  macros special-case directories named `python` and `brainvisa` and write
  `brainvisa/compilation_info.py` into the build tree.
- **Install layout.** Config files under `share/<project>-<major.minor>/cmake`,
  SIP files under `share/<project>-<major.minor>/sip`, docs under
  `share/doc`. Keeping a versioned share directory matters for compatibility
  with existing downstream code that finds files there.
- **Components.** Install components `runtime`, `dev`, `doc`, `usrdoc`, and
  per-component `-dev` variants. These map onto CPack or RPM subpackages.
- **Out-of-tree installs.** `make install-*` targets honour a
  `BRAINVISA_INSTALL_PREFIX` make variable instead of `DESTDIR`.
- **Guards for Singularity.** A `CMAKE_IGNORE_PATH` entry for
  `/.singularity.d/libs`; irrelevant outside the casa-distro containers.
- **Build output.** Executables and (on Windows) libraries go to `build/bin`,
  libraries to `build/lib`.
- **Source lists.** 248 qmake `.pro` files (with `#!include ../../config-app`)
  hold the source lists of the command-line tools and libraries, read by
  `BRAINVISA_GET_FILE_LIST_FROM_PRO` (41 uses). Converting them is mechanical
  and is the largest single piece of stage 3.
- **CMake hygiene.** Of the `cmake_minimum_required` calls: 2 say 2.6, 7 say
  3.10, 3 say 3.20. There are 32 `add_definitions` and 18 `file(GLOB)` calls
  outside `brainvisa-cmake`, and `exec_program` is still used in 2 places (a third use in
  `anatomist-gpl` was one of the Fedora fixes).

## 5. Measurements of the imported tree

- 2,950 C++ files, about 649,000 lines; 132 SIP files; 422 Python files;
  389 CMake files; 248 `.pro` files.
- 103 headers declare `Q_OBJECT`.
- `DESIRED_QT_VERSION` appears on 96 lines in 27 files; `QT_VERSION`
  preprocessor tests on 46 lines in 22 C/C++ files; Qt 3 names (`Q3*`,
  `QT3`, `Qt3`) in 7 files.
- Tests: 29 `add_test` or `enable_testing` lines in total; `BRAINVISA_ADD_TEST`
  is not used outside `brainvisa-cmake`. Upstream tests are thin; the Fedora
  `%check` stages (headless Anatomist under xvfb, aimsFileInfo reads) are
  most of what exists.

## 6. Risks

- **Size.** About 650,000 lines of C++ and a build system that every
  component depends on. Mitigation: bottom-up, green at each step.
- **Divergence from CEA.** Upstream keeps changing. The fork will drift unless
  stage 2 and 3 are kept in separable commits and upstream is asked early.
- **Licence mix.** CeCILL-B (permissive), CeCILL-2.0 and CeCILL-2.1 (copyleft,
  GPL-compatible), and a GPL-1.0-or-later widget (Fraqtive). The Fraqtive
  files are six; a rewrite is small. Distribution terms of binaries that link
  `aims-gpl` and `anatomist-gpl` are those of the copyleft licences.
- **Name.** "BrainVISA" and "Anatomist" are the CEA projects' names; the fork
  is named Insula and should not present itself as the original.
- **Non-free code.** ECAT is removed. Any future import must repeat the
  search for its copyright text. `brainvisa-share` (8,540 files) has not been
  audited file by file.
- **SIP.** SIP 6 and Python 3.14/3.15 keep breaking things (five of the 20 Fedora fix commits are SIP-related).
  Moving to `sip-build` removes the wrapper but not SIP itself.
- **Deleting Qt 3/4/5 code.** Mechanical, but some of it is entangled with
  the SIP files; every deletion needs the bindings to rebuild.
- **Tests.** There are few, so "green" at each stage means the RPM `%check`
  plus whatever is added. Adding tests before stage 3 would make the
  conversion safer.
- **Morphologist.** Its core (VIP) is proprietary and cannot be included;
  anything that needs it stays out.
