load(":install.bzl", "install")

def pybind_py_library(
        name,
        cc_srcs = [],
        cc_deps = [],
        cc_so_name = None,
        cc_binary_rule = native.cc_binary,
        py_srcs = [],
        py_deps = [],
        py_imports = [],
        py_library_rule = native.py_library,
        visibility = None,
        testonly = None):
    """Declares a pybind11 Python library with C++ and Python portions.

    @param cc_srcs
        C++ source files.
    @param cc_deps (optional)
        C++ dependencies.
        At present, these should be libraries that will not cause ODR
        conflicts (generally, header-only).
    @param cc_so_name (optional)
        Shared object name. By default, this is `${name}`, so that the C++
        code can be then imported in a more controlled fashion in Python.
        If overridden, this could be the public interface exposed to the user.
    @param py_srcs (optional)
        Python sources.
    @param py_deps (optional)
        Python dependencies.
    @param py_imports (optional)
        Additional Python import directories.
    @return struct(cc_so_target = ..., py_target = ...)
    """
    py_name = name
    if not cc_so_name:
        cc_so_name = name

    # TODO(eric.cousineau): See if we can keep non-`*.so` target name, but
    # output a *.so, so that the target name is similar to what is provided.
    cc_so_target = cc_so_name + ".so"

    # Add C++ shared library.
    cc_binary_rule(
        name = cc_so_target,
        srcs = cc_srcs,
        # This is how you tell Bazel to create a shared library.
        linkshared = 1,
        linkstatic = 1,
        copts = [
            # GCC and Clang don't always agree / succeed when inferring storage
            # duration (#9600). Workaround it for now.
            "-Wno-unused-lambda-capture",
        ],
        # Always link to pybind11.
        deps = [
            "@pybind11",
        ] + cc_deps,
        testonly = testonly,
        visibility = visibility,
    )

    # Add Python library.
    py_library_rule(
        name = py_name,
        data = [cc_so_target],
        srcs = py_srcs,
        deps = py_deps,
        imports = py_imports,
        testonly = testonly,
        visibility = visibility,
    )
    return struct(
        cc_so_target = cc_so_target,
        py_target = py_name,
    )

def pybind_library(
        name,
        cc_srcs = [],
        cc_deps = [],
        cc_so_name = None,
        package_info = None,
        py_srcs = [],
        py_deps = [],
        py_imports = [],
        add_install = True,
        visibility = None,
        testonly = None):
    """Declares a pybind11 library with C++ and Python portions.

    For parameters `cc_srcs`, `py_srcs`, `py_deps`, `py_imports`, please refer
    to `pybind_py_library`.

    @param cc_deps (optional)
        C++ dependencies.
        At present, these should be libraries that will not cause ODR
        conflicts (generally, header-only).
        By default, this includes `pydrake_pybind` and
        `//:drake_shared_library`.
    @param cc_so_name (optional)
        Shared object name. By default, this is `${name}` (without the `_py`
        suffix if it's present).
    @param package_info
        This should be the result of `get_pybind_package_info` called from the
        current package. This dictates how `PYTHONPATH` is configured, and
        where the modules will be installed.
    @param add_install (optional)
        Add install targets.
    """
    if package_info == None:
        # fail("`package_info` must be supplied.")
        imports = []
        dest = "@PYTHON_SITE_PACKAGES@/"
    else:
        imports = package_info.py_imports
        dest = package_info.py_dest

    if not cc_so_name:
        if name.endswith("_py"):
            cc_so_name = name[:-3]
        else:
            cc_so_name = name
    install_name = name + "_install"
    targets = pybind_py_library(
        name = name,
        cc_so_name = cc_so_name,
        cc_srcs = cc_srcs,
        cc_deps = cc_deps,
        cc_binary_rule = native.cc_binary,
        py_srcs = py_srcs,
        py_deps = py_deps,
        py_imports = imports + py_imports,
        py_library_rule = native.py_library,
        testonly = testonly,
        visibility = visibility,
    )

    # Add installation target for C++ and Python bits.
    if add_install:
        install(
            name = install_name,
            targets = [
                targets.cc_so_target,
                targets.py_target,
            ],
            py_dest = dest,
            library_dest = dest,
            visibility = visibility,
        )
