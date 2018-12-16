load(
    ":pathutils.bzl",
    "dirname",
    "join_paths",
    "output_path",
)
load("@python//:version.bzl", "PYTHON_SITE_PACKAGES_RELPATH")

InstallInfo = provider()

InstalledTestInfo = provider()

#==============================================================================
#BEGIN internal helpers

#------------------------------------------------------------------------------
def _workspace(ctx):
    """Compute name of current workspace."""

    # Check for override
    if hasattr(ctx.attr, "workspace"):
        if len(ctx.attr.workspace):
            return ctx.attr.workspace

    # Check for meaningful workspace_root
    workspace = ctx.label.workspace_root.split("/")[-1]
    if len(workspace):
        return workspace

    # If workspace_root is empty, assume we are the root workspace
    return ctx.workspace_name

def _rename(file_dest, rename):
    """Compute file name if file renamed."""
    if file_dest in rename:
        renamed = rename[file_dest]
        return join_paths(dirname(file_dest), renamed)
    return file_dest

def _depset_to_list(l):
    """Helper function to convert depset to list."""
    iter_list = l.to_list() if type(l) == "depset" else l
    return iter_list

#------------------------------------------------------------------------------
def _output_path(ctx, input_file, strip_prefix = [], warn_foreign = True):
    """Compute output path (without destination prefix) for install action.

    This computes the adjusted output path for an input file. It is the same as
    :func:`output_path`, but additionally handles files outside the current
    package when :func:`install` or :func:`install_files` is invoked with
    non-empty ``allowed_externals``.
    """

    # Try the current package first
    path = output_path(ctx, input_file, strip_prefix)
    if path != None:
        return path

    # If we were unable to resolve a path, the file must be "foreign", so try
    # to resolve against the list of allowed externals.
    if path == None and hasattr(ctx.attr, "allowed_externals"):
        for x in ctx.attr.allowed_externals:
            package_root = join_paths(x.label.workspace_root, x.label.package)
            path = output_path(ctx, input_file, strip_prefix, package_root)
            if path != None:
                return path

    # If we get here, we were not able to resolve the path; give up, and print
    # a warning about installing the "foreign" file.
    if warn_foreign:
        print("%s installing file %s which is not in current package" %
              (ctx.label, input_file.path))
    return input_file.basename

#------------------------------------------------------------------------------
def _guess_files(target, candidates, scope, attr_name):
    if scope == "EVERYTHING":
        return candidates

    elif scope == "WORKSPACE":
        return [
            f
            for f in _depset_to_list(candidates)
            if target.label.workspace_root == f.owner.workspace_root
        ]

    elif scope == "PACKAGE":
        return [
            f
            for f in _depset_to_list(candidates)
            if (target.label.workspace_root == f.owner.workspace_root and
                target.label.package == f.owner.package)
        ]

    else:
        msg_fmt = "'install' given unknown '%s' value '%s'"
        fail(msg_fmt % (attr_name, scope), scope)

#------------------------------------------------------------------------------
def _install_action(
        ctx,
        artifact,
        dests,
        strip_prefixes = [],
        rename = {},
        warn_foreign = True):
    """Compute install action for a single file.

    This takes a single file artifact and returns the appropriate install
    action for the file. The parameters are the same as for
    :func:`_install_action`.
    """
    if type(dests) == "dict":
        dest = dests.get(artifact.extension, dests[None])
    else:
        dest = dests

    dest_replacements = (
        ("@WORKSPACE@", _workspace(ctx)),
        ("@PYTHON_SITE_PACKAGES@", PYTHON_SITE_PACKAGES_RELPATH),
    )
    for old, new in dest_replacements:
        if old in dest:
            dest = dest.replace(old, new)

    if type(strip_prefixes) == "dict":
        strip_prefix = strip_prefixes.get(
            artifact.extension,
            strip_prefixes[None],
        )
    else:
        strip_prefix = strip_prefixes

    file_dest = join_paths(
        dest,
        _output_path(ctx, artifact, strip_prefix, warn_foreign),
    )
    file_dest = _rename(file_dest, rename)

    if "/attic/" in file_dest:
        fail("Do not expose attic paths to the install tree ({})".format(
            file_dest,
        ))

    return struct(src = artifact, dst = file_dest)

#------------------------------------------------------------------------------
def _install_actions(
        ctx,
        file_labels,
        dests,
        strip_prefixes = [],
        excluded_files = [],
        rename = {},
        warn_foreign = True):
    """Compute install actions for files.

    This takes a list of labels (targets or files) and computes the install
    actions for the files owned by each label.

    Args:
        file_labels (:obj:`list` of :obj:`Label`): List of labels to install.
        dests (:obj:`str` or :obj:`dict` of :obj:`str` to :obj:`str`):
            Install destination. A :obj:`dict` may be given to supply a mapping
            of file extension to destination path. The :obj:`dict` must have an
            entry with the key ``None`` that is used as the default when there
            is no entry for the specific extension.
        strip_prefixes (:obj:`list` of :obj:`str` or :obj:`dict` of :obj:`list`
            of :obj:`str` to :obj:`str`): List of prefixes to strip from the
            input path before prepending the destination. A :obj:`dict` may be
            given to supply a mapping of file extension to list of prefixes to
            strip. The :obj:`dict` must have an entry with the key ``None``
            that is used as the default when there is no entry for the specific
            extension.
        excluded_files (:obj:`list` of :obj:`str`): List of files to exclude
            from installation.

    Returns:
        :obj:`list`: A list of install actions.
    """
    actions = []

    # Iterate over files. We expect a list of labels, which will have a 'files'
    # attribute that is a list of file artifacts. Thus this two-level loop.
    for f in file_labels:
        for a in _depset_to_list(f.files):
            # TODO(mwoehlke-kitware) refactor this to separate computing the
            # original relative path and the path with prefix(es) stripped,
            # then use the original relative path for both exclusions and
            # renaming.
            if _output_path(ctx, a, warn_foreign = False) in excluded_files:
                continue

            actions.append(
                _install_action(
                    ctx,
                    a,
                    dests,
                    strip_prefixes,
                    rename,
                    warn_foreign,
                ),
            )

    return actions

#------------------------------------------------------------------------------
# Compute install actions for a cc_library or cc_binary.
def _install_cc_actions(ctx, target):
    # Compute actions for target artifacts.
    dests = {
        "a": ctx.attr.archive_dest,
        "so": ctx.attr.library_dest,
        None: ctx.attr.runtime_dest,
    }
    strip_prefixes = {
        "a": ctx.attr.archive_strip_prefix,
        "so": ctx.attr.library_strip_prefix,
        None: ctx.attr.runtime_strip_prefix,
    }
    actions = _install_actions(
        ctx,
        [target],
        dests,
        strip_prefixes,
        rename = ctx.attr.rename,
    )

    # Compute actions for guessed resource files.
    if ctx.attr.guess_data != "NONE":
        data = [
            f
            for f in _depset_to_list(target.data_runfiles.files)
            if f.is_source
        ]
        data = _guess_files(target, data, ctx.attr.guess_data, "guess_data")
        actions += _install_actions(
            ctx,
            [struct(files = data)],
            ctx.attr.data_dest,
            ctx.attr.data_strip_prefix,
            ctx.attr.guess_data_exclude,
            rename = ctx.attr.rename,
        )

    # Compute actions for guessed headers.
    if ctx.attr.guess_hdrs != "NONE":
        hdrs = _guess_files(
            target,
            target.cc.transitive_headers,
            ctx.attr.guess_hdrs,
            "guess_hdrs",
        )
        actions += _install_actions(
            ctx,
            [struct(files = hdrs)],
            ctx.attr.hdr_dest,
            ctx.attr.hdr_strip_prefix,
            ctx.attr.guess_hdrs_exclude,
            rename = ctx.attr.rename,
        )

    # Return computed actions.
    return actions

#------------------------------------------------------------------------------
# Compute install actions for a java_library or java_binary.
def _install_java_actions(ctx, target):
    dests = {
        "jar": ctx.attr.java_dest,
        None: ctx.attr.runtime_dest,
    }
    strip_prefixes = {
        "jar": ctx.attr.java_strip_prefix,
        None: ctx.attr.runtime_strip_prefix,
    }
    excluded_files = []
    if target.files_to_run.executable:
        excluded_files = [
            _output_path(
                ctx,
                target.files_to_run.executable,
                warn_foreign = False,
            ),
        ]
    return _install_actions(
        ctx,
        [target],
        dests,
        strip_prefixes,
        excluded_files,
        rename = ctx.attr.rename,
    )

#------------------------------------------------------------------------------
# Compute install actions for a py_library or py_binary.
# TODO(jamiesnape): Install native shared libraries that the target may use.
def _install_py_actions(ctx, target):
    return _install_actions(
        ctx,
        [target],
        ctx.attr.py_dest,
        ctx.attr.py_strip_prefix,
        rename = ctx.attr.rename,
    )

#------------------------------------------------------------------------------
# Compute install actions for a script or an executable.
def _install_runtime_actions(ctx, target):
    return _install_actions(
        ctx,
        [target],
        ctx.attr.runtime_dest,
        ctx.attr.runtime_strip_prefix,
        rename = ctx.attr.rename,
    )

#------------------------------------------------------------------------------
def _install_test_actions(ctx):
    """Compute and return list of install test command lines.

    This computes the install path for the install tests (tests run to verify
    that the project works once installed).

    Returns:
        :obj:`struct`: A list of test actions containing the location of the
        tests files in the source tree and in the install tree.
    """
    test_actions = []

    # For files, we run the file from the build tree.
    for test in ctx.attr.install_tests:
        for f in _depset_to_list(test.files):
            test_actions.append(
                struct(src = f, cmd = f.path),
            )

    return test_actions

#------------------------------------------------------------------------------
# Generate install code for an install action.
def _install_code(action):
    return "install(%r, %r)" % (action.src.short_path, action.dst)

#------------------------------------------------------------------------------
# Generate install code for a java launcher.
def _java_launcher_code(action):
    return "create_java_launcher(%r, %r, %r, %r)" % (
        action.dst,
        action.classpath,
        " ".join(action.jvm_flags),
        action.main_class,
    )

#END internal helpers
#==============================================================================
#BEGIN rules

#------------------------------------------------------------------------------
# Generate information to install "stuff". "Stuff" can be library or binary
# targets, headers, or documentation files.
def _install_impl(ctx):
    actions = []
    installed_tests = []
    rename = dict(ctx.attr.rename)

    # Collect install actions from dependencies.
    for d in ctx.attr.deps:
        actions += d[InstallInfo].install_actions
        rename.update(d[InstallInfo].rename)
        if InstalledTestInfo in d:
            installed_tests += d[InstalledTestInfo].tests

    # Generate actions for data, docs and includes.
    actions += _install_actions(
        ctx,
        ctx.attr.docs,
        ctx.attr.doc_dest,
        strip_prefixes = ctx.attr.doc_strip_prefix,
        rename = rename,
    )
    actions += _install_actions(
        ctx,
        ctx.attr.data,
        ctx.attr.data_dest,
        strip_prefixes = ctx.attr.data_strip_prefix,
        rename = rename,
    )
    actions += _install_actions(
        ctx,
        ctx.attr.hdrs,
        ctx.attr.hdr_dest,
        strip_prefixes = ctx.attr.hdr_strip_prefix,
        rename = rename,
    )

    for t in ctx.attr.targets:
        # TODO(jwnimmer-tri): Raise an error if a target has testonly=1.
        if hasattr(t, "cc"):
            actions += _install_cc_actions(ctx, t)
        elif hasattr(t, "java"):
            actions += _install_java_actions(ctx, t)
        elif hasattr(t, "py"):
            actions += _install_py_actions(ctx, t)
        elif hasattr(t, "files_to_run") and t.files_to_run.executable:
            # Executable scripts copied from source directory.
            actions += _install_runtime_actions(ctx, t)

    # Generate install test actions.
    installed_tests += _install_test_actions(ctx)

    # Generate code for install actions.
    script_actions = []
    installed_files = {}
    for a in actions:
        src = None
        if hasattr(a, "src"):
            src = a.src
        if a.dst not in installed_files:
            if src:
                script_actions.append(_install_code(a))
            else:
                script_actions.append(_java_launcher_code(a))
            installed_files[a.dst] = src
        elif src != installed_files[a.dst]:
            fail("Install conflict detected:\n" +
                 "\n  src1 = " + repr(installed_files[a.dst]) +
                 "\n  src2 = " + repr(src) +
                 "\n  dst = " + repr(a.dst))

    ctx.actions.expand_template(
        template = ctx.executable.install_script_template,
        output = ctx.outputs.executable,
        substitutions = {"<<actions>>": "\n    ".join(script_actions)},
    )

    script_tests = []

    # Generate list containing all commands to run to test.
    for i in installed_tests:
        script_tests.append(i.cmd)

    # Generate test installation script
    if ctx.attr.install_tests_script and not script_tests:
        fail("`install_tests_script` is not empty but no `script_tests` were provided.")  # noqa
    if ctx.attr.install_tests_script:
        ctx.actions.write(
            output = ctx.outputs.install_tests_script,
            content = "\n".join(script_tests),
            is_executable = False,
        )

    # Return actions.
    files = ctx.runfiles(
        files = [a.src for a in actions if not hasattr(a, "main_class")] +
                [i.src for i in installed_tests],
    )
    return [
        InstallInfo(install_actions = actions, rename = rename),
        InstalledTestInfo(tests = installed_tests),
        DefaultInfo(runfiles = files),
    ]

_install_rule = rule(
    # Update buildifier-tables.json when this changes.
    attrs = {
        "deps": attr.label_list(providers = [InstallInfo]),
        "docs": attr.label_list(allow_files = True),
        "doc_dest": attr.string(default = "share/doc/@WORKSPACE@"),
        "doc_strip_prefix": attr.string_list(),
        "data": attr.label_list(allow_files = True),
        "data_dest": attr.string(default = "share/@WORKSPACE@"),
        "data_strip_prefix": attr.string_list(),
        "guess_data": attr.string(default = "NONE"),
        "guess_data_exclude": attr.string_list(),
        "hdrs": attr.label_list(allow_files = True),
        "hdr_dest": attr.string(default = "include"),
        "hdr_strip_prefix": attr.string_list(),
        "guess_hdrs": attr.string(default = "NONE"),
        "guess_hdrs_exclude": attr.string_list(),
        "targets": attr.label_list(),
        "archive_dest": attr.string(default = "lib"),
        "archive_strip_prefix": attr.string_list(),
        "library_dest": attr.string(default = "lib"),
        "library_strip_prefix": attr.string_list(),
        "runtime_dest": attr.string(default = "bin"),
        "runtime_strip_prefix": attr.string_list(),
        "java_dest": attr.string(default = "share/java"),
        "java_strip_prefix": attr.string_list(),
        "py_dest": attr.string(default = "@PYTHON_SITE_PACKAGES@"),
        "py_strip_prefix": attr.string_list(),
        "rename": attr.string_dict(),
        "install_tests": attr.label_list(
            default = [],
            allow_files = True,
        ),
        "workspace": attr.string(),
        "allowed_externals": attr.label_list(allow_files = True),
        "install_script_template": attr.label(
            allow_files = True,
            executable = True,
            cfg = "target",
            default = Label(
                "@com_github_mingkaic_cortenn//third_party/drake_rules:install.py.in"),
        ),
        "install_tests_script": attr.output(),
    },
    executable = True,
    implementation = _install_impl,
)

def install(tags = [], **kwargs):
    # (The documentation for this function is immediately below.)
    _install_rule(
        tags = tags + ["install"],
        **kwargs
    )

"""Generate installation information for various artifacts.

This generates installation information for various artifacts, including
documentation and header files, and targets (e.g. ``cc_binary``). By default,
the path of any files is included in the install destination.
See :rule:`install_files` for details.

Normally, you should not install files or targets from a workspace other than
the one invoking ``install``, and ``install`` will warn if asked to do so. In
cases (e.g. adding install rules to a project that is natively built with
bazel, but does not define an install) where this *is* the right thing to do,
the ``allowed_externals`` argument may be used to specify a list of externals
whose files it is okay to install, which will suppress the warning.

Destination paths may include the following placeholders:

* ``@WORKSPACE@``, replaced with ``workspace`` (if specified) or the name of
  the workspace which invokes ``install``.
* ``@PYTHON_SITE_PACKAGES``, replaced with the Python version-specific path of
  "site-packages".

Note:
    By default, headers and resource files to be installed must be explicitly
    listed. This is to work around an issue where Bazel does not appear to
    provide any mechanism to obtain the public headers of a target, nor the
    *direct* data files of a target, at rule instantiation. The ``guess_hdrs``
    and ``guess_data`` parameters may be used to tell ``install`` to guess at
    what headers and/or resource files will be installed. Possible values are:

    * ``"NONE"``: Only install files which are explicitly listed (i.e. by
      ``hdrs`` or ``data``).
    * ``PACKAGE``:  For each target, install those files which are used by the
      target and owned by a target in the same package.
    * ``WORKSPACE``: For each target, install those files which are used by the
      target and owned by a target in the same workspace.
    * ``EVERYTHING``: Install all headers/resources used by the target.

    The headers and resource files considered are *all* headers or resource
    files transitively used by the target. Any option other than ``NONE`` is
    also likely to install private headers, and may install resource files used
    by other targets. In either case, this may result in the same file being
    considered for installation more than once.

    Note also that, because Bazel includes *all* run-time dependencies —
    including e.g. shared libraries — in a target's ``runfiles``, only *source*
    artifacts are considered when guessing resource files.

    A file containing all the commands to test executables after installation
    is created if `install_tests_script` is set. The list of commands to run
    is given by `install_tests`. The generated file location can be passed to
    `install_test()` as an `args`.

Args:
    deps: List of other install rules that this rule should include.
    docs: List of documentation files to install.
    doc_dest: Destination for documentation files
        (default = "share/doc/@WORKSPACE@").
    doc_strip_prefix: List of prefixes to remove from documentation paths.
    guess_data: See note.
    guess_data_exclude: List of resources found by ``guess_data`` to exclude
        from installation.
    data: List of (platform-independent) resource files to install.
    data_dest: Destination for resource files (default = "share/@WORKSPACE@").
    data_strip_prefix: List of prefixes to remove from resource paths.
    guess_hdrs: See note.
    guess_hdrs_exclude: List of headers found by ``guess_hdrs`` to exclude from
        installation.
    hdrs: List of header files to install.
    hdr_dest: Destination for header files (default = "include").
    hdr_strip_prefix: List of prefixes to remove from header paths.
    targets: List of targets to install.
    archive_dest: Destination for static library targets (default = "lib").
    archive_strip_prefix: List of prefixes to remove from static library paths.
    library_dest: Destination for shared library targets (default = "lib").
    library_strip_prefix: List of prefixes to remove from shared library paths.
    runtime_dest: Destination for executable targets (default = "bin").
    runtime_strip_prefix: List of prefixes to remove from executable paths.
    java_dest: Destination for Java library targets (default = "share/java").
    java_strip_prefix: List of prefixes to remove from Java library paths.
    py_dest: Destination for Python targets
        (default = "lib/python{MAJOR}.{MINOR}/site-packages").
    py_strip_prefix: List of prefixes to remove from Python paths.
    rename: Mapping of install paths to alternate file names, used to rename
      files upon installation.
    install_tests: List of scripts that are designed to test the install
        tree. These scripts will not be installed.
    install_tests_script: Name of the generated file that contains the commands
        run to test the install tree. This only needs to be specified for the
        main `install()` call, and the same name should be passed to
        `install_test()` as `"$(location :" + install_tests_script + ")"`.
    workspace: Workspace name to use in default paths (overrides built-in
        guess).
    allowed_externals: List of external packages whose files may be installed.
"""
