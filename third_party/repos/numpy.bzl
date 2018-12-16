load("//third_party/drake_rules:execute.bzl", "which")

_BUILD_CONTENT = """licenses([
    "notice", # BSD-2-Clause AND BSD-3-Clause AND MIT AND Python-2.0
    "unencumbered", # Public-Domain
])

cc_library(
    name = "numpy",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""

def _impl(repository_ctx):
    python = which(repository_ctx, "python{}".format(
        repository_ctx.attr.python_version,
    ))

    if not python:
        fail("Could NOT find python")

    result = repository_ctx.execute([
        str(python),
        "-c",
        "; ".join([
            "from __future__ import print_function",
            "import numpy",
            "print(numpy.get_include())",
        ]),
    ])

    if result.return_code != 0:
        fail("Could NOT determine NumPy include", attr = result.stderr)

    source = repository_ctx.path(result.stdout.strip())
    destination = repository_ctx.path("include")
    repository_ctx.symlink(source, destination)

    repository_ctx.file(
        "BUILD.bazel",
        content = _BUILD_CONTENT,
        executable = False,
    )

numpy_repository = repository_rule(
    _impl,
    attrs = {"python_version": attr.string(default = "2")},
    local = True,
)
