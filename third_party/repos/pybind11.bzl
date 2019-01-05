load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_BUILD_CONTENT = """load(
    "@com_github_mingkaic_cortenn//third_party/drake_rules:install.bzl",
    "install",
)

licenses(["notice"])  # BSD-3-Clause

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11",
    hdrs = glob([ "include/pybind11/**/*.h" ]),
    includes = ["include"],
    deps = [
        "@eigen",
        "@numpy",
        "@python",
    ],
)

install(
    name = "install",
    targets = [":pybind11"],
    hdr_dest = "include/pybind11",
    hdr_strip_prefix = ["include"],
    guess_hdrs = "PACKAGE",
    docs = ["LICENSE"],
)
"""

def pybind11_repository(name):
    new_git_repository(
        name = name,
        remote = "https://github.com/pybind/pybind11.git",
        tag = "v2.2.4",
        build_file_content = _BUILD_CONTENT,
    )
