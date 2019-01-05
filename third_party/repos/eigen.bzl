load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_BUILD_CONTENT = """load(
    "@com_github_mingkaic_cortenn//third_party/drake_rules:install.bzl",
    "install",
)

licenses([
    "notice",  # BSD-3-Clause
    "reciprocal",  # MPL-2.0
    "unencumbered",  # Public-Domain
])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "eigen",
    hdrs = glob(
        include = [
            "Eigen/**/*.h",
            "unsupported/Eigen/*",
        ],
        exclude = ["**/CMakeLists.txt"],
    ),
    defines = ["EIGEN_MPL2_ONLY"],
    includes = ["."],
)

install(
    name = "install",
    targets = [":eigen"],
    hdr_dest = "include/eigen3",
    guess_hdrs = "PACKAGE",
    docs = glob(["COPYING.*"]),
    doc_dest = "share/doc/eigen3",
)
"""

def eigen_repository(name):
    new_git_repository(
        name = name,
        remote = "https://github.com/eigenteam/eigen-git-mirror.git",
        tag = "3.3.5",
        build_file_content = _BUILD_CONTENT,
    )
