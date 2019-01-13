load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository():
    git_repository(
        name = "com_github_mingkaic_tenncor",
        remote = "https://github.com/mingkaic/tenncor",
        commit = "4e6a45996f14cd281ee7435dc58fca61c27e806c",
    )
