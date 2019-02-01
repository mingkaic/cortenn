load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository():
    git_repository(
        name = "com_github_mingkaic_tenncor",
        remote = "https://github.com/mingkaic/tenncor",
        commit = "7e5cdca44256527075ff4cf294d90efd566a5c4f",
    )
