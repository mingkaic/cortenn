load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository(name):
    git_repository(
        name = name,
        remote = "https://github.com/mingkaic/tenncor",
        commit = "3733217e4f9ee10cdebb90280e5990c18446f6f6",
    )
