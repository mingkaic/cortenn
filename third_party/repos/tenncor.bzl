load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository(name):
    git_repository(
        name = name,
        remote = "https://github.com/mingkaic/tenncor",
        commit = "3107f761a8d5c5dd86e2c038a6e5d9f7dfb9e64f",
    )
