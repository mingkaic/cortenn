load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository(name):
    git_repository(
        name = name,
        remote = "https://github.com/mingkaic/tenncor",
        commit = "96858057a78340313873b48388a00348ffce6105",
    )
