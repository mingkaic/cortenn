load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository(name):
    git_repository(
        name = name,
        remote = "https://github.com/mingkaic/tenncor",
        commit = "0eb973395f7b13203c4f329a4141c572f2481521",
    )
