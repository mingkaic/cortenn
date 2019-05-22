load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository():
    git_repository(
        name = "com_github_mingkaic_tenncor",
        remote = "https://github.com/mingkaic/tenncor",
        commit = "950e9987e4787bed7889ed527b9397bfb9252bdf",
    )
