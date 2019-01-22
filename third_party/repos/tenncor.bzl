load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository():
    git_repository(
        name = "com_github_mingkaic_tenncor",
        remote = "https://github.com/mingkaic/tenncor",
        commit = "d7318b59e50b408ffcc73d8b2af4a6a6d9dfc48b",
    )
