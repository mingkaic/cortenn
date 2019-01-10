load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository():
    git_repository(
        name = "com_github_mingkaic_tenncor",
        remote = "https://github.com/mingkaic/tenncor",
        commit = "8d63a50ceab47e8ce893f31aa64bbfea293d77de",
    )
