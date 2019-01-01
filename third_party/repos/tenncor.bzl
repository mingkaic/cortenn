load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tenncor_repository(name):
    git_repository(
        name = name,
        remote = "https://github.com/mingkaic/tenncor",
        commit = "51506e4339263aceaf8697e480296139f4918ee5",
    )
