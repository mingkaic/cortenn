load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def protobuf_rules_repository(name):
    git_repository(
        name = name,
        remote = "https://github.com/mingkaic/rules_protobuf",
        commit = "f5615fa9d544d0a69cd73d8716364d8bd310babe",
    )
