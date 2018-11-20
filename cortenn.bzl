load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "com_github_mingkaic_tenncor" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_tenncor",
            remote = "https://github.com/mingkaic/tenncor",
            commit = "09664002ca45f5476b91aa4df86ff349ebed801a",
        )

    if "org_pubref_rules_protobuf" not in native.existing_rules():
        git_repository(
            name = "org_pubref_rules_protobuf",
            remote = "https://github.com/mingkaic/rules_protobuf",
            commit = "f5615fa9d544d0a69cd73d8716364d8bd310babe",
        )
