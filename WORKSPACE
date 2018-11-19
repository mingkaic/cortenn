workspace(name = "com_github_mingkaic_cortenn")

# test dependencies

git_repository(
    name = "com_github_mingkaic_testify",
    remote = "https://github.com/raggledodo/testify",
    commit = "e96e793b7082c3eb95f6177d5e7b0612ef6cd29c",
)

load("@com_github_mingkaic_testify//:testify.bzl", "dependencies")

dependencies()

load("@com_github_raggledodo_dora//:dora.bzl", "dependencies")

dependencies()

# pbm dependencies

load("//:cortenn.bzl", "dependencies")

dependencies()

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_repositories")

cpp_proto_repositories()

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_repositories")

py_proto_repositories()
