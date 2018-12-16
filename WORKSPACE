_REPO_NAME = "com_github_mingkaic_cortenn"
workspace(name = _REPO_NAME)

# local dependencies

load("//:third_party/all.bzl", "cortenn_repositories")
cortenn_repositories(_REPO_NAME)

load("@tenncor//:tenncor.bzl", "dependencies")
dependencies()

load("@cppkg//:cppkg.bzl", "dependencies")
dependencies()

load("@protobuf//cpp:rules.bzl", "cpp_proto_repositories")
cpp_proto_repositories()

load("@protobuf//python:rules.bzl", "py_proto_repositories")
py_proto_repositories()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

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
