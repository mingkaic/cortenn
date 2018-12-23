_REPO_NAME = "com_github_mingkaic_cortenn"
workspace(name = _REPO_NAME)

# local dependencies

load("//:third_party/all.bzl", "cortenn_repositories")
cortenn_repositories(_REPO_NAME)

load("@tenncor//:tenncor.bzl", "dependencies")
dependencies()

load("@protobuf_rules//cpp:deps.bzl", "cpp_proto_library")
cpp_proto_library()

# test dependencies

load("@cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")


git_repository(
    name = "io_bazel_rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    commit = "e6399b601e2f72f74e5aa635993d69166784dde1",
)

load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip_import")
pip_repositories()

pip_import(
	name = "protobuf_py_deps",
	requirements = "//llo:test/requirements.txt",
)

load("@protobuf_py_deps//:requirements.bzl", protobuf_pip_install = "pip_install")
protobuf_pip_install()
