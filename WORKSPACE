workspace(name = "com_github_mingkaic_cortenn")

# local dependencies

load("//:third_party/all.bzl", "dependencies")
dependencies()

load("@com_github_mingkaic_tenncor//:tenncor.bzl", "dependencies")
dependencies()

load("@protobuf_rules//cpp:deps.bzl", "cpp_proto_library")
cpp_proto_library()

# test dependencies

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

git_repository(
    name = "com_github_google_benchmark",
    remote = "https://github.com/google/benchmark",
    commit = "e776aa0275e293707b6a0901e0e8d8a8a3679508",
)
