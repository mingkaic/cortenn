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
