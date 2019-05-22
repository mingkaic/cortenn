workspace(name = "com_github_mingkaic_cortenn")

# local dependencies

load("//:third_party/all.bzl", "dependencies")
dependencies()

load("@com_github_mingkaic_tenncor//:third_party/all.bzl", "dependencies")
dependencies()

# test dependencies

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

load(
    "@com_github_mingkaic_tenncor//third_party/repos:benchmark.bzl",
    "benchmark_repository"
)
benchmark_repository()

# external dependencies

load("@protobuf_rules//cpp:deps.bzl", "cpp_proto_library", "cpp_grpc_library")
cpp_proto_library()
cpp_grpc_library()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
