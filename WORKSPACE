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

# local dependencies

load("//:cortenn.bzl", "dependencies")

dependencies()

load("@com_github_mingkaic_tenncor//:tenncor.bzl", "dependencies")

dependencies()

load("@com_github_mingkaic_cppkg//:cppkg.bzl", "dependencies")

dependencies()

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_repositories")

cpp_proto_repositories()

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_repositories")

py_proto_repositories()

## Replace after removing rocnnet

load("//rocnnet:rocnnet.bzl", "dependencies")

dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()


new_local_repository(
    name = "python_linux",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "python27-lib",
    srcs = ["lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so"],
    hdrs = glob(["include/python2.7/*.h"]),
    includes = ["include/python2.7"],
    visibility = ["//visibility:public"]
)
""",
)
