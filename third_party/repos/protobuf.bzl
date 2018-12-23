load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def protobuf_rules_repository(name):
    http_archive(
        name = name,
        urls = ["https://github.com/stackb/rules_proto/archive/844077a71597f91c41b02d4509c5f79d51588552.tar.gz"],
        sha256 = "867d09bf45515cb3ddeb06f7bdd2182eecf171ae3cd6b716b3b9d2fce50f292f",
        strip_prefix = "rules_proto-844077a71597f91c41b02d4509c5f79d51588552",
    )
