load("//third_party/repos:eigen.bzl", "eigen_repository")
load("//third_party/repos:numpy.bzl", "numpy_repository")
load("//third_party/repos:protobuf.bzl", "protobuf_rules_repository")
load("//third_party/repos:pybind11.bzl", "pybind11_repository")
load("//third_party/repos:python.bzl", "python_repository")
load("//third_party/repos:tenncor.bzl", "tenncor_repository")

def cortenn_repositories(repository_name, excludes = []):
    ignores = native.existing_rules().keys() + excludes
    if "eigen" not in ignores:
        eigen_repository(name = "eigen", repository_name = repository_name)

    if "numpy" not in ignores:
        numpy_repository(name = "numpy")

    if "protobuf" not in ignores:
        protobuf_rules_repository(name = "protobuf")

    if "pybind11" not in ignores:
        pybind11_repository(name = "pybind11", repository_name = repository_name)

    if "python" not in ignores:
        python_repository(name = "python")

    if "tenncor" not in ignores:
        tenncor_repository(name = "tenncor")
