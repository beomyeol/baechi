py_library(
    name = "logger",
    srcs = ["logger.py"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "communication_benchmark",
    srcs = ["communication_benchmark.py"],
    deps = [":logger"],
)
