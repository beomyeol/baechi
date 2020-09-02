py_library(
    name = "placer_lib",
    srcs = ["placer_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":adjuster",
        ":grouper",
        ":m_etf",
        ":m_sct",
        ":m_topo",
        ":placer_utils",
        ":virtual_scheduler",
        "//utils:logger",
    ],
)

py_test(
    name = "placer_lib_test",
    srcs = ["placer_lib_test.py"],
    deps = [":placer_lib"],
)

py_library(
    name = "device",
    srcs = ["device.py"],
    deps = [
        "//utils:logger",
    ],
)

py_test(
    name = "device_test",
    srcs = ["device_test.py"],
    deps = [
        ":device",
    ],
)

py_library(
    name = "placer_utils",
    srcs = ["placer_utils.py"],
    deps = [
        "//utils:logger",
    ],
)

py_test(
    name = "placer_utils_test",
    srcs = ["placer_utils_test.py"],
    deps = [":placer_utils"],
)

py_library(
    name = "m_topo",
    srcs = ["m_topo.py"],
    deps = [
        ":device",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_library(
    name = "m_etf",
    srcs = ["m_etf.py"],
    deps = [
        ":device",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_library(
    name = "m_sct",
    srcs = ["m_sct.py"],
    deps = [
        ":device",
        ":m_etf",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_library(
    name = "virtual_scheduler",
    srcs = ["virtual_scheduler.py"],
    deps = ["//utils:logger"],
)

py_library(
    name = "grouper",
    srcs = ["grouper.py"],
    deps = [
        ":placer_utils",
        "//utils:logger",
    ],
)

py_binary(
    name = "grouper_test",
    srcs = ["grouper_test.py"],
    deps = [":grouper"],
)

py_library(
    name = "adjuster",
    srcs = ["adjuster.py"],
    deps = [
        ":placer_utils",
        "//utils:logger",
    ],
)

py_binary(
    name = "adjuster_test",
    srcs = ["adjuster_test.py"],
    deps = [":adjuster"],
)

py_library(
    name = "cost",
    srcs = ["cost.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:logger",
    ],
)
