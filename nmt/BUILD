py_library(
    name = "nmt",
    srcs = ["nmt.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_helper",
    ],
)

py_library(
    name = "model_helper",
    srcs = ["model_helper.py"],
    deps = [
        "//utils:logger",
    ],
)

py_library(
    name = "model_factory",
    srcs = ["model_factory.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_nmt",
        ":gnmt",
        ":nmt",
    ],
)

py_library(
    name = "attention_nmt",
    srcs = ["attention_nmt.py"],
    deps = [
        ":nmt",
        "//utils:logger",
    ],
)

py_library(
    name = "gnmt",
    srcs = ["gnmt.py"],
    deps = [
        ":attention_nmt",
        ":model_helper",
        ":nmt",
        "//utils:logger",
    ],
)
