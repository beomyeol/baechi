py_library(
    name = "train_lib",
    srcs = ["train.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//image_classifier/networks:nets_factory",
        "//nmt:model_factory",
        "//placer:cost",
        "//placer:placer_lib",
        "//third_party/grappler:graph_placer",
        "//utils:logger",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":train_lib",
    ],
)

