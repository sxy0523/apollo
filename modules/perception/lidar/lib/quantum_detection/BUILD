load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "quantum_layer",
    srcs = ["quantum_layer.cc"],
    hdrs = ["quantum_layer.h"],
    deps = [
        "//modules/common/util:util",
        "@eigen",
    ],
)

cc_library(
    name = "quantum_point_pillars",
    srcs = ["quantum_point_pillars.cc"],
    hdrs = ["quantum_point_pillars.h"],
    deps = [
        ":quantum_layer",
        "//modules/perception/lidar/lib:interface",
        "//modules/common/util:util",
        "@eigen",
    ],
)
