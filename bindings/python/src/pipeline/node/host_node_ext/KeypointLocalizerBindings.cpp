//
// Created by thwdpc on 8/22/25.
//

#include <pipeline/node/NodeBindings.hpp>

#include "postprocessors/KeypointLocalizer.hpp"
#include "pybind11_common.hpp"

void bind_keypointlocalizer(pybind11::module& m, void* pCallstack) {
    using namespace dai;
    using namespace dai::node;
    auto keypointLocalizer =
        py::class_<KeypointLocalizer, HostNode, ThreadedHostNode, std::shared_ptr<KeypointLocalizer>>(m, "KeypointLocalizer", DOC(dai, node, KeypointLocalizer));

    auto kp3D3C = py::class_<Keypoints3D3C>(m, "KP3Dim3Conf");
    auto kp3D1C = py::class_<Keypoints3D1C>(m, "KP3Dim1Conf");

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    // Call the rest of the type defines, then perform the actual bindings
    Callstack* callstack = static_cast<Callstack*>(pCallstack);
    const auto cb = callstack->top();
    callstack->pop();
    cb(m, pCallstack);
    // Actual bindings
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    keypointLocalizer
        .def(py::init<>([]() {
            auto node = std::make_shared<KeypointLocalizer>();
            getImplicitPipeline()->add(node);
            return node;
        }))
        .def(
            "build",
            [=](KeypointLocalizer& self, Node::Output& incomingKeypoints, const std::shared_ptr<ObjectTracker>& t, py::object kp_type) {
                if(kp_type.is(py::type::of<Keypoints3D3C>())) {
                    return self.build<singlekp::ValuesPerKeypoint::Three, true>(incomingKeypoints, t);
                }
                // if(kp_type.is(py::type::of<Keypoint3D1C>())) {
                //     return self.build<Keypoint3D1C>(o, t);
                // }
                throw std::runtime_error("Unsupported keypoint type");
            },
            py::arg("kp_input"),
            py::arg("tracker"),
            py::arg("kp_type") = kp3D3C)
        .def(
            "build",
            [=](KeypointLocalizer& self, Node::Output& incomingKeypoints, const std::shared_ptr<SpatialDetectionNetwork>& d, py::object kp_type) {
                if(kp_type.is(py::type::of<Keypoints3D3C>())) {
                    return self.build<singlekp::ValuesPerKeypoint::Three, true>(incomingKeypoints, d);
                }
                // if(kp_type.is(py::type::of<Keypoint3D1C>())) {
                //     return self.build<Keypoint3D1C>(o, d);
                // }
                throw std::runtime_error("Unsupported keypoint type");
            },
            py::arg("kp_input"),
            py::arg("detector"),
            py::arg("kp_type") = kp3D3C)
        .def(
            "build",
            [=](KeypointLocalizer& self, Node::Output& incomingKeypoints, Node::Output& splitSingularTracklets, py::object kp_type) {
                if(kp_type.is(py::type::of<Keypoints3D3C>())) {
                    return self.build<singlekp::ValuesPerKeypoint::Three, true>(incomingKeypoints, splitSingularTracklets);
                }
                // if(kp_type.is(py::type::of<Keypoint3D1C>())) {
                //     return self.build<Keypoint3D1C>(o, d);
                // }
                throw std::runtime_error("Unsupported keypoint type");
            },
            py::arg("kp_input"),
            py::arg("detector"),
            py::arg("kp_type") = kp3D3C)
        .def_property_readonly("out", [](KeypointLocalizer& node) { return std::ref(node.out); }, py::return_value_policy::reference_internal);
}
