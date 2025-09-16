//
// Created by thwdpc on 7/31/25.
//

#include <pipeline/node/NodeBindings.hpp>

#include "depthai/pipeline/node/host/contrib/host_nodes_ext/ParsingNeuralNetwork.hpp"
#include "pybind11_common.hpp"

void bind_parsingneuralnetwork(pybind11::module& m, void* pCallstack) {
    using namespace dai;
    using namespace dai::node;
    auto parsingNeuralNetwork = py::class_<ParsingNeuralNetwork, ThreadedHostNode, std::shared_ptr<ParsingNeuralNetwork>>(
        m, "ParsingNeuralNetwork", DOC(dai, node, ParsingNeuralNetwork));

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

    parsingNeuralNetwork
        .def(py::init<>([]() {
            auto node = std::make_shared<ParsingNeuralNetwork>();
            getImplicitPipeline()->add(node);
            return node;
        }))
        .def("build", py::overload_cast<Node::Output&, const NNArchive&>(&ParsingNeuralNetwork::build), py::arg("input"), py::arg("nnArchive"))
        .def("build",
             py::overload_cast<const std::shared_ptr<Camera>&, NNModelDescription, std::optional<float>>(&ParsingNeuralNetwork::build),
             py::arg("input"),
             py::arg("modelDesc"),
             py::arg("fps") = std::nullopt)
        .def("build",
             py::overload_cast<const std::shared_ptr<Camera>&, const NNArchive&, std::optional<float>>(&ParsingNeuralNetwork::build),
             py::arg("input"),
             py::arg("nnArchive"),
             py::arg("fps") = std::nullopt)
        .def_property_readonly(
            "out",
            [](ParsingNeuralNetwork& node) {
                if(const auto outIsSome = node.getOut()) {
                    return outIsSome.value();
                }
                throw py::attribute_error("ParsingNeuralNetwork.out is not set(was .build() called?)");
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "passthroughs",
            [](ParsingNeuralNetwork& node) {
                if(const auto passthroughsIsSome = node.getPassthroughs()) {
                    return passthroughsIsSome.value();
                }
                throw py::attribute_error("ParsingNeuralNetwork.passthroughs is not set(was .build() called?)");
            })
        .def_property_readonly(
            "passthrough",
            [](ParsingNeuralNetwork& node) {
                if(const auto passthroughIsSome = node.getPassthrough()) {
                    return passthroughIsSome.value();
                }
                throw py::attribute_error("ParsingNeuralNetwork.passthrough is not set(was .build() called?)");
            });
}