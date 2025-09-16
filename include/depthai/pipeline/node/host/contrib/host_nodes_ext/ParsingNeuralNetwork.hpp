//
// Created by thwdpc on 7/24/25.
//
#pragma once
#include <variant>
#include <vector>

#include "../ParserGenerator.hpp"
#include "depthai/depthai.hpp"
#include "parsers/BaseParser.hpp"
#include "parsers/KeypointParser.hpp"

namespace dai::node {

class ParsingNeuralNetwork : public CustomThreadedNode<ParsingNeuralNetwork> {
   public:
    std::shared_ptr<ParsingNeuralNetwork> build(Output& input, const NNArchive& nnArchive);
    std::shared_ptr<ParsingNeuralNetwork> build(const std::shared_ptr<Camera>& input, NNModelDescription modelDesc, std::optional<float> fps = std::nullopt);
    std::shared_ptr<ParsingNeuralNetwork> build(const std::shared_ptr<Camera>& input, const NNArchive& nnArchive, std::optional<float> fps = std::nullopt);

    template <typename T>
    std::optional<size_t> getIndexOfFirstParserOfType() const {
        if(auto& sIsSome = s) {
            const auto which = std::find_if(sIsSome->parsers.begin(), sIsSome->parsers.end(), [](const auto& p) {
                return std::visit([](auto& anyP) { return std::dynamic_pointer_cast<T>(anyP) != nullptr; }, p);
                ;
            });
            return which == sIsSome->parsers.end() ? std::nullopt : static_cast<std::optional<size_t>>(std::distance(sIsSome->parsers.begin(), which));
        }
        return std::nullopt;
    }

    void run() override;

    std::optional<std::reference_wrapper<InputMap>> getInputs() const { return s.has_value() ? std::optional<std::reference_wrapper<InputMap>>(s->inputs) : std::nullopt;}
    std::optional<std::reference_wrapper<Input>> getInput() const { return s.has_value() ? std::optional<std::reference_wrapper<Input>>(s->input) : std::nullopt;}
    std::optional<std::reference_wrapper<Output>> getPassthrough() const { return s.has_value() ? std::optional<std::reference_wrapper<Output>>(s->passthrough) : std::nullopt;}
    std::optional<std::reference_wrapper<OutputMap>> getPassthroughs() const { return s.has_value() ? std::optional<std::reference_wrapper<OutputMap>>(s->passthroughs) : std::nullopt;}
    std::optional<std::reference_wrapper<Output>> getOut() const { return s.has_value() ? std::optional<std::reference_wrapper<Output>>(s->out) : std::nullopt;}

   private:
    struct BuiltState {
        BuiltState(const std::vector<HostOrDeviceParser>& parsers, const std::shared_ptr<NeuralNetwork>& nn, Output& out)
            : parsers(parsers), nn(nn), out(out) {}
        std::vector<HostOrDeviceParser> parsers;
        std::shared_ptr<NeuralNetwork> nn;
        InputMap& inputs = nn->inputs;
        Input& input = nn->input;
        Output& passthrough = nn->passthrough;
        OutputMap& passthroughs = nn->passthroughs;
        Output& out;
    };
    std::shared_ptr<ParsingNeuralNetwork> ensure_nn_and_build_via_closure(
       const std::function<const NNArchive&(std::shared_ptr<NeuralNetwork>)>& builder_returning_archive);

    Output& linkOneOrMoreParsers(const std::shared_ptr<NeuralNetwork>& nn, std::vector<HostOrDeviceParser> newParsers);
    std::optional<Subnode<Sync>> parserSync = std::nullopt;
    std::optional<BuiltState> s;
};

}  // namespace dai::node
