//
// Created by thwdpc on 7/24/25.
//

#include "ParsingNeuralNetwork.hpp"

#include <spdlog/spdlog.h>

#include <iosfwd>
#include <utility/ErrorMacros.hpp>
#include <utility/Logging.hpp>
#include <utility>
#include <variant>
#include <vector>
namespace dai::node {

std::shared_ptr<ParsingNeuralNetwork> ParsingNeuralNetwork::ensure_nn_and_build_via_closure(
    const std::function<const NNArchive&(std::shared_ptr<NeuralNetwork>)>& builder_returning_archive) {
    if(auto sIsSome = s){
        logger::warn("ParsingNeuralNetwork Node being re-built(build function called twice), deleting old neural network node");
        getParentPipeline().remove(s->nn);
        for(auto parser : s->parsers) {
            std::visit([this](auto& p) { getParentPipeline().remove(p); }, parser);
        }
        if(s->parsers.size() > 1) {
            parserSync = std::nullopt;
        }
        s = std::nullopt;
    }

    auto nn = getParentPipeline().create<NeuralNetwork>();
    const NNArchive& archive = builder_returning_archive(nn);
    std::vector<HostOrDeviceParser> newParsers = ParserGenerator::generateAllParsers(getParentPipeline(), archive);
    Output& syncOrSingle = linkOneOrMoreParsers(nn, newParsers);
    s.emplace(newParsers, nn, syncOrSingle);

    return std::static_pointer_cast<ParsingNeuralNetwork>(shared_from_this());
}

std::shared_ptr<ParsingNeuralNetwork> ParsingNeuralNetwork::build(Output& input, const NNArchive& nnArchive) {
    return ensure_nn_and_build_via_closure([this, &input, &nnArchive](std::shared_ptr<NeuralNetwork> nn) -> const NNArchive& {
        nn->build(input, nnArchive);
        return nnArchive;
    });
}

std::shared_ptr<ParsingNeuralNetwork> ParsingNeuralNetwork::build(const std::shared_ptr<Camera>& input,
                                                                  NNModelDescription modelDesc,
                                                                  const std::optional<float> fps) {
    return ensure_nn_and_build_via_closure([this, &input, &modelDesc, &fps](std::shared_ptr<NeuralNetwork> nn) -> const NNArchive& {
        // mash these into the same statement for future proofing
        DAI_CHECK(nn->build(input, std::move(modelDesc), fps) != nullptr && nn->getNNArchive().has_value(), "NeuralNetwork node failed to create an archive(and did so silently?)");
        return nn->getNNArchive()->get();
    });
}

std::shared_ptr<ParsingNeuralNetwork> ParsingNeuralNetwork::build(const std::shared_ptr<Camera>& input,
                                                                  const NNArchive& nnArchive,
                                                                  const std::optional<float> fps) {
    return ensure_nn_and_build_via_closure([this, &input, &nnArchive, &fps](std::shared_ptr<NeuralNetwork> nn) -> const NNArchive& {
        nn->build(input, nnArchive, fps);
        return nnArchive;
    });
}

Node::Output& ParsingNeuralNetwork::linkOneOrMoreParsers(const std::shared_ptr<NeuralNetwork>& nn, std::vector<HostOrDeviceParser> newParsers) {
    DAI_CHECK_IN(!newParsers.empty());
    if(auto& newParser = newParsers.front(); newParsers.size() == 1) {
        return std::visit(
            [nn](auto& p) {
                nn->out.link(p->input);
                return std::ref(p->out);
            },
            newParser);
    }
    auto sync = parserSync.value_or(Subnode<Sync>(*this, "sync"));
    for(std::size_t idx = 0; idx < newParsers.size(); ++idx) {
        std::visit(
            [nn, idx, sync](auto& p) {
                nn->out.link(p->input);
                p->out.link(sync->inputs[std::to_string(idx)]);
            },
            newParsers[idx]);
    }
    parserSync = sync;
    return sync->out;
}

void ParsingNeuralNetwork::run() {
    DAI_CHECK(s.has_value(), "ParsingNeuralNetwork run before NN was initialized(was this node built via `build()`?)");
}


}  // namespace dai::node
