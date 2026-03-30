//
// Created by thwdpc on 7/25/25.
//

#include "parsers/KeypointParser.hpp"

#include <spdlog/spdlog.h>

#include "fmt/ranges.h"
#include "pipeline/ThreadedNodeImpl.hpp"
#include "utility/ErrorMacros.hpp"
#include "utility/Logging.hpp"

#define XT_KEEP(arr, n) XT_KEEP_##n(arr)
#define XT_KEEP_1(arr) xt::keep(arr[0])
#define XT_KEEP_2(arr) xt::keep(arr[0], arr[1])
#define XT_KEEP_3(arr) xt::keep(arr[0], arr[1], arr[2])

std::pair<uint8_t, std::pair<xt::xkeep_slice<size_t>, xt::xkeep_slice<size_t>>> mappingToReorderedIndexes(const std::vector<int>& mapping,
                                                                                                          std::string& nodeName,
                                                                                                          std::string& layerName) {
    // Collect (destinationIndex, sourceIndex) pairs, skipping dropped entries
    std::vector<std::pair<int, size_t>> pairs;
    pairs.reserve(mapping.size());

    for(size_t srcIdx = 0; srcIdx < mapping.size(); ++srcIdx) {
        const int dstIdx = mapping[srcIdx];
        if(dstIdx >= 0) {
            pairs.emplace_back(dstIdx, srcIdx);
        }
    }

    // Sort by destination index so the output is in destination order
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Extract just the source indexes, now ordered by their destination position
    std::vector<size_t> lhsKeeps;
    std::vector<size_t> rhsKeeps;
    lhsKeeps.reserve(pairs.size());
    rhsKeeps.reserve(pairs.size());
    for(const auto& [dstIdx, srcIdx] : pairs) {
        lhsKeeps.push_back(dstIdx);
        rhsKeeps.push_back(srcIdx);
    }

    switch(lhsKeeps.size()) {
        case 1:
            return {1, {XT_KEEP(lhsKeeps, 1), XT_KEEP(rhsKeeps, 1)}};
        case 2:
            return {2, {XT_KEEP(lhsKeeps, 2), XT_KEEP(rhsKeeps, 2)}};
        case 3:
            return {3, {XT_KEEP(lhsKeeps, 3), XT_KEEP(rhsKeeps, 3)}};
        default:
            throw std::runtime_error(
                fmt::format("{}: confidence output {} has {} final dimensions, which is not supported", nodeName, layerName, lhsKeeps.size()));
    }
}

namespace dai::node {
void KeypointParser::buildImpl(const nn_archive::v1::Head& head, const nn_archive::v1::Model& model) {
    auto nodeName = std::string{getName()};
    bool fallback = false;
    if(const auto layers = head.metadata.keypointsOutputs) {
        for(auto& layerName : *layers) {
            auto output = std::find_if(model.outputs.begin(), model.outputs.end(), [&](const auto& o) { return o.name == layerName; });
            DAI_CHECK_V(output != model.outputs.end(), "{}: keypoint output {} not found in model", nodeName, layerName);
            keypointsOutputs.push_back(*output);
        }
    } else {
        logger::trace("KeypointParser(or subclass) did not receive keypoints_outputs, fallback to using all outputs");
        for(auto& output : model.outputs) {
            keypointsOutputs.push_back(output);
        };
        fallback = true;
    }

    const uint8_t ko_sz = keypointsOutputs.size();
    if(ko_sz < 1 || ko_sz > 3) {
        const std::string where = fallback ? "During fallback to use all outputs" : "Configured keypoints_outputs";
        throw std::runtime_error(fmt::format("{w}: size {sz} must satisfy 1 <= {sz} <= 3 ", fmt::arg("w", where), fmt::arg("sz", ko_sz)));
    }

    if(auto outputs3d = head.metadata.extraParams.value<std::optional<std::vector<KeypointsOutputExpanded>>>("keypoints_outputs_spatial", std::nullopt)) {
        DAI_CHECK_V(!fallback, "keypoints_outputs must contain values if keypoints_outputs_spatial is populated");
        uint8_t dimensionsFound = 0;
        bool xFound = false, yFound = false, zFound = false;
        for(auto& layer : *outputs3d) {
            auto output = std::find_if(model.outputs.begin(), model.outputs.end(), [&](const auto& o) { return o.name == layer.name; });
            DAI_CHECK_V(output != model.outputs.end(), "{}: 3D keypoint output {} not found in model", nodeName, layer.name);
            keypointsOutputsSpatial.push_back(layer);
            dimensionsFound += layer.dimensions.size();
            xFound |= std::find_if(layer.dimensions.begin(), layer.dimensions.end(), [&](const auto& o) { return o == 0; }) != layer.dimensions.end();
            yFound |= std::find_if(layer.dimensions.begin(), layer.dimensions.end(), [&](const auto& o) { return o == 1; }) != layer.dimensions.end();
            zFound |= std::find_if(layer.dimensions.begin(), layer.dimensions.end(), [&](const auto& o) { return o == 2; }) != layer.dimensions.end();
            DAI_CHECK_V(!layer.dimensions.empty(), "{}: Dimension array specified but not populated for confidence layer '{}'", nodeName, layer.name);
            keypointSpatialSlicers.emplace_back(mappingToReorderedIndexes(layer.dimensions, nodeName, layer.name).second);
        }
        DAI_CHECK_V(dimensionsFound == 3, "Expected 3 dimensions per 3D keypoint output, got {} dimensions", dimensionsFound);
        DAI_CHECK_V(xFound && yFound && zFound,
                    "Expected 3D keypoint outputs to cumulatively contain all of dimensions: [0, 1, 2] corresponding to [x, y, z] dimensions");
    }

    uint8_t confFound = 0;
    if(auto conf = head.metadata.extraParams.value<std::optional<std::vector<KeypointsOutputExpanded>>>("confidence_outputs", std::nullopt)) {
        confidenceOutputs = *conf;
        for(auto& layer : confidenceOutputs) {
            auto output = std::find_if(model.outputs.begin(), model.outputs.end(), [&](const auto& o) { return o.name == layer.name; });
            DAI_CHECK_V(output != model.outputs.end(), "{}: confidence output {} not found in model", nodeName, layer.name);
            DAI_CHECK_V(!layer.dimensions.empty(), "{}: Dimension array specified but not populated for spatial keypoint layer '{}'", nodeName, layer.name);
            auto [numDims, slicers] = mappingToReorderedIndexes(layer.dimensions, nodeName, layer.name);
            confidenceSlicers.emplace_back(slicers);
            confFound += numDims;
        }
    }

    DAI_CHECK_V(ko_sz == 1 || ko_sz == 2,
                "Expected one output per keypoint dimension, or one output that contains all keypoints(ordered as u, v), got {} layers vs dimensionality {}.",
                ko_sz,
                2);

    if(const auto n = head.metadata.nKeypoints) {
        nKeypoints = *n;
    } else {
        logger::warn("SimCCKeypointParser did not receive n_keypoints, defaulting to standard COCO 17. Populating this field is strongly encouraged");
    }

    keypointNames = head.metadata.extraParams.value("keypoint_names", keypointNames);
    skeletonEdges = head.metadata.extraParams.value("skeleton_edges", skeletonEdges);
}

void KeypointParser::run() {
    uint8_t confidenceValues = 0;
    for(auto& output : confidenceOutputs) {
        for(auto& dim : output.dimensions) {
            if(dim < 0) {
                continue;
            }
            confidenceValues += 1;
        }
    }

    while(mainLoop()) {
        std::chrono::time_point<std::chrono::system_clock> t1, t2;
        const std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Buffer> outputMessage;
        {
            std::shared_ptr<NNData> result;
            try {
                result = this->input.get<NNData>();
            } catch([[maybe_unused]] MessageQueue::QueueException(&e)) {
                break;
            }
            t1 = std::chrono::high_resolution_clock::now();

            // {
            //     auto z = result->getTensor<float_t>("dbg_torso_root_center_pred");
            //     std::vector<float_t> zValsFlat;
            //     zValsFlat.reserve(z.shape()[0]);
            //     std::copy(z.begin(), z.end(), std::back_inserter(zValsFlat));
            //     // print all zValsFlat to warn channel:
            //     logger::warn("torso root: [{}]", fmt::join(zValsFlat, ", "));
            // }
            // {
            //     auto z = result->getTensor<float_t>("dbg_kp_z_pred");
            //     std::vector<float_t> zValsFlat;
            //     zValsFlat.reserve(z.shape()[0]);
            //     std::copy(z.begin(), z.end(), std::back_inserter(zValsFlat));
            //     // print all zValsFlat to warn channel:
            //     logger::warn("z offsets: [{}]", fmt::join(zValsFlat, ", "));
            // }

            xt::xtensor<float_t, 2> outputPx = xt::empty<float_t>({static_cast<size_t>(nKeypoints), static_cast<size_t>(2)});
            for(int i = 0; i < keypointsOutputs.size(); i++) {
                std::string& layerName = keypointsOutputs[i].name;
                DAI_CHECK_V(result->hasLayer(layerName), "Expecting layer {} in NNData", layerName)
                auto prediction = xt::view(result->getTensor<float_t>(layerName), static_cast<size_t>(nKeypoints), xt::all());
                auto consolidatedDim = prediction.shape()[1] == 2 ? xt::keep(i, i + 1) : xt::keep(i);
                xt::view(outputPx, xt::all(), consolidatedDim) = prediction;
            }
            xt::xtensor<float_t, 1> confidences;
            if(confidenceOutputs.empty()) {
                confidences = xt::ones<float_t>({static_cast<size_t>(nKeypoints)});
            } else {
                xt::xtensor<float_t, 2> confidences_2d = xt::empty<float_t>({static_cast<size_t>(nKeypoints), static_cast<size_t>(confidenceValues)});
                for(int i = 0; i < confidenceOutputs.size(); i++) {
                    auto& confidenceLayer = confidenceOutputs[i];
                    // This was checked in build already
                    // DAI_CHECK_IN(!confidenceLayer.dimensions.empty())
                    std::string& layerName = confidenceLayer.name;
                    DAI_CHECK_V(result->hasLayer(layerName), "Expecting layer {} in NNData", layerName)
                    xt::xarray<float_t> tensor = result->getTensor<float_t>(layerName);
                    auto confidenceReshaped = xt::eval(xt::reshape_view(tensor, {static_cast<size_t>(nKeypoints), confidenceLayer.dimensions.size()}));
                    auto& [lhsKeep, rhsKeep] = confidenceSlicers[i];
                    xt::view(confidences_2d, xt::all(), lhsKeep) = xt::view(confidenceReshaped, xt::all(), rhsKeep);
                }
                if(confidenceValues > 1) {
                    confidences = xt::mean(confidences_2d, 1);
                } else {
                    confidences = xt::reshape_view(confidences_2d, {static_cast<size_t>(nKeypoints)});
                }
                // TODO maybe different reduction strategies?
            }
            std::optional<xt::xtensor<float_t, 2>> keypoints3D = std::nullopt;
            if(!keypointsOutputsSpatial.empty()) {
                keypoints3D = xt::empty<float_t>({static_cast<size_t>(nKeypoints), static_cast<size_t>(3)});
                for(int i = 0; i < keypointsOutputsSpatial.size(); i++) {
                    auto& kpLayer = keypointsOutputsSpatial[i];
                    // This was checked in build already
                    // DAI_CHECK_IN(!layer.dimensions.empty())
                    std::string& layerName = kpLayer.name;
                    DAI_CHECK_V(result->hasLayer(layerName), "Expecting layer {} in NNData", layerName)
                    xt::xarray<float_t> tensor = result->getTensor<float_t>(layerName);
                    // TODO why does SNPE force this one to be NCD and the rest NDC?????????????
                    auto keypointsReshaped = xt::transpose(xt::reshape_view(tensor, {  kpLayer.dimensions.size(), static_cast<size_t>(nKeypoints) }));
                    auto& [lhsKeep, rhsKeep] = keypointSpatialSlicers[i];
                    xt::view(*keypoints3D, xt::all(), lhsKeep) = xt::view(keypointsReshaped, xt::all(), rhsKeep);
                }
            }
            auto skeletonEdgesCpy = skeletonEdges;
            if(keypointsOutputsSpatial.empty()) {
                outputMessage = std::make_shared<Keypoints2D>(std::move(result), std::move(outputPx), std::move(confidences), std::move(skeletonEdgesCpy));
            } else {
                outputMessage = std::make_shared<Keypoints3D>(
                    std::move(result), std::move(outputPx), std::move(keypoints3D.value()), std::move(confidences), std::move(skeletonEdgesCpy));
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        pimpl->logger->trace("KeypointParser:   Message Wait: {}ms   Processing time taken: {}μs",
                             std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000,
                             std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
        out.send(std::static_pointer_cast<Buffer>(outputMessage));
    }
}

}  // namespace dai::node