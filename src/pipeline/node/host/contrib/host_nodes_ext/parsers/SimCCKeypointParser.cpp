#include "parsers/SimCCKeypointParser.hpp"

#include "depthai/nn_archive/v1/InputType.hpp"
#include "messages/Keypoints.hpp"
#include "pipeline/ThreadedNodeImpl.hpp"
#include "spdlog/spdlog.h"
#include "utility/ErrorMacros.hpp"
#include "utility/Logging.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/misc/xsort.hpp"
#include "xtensor/views/xindex_view.hpp"
#include <xtensor/views/xstrided_view.hpp>

namespace dai::node {

void SimCCKeypointParser::foggyGuessesForOneDim(const nn_archive::v1::Head& head,
                                                const nn_archive::v1::Model& model,
                                                const nn_archive::v1::Input& imgInput,
                                                const std::pair<std::optional<int64_t>, std::optional<int64_t>>& imgWHMaybe) {
    std::pair<uint16_t, uint16_t> imgWidthHeight;
    if(imgWHMaybe.first && imgWHMaybe.second) {
        imgWidthHeight = {static_cast<uint16_t>(*imgWHMaybe.first), static_cast<uint16_t>(*imgWHMaybe.second)};
    } else {
        std::unordered_set<int64_t> seen;
        for(int64_t n : imgInput.shape) {
            if(seen.find(n) != seen.end()) {
                logger::warn("Input layout not found, assuming duplicated dimension {} is width and height", n);
                imgWidthHeight = {static_cast<uint16_t>(n), static_cast<uint16_t>(n)};
                break;
            }
            seen.insert(n);
        }
    }

    DAI_CHECK_V(imgWidthHeight.first && imgWidthHeight.first == imgWidthHeight.second,
                "Input image width and height must match for a single output combined dim")
    uint16_t inputImgDim = imgWidthHeight.first;

    simCCDimLengths = {static_cast<uint16_t>(inputImgDim * pixelSubdivisions)};

    if(auto shapeIsSome = keypointsOutputs[0].shape) {
        // N 1 C ( 2/3 ) * KPDims D = W = H
        const int product = std::accumulate(shapeIsSome->begin(), shapeIsSome->end(), 1, std::multiplies<int>());
        const int expected = nKeypoints * static_cast<uint8_t>(valuesPerKeypoint) * (pixelSubdivisions * (inputImgDim * 2));
        // I don't know what to do, phone it in
        DAI_CHECK_V(product == expected, "The developer of this project was not even sure a model like yours would exist, best efforts were made but alas...")
        // Whether the keypoint # dim is separate from the keypoint XY(Z) dimension(yolo does this)
        for(auto& dim : *shapeIsSome) {
            if(dim == nKeypoints * static_cast<uint8_t>(valuesPerKeypoint)) {
                collapsedDimsAreInterleaved = head.metadata.extraParams.value("collapsed_dims_are_interleaved", false);
                logger::trace("Observed output dim {} == {} * {}, assuming collapsed dim", dim, nKeypoints, static_cast<uint8_t>(valuesPerKeypoint));
                break;
            }
        }
    }
}

// Parse the outputs to find XY(Z) ordering, throwing if the output layout exists and "disagrees" with the order of the outputs in the config
void SimCCKeypointParser::inferConfigFromMultipleOutputs(const nn_archive::v1::Head& head,
                                                         const nn_archive::v1::Model& model,
                                                         const nn_archive::v1::Input& imgInput,
                                                         std::pair<std::optional<int64_t>, std::optional<int64_t>>& imgWidthHeight) {
    if(const auto& [maybeWidth, maybeHeight] = imgWidthHeight; maybeWidth && maybeHeight) {
        simCCDimLengths = {static_cast<uint16_t>(*maybeWidth * pixelSubdivisions), static_cast<uint16_t>(*maybeHeight * pixelSubdivisions)};
        if(valuesPerKeypoint == singlekp::ValuesPerKeypoint::Three) {
            replicateXDimToZDim = head.metadata.extraParams.value("input_z_dim_equal_x_dim", replicateXDimToZDim);
            simCCDimLengths.push_back(static_cast<uint16_t>(replicateXDimToZDim ? *maybeWidth * pixelSubdivisions : *maybeHeight * pixelSubdivisions));
        }
    } else {
        throw std::runtime_error(fmt::format(
            "Refusing to construct node {} without a type: image model.input with populated layout which contains W and H, too much guesswork involved.",
            getName()));
    }
    // X == W || X == D, Y == H || Y == D, Z == D
    std::vector<char> outputDimPx = {'w', 'h', 'd'};
    bool anyMissingLayouts = false;
    for(int i = 0; i < keypointsOutputs.size(); i++) {
        if(auto& shapeIsSome = keypointsOutputs[i].shape) {
            std::vector<int64_t> shapeV = *shapeIsSome;
            DAI_CHECK_V(shapeV.back() == simCCDimLengths[i],
                        "Expected output '{}' last dim {} == {}(pixel dimension from input * SimCC pixel subdivisions)",
                        keypointsOutputs[i].name,
                        shapeV.back(),
                        simCCDimLengths[i]);
        }
        if(auto& layoutIsSome = keypointsOutputs[i].layout) {
            bool found = false;
            for(auto& dim : *layoutIsSome) {
                char dimL = std::tolower(dim);
                found |= dimL == outputDimPx[i] || dimL == 'd';
            }
            DAI_CHECK_V(found, "Output '{}' layout does not contain expected `{}` or wildcard `d` (case insensitive)", keypointsOutputs[i].name, outputDimPx[i])
        } else {
            anyMissingLayouts = true;
        }
    }
    if(anyMissingLayouts) {
        std::string assumptions = "";
        std::vector outputDims = {'X', 'Y', 'Z'};
        for(int i = 0; i < keypointsOutputs.size(); i++) {
            std::string assumedLayout = fmt::format("[{}, {} * {}]", nKeypoints, simCCDimLengths[i] / 2.0f, pixelSubdivisions);
            assumptions += fmt::format("- Output {} is dimension: {}, with layout {}\n", keypointsOutputs[i].name, outputDims[i], assumedLayout);
        }
        logger::warn(
            "One or more output layouts not found for keypoint outputs, it is highly encouraged to include output layouts, assuming/found the following: \n{}",
            assumptions);
    }
}

void SimCCKeypointParser::buildImpl(const nn_archive::v1::Head& head, const nn_archive::v1::Model& model) {
    KeypointParser::buildImpl(head, model);
    try {
        pixelSubdivisions = head.metadata.extraParams["pixel_subdivisions"];
    } catch(...) {
        logger::warn("{}: pixel_subdivisions not found in metadata, using default value {}", getName(), pixelSubdivisions);
    }

    // Find the image input so we can educate our guesses
    auto imgInput = std::find_if(model.inputs.begin(), model.inputs.end(),  [](const auto& i) { return i.inputType == nn_archive::v1::InputType::IMAGE; });
    DAI_CHECK_V(imgInput != model.inputs.end(), "Expecting image input to find pixel comparisons")

    std::pair<std::optional<int64_t>, std::optional<int64_t>> imgWidthHeight;
    if(const auto inputImgLayout = imgInput->layout) {
        const std::string& layout = *inputImgLayout;
        DAI_CHECK_V(imgInput->shape.size() == layout.size(), "Input shape and layout length mismatch")
        for(int i = 0; i < inputImgLayout->size(); i++) {
            if(std::tolower(layout[i]) == 'w') {
                imgWidthHeight.first = imgInput->shape.at(i);
            }
            if(std::tolower(layout[i]) == 'h') {
                imgWidthHeight.second = imgInput->shape.at(i);
            }
        }
    }

    if(keypointsOutputs.size() == 1) {
        foggyGuessesForOneDim(head, model, *imgInput, imgWidthHeight);
    } else {
        inferConfigFromMultipleOutputs(head, model, *imgInput, imgWidthHeight);
    }

    // TODO don't make assumptions that last dim of output is simcc
}

void SimCCKeypointParser::run() {
    while(isRunning()) {
        const auto t1 = std::chrono::high_resolution_clock::now();
        // TODO not just my application
        std::shared_ptr<Keypoints3D3C> outputMessage;
        // Build a new nested scope to prevent use after move because I wish I was in rust
        {
            std::shared_ptr<NNData> result;
            try {
                result = this->input.get<NNData>();
            } catch([[maybe_unused]] MessageQueue::QueueException(&e)) {
                break;
            }

            xt::xtensor<float_t, 2> output = xt::empty<float_t>({static_cast<size_t>(nKeypoints), keypointsOutputs.size() * 2});
            for(int i = 0; i < keypointsOutputs.size(); i++) {
                std::string& layerName = keypointsOutputs[i].name;
                DAI_CHECK_V(result->hasLayer(layerName), "Expecting layer {} in NNData", layerName)
                auto prediction = xt::reshape_view(
                    result->getTensor<float_t>(layerName), {nKeypoints, simCCDimLengths[i]}); // view as KP#, PX * splitRatio(pixelSubdivisions)

                // TODO IF THIS IS SLOW https://github.com/xtensor-stack/xtensor/issues/2046
                // locationsUnsplit: shape = [nKeypoints], integer indices
                auto locationsUnsplit = xt::flatten(xt::argmax(prediction, -1));
                // Normalize
                auto locationsNormalized = locationsUnsplit / static_cast<float_t>(simCCDimLengths[i]);

                // TODO XTensor not working as shown in docs, I'm done caring and just using xt::amax
                // // Create indices shape [nKeypoints, 2], where kp index, px subdivision index
                // xt::xarray<size_t> indices = xt::stack(xt::xtuple(xt::arange<size_t>(nKeypoints), locationsUnsplit), 1);
                // std::vector<size_t> expected{nKeypoints, 2};
                // DAI_CHECK_IN(indices.shape() == expected);
                // xt::xarray<float> a = {{11, 12, 13}, {24, 25, 26}};
                // xt::xarray<size_t> idxs = {{0, 0}, {1, 0}, {0, 1}};
                // auto b = xt::index_view(a, idxs);
                // for(auto el : b) {
                //     std::cout << el << ", "; // Why does this print 11, 11, 12, 11, 11, 12, ??????
                // }
                // std::cout << std::endl;
                // xt::ravel_return_type_t<std::vector<xt::svector<unsigned long>>, xt::ravel_tensor_tag> flat_indices =
                //     xt::ravel_indices(xt::argwhere(a >= 6), a.shape());
                // why is this 266 not 133?
                // auto confidence = xt::index_view(prediction, indices);

                // Clamp
                auto confidenceClamped = xt::clip(xt::amax(prediction, -1), 0.0f, 1.0f);

                xt::view(output, xt::all(), 2 * i) = locationsNormalized;
                xt::view(output, xt::all(), 2 * i + 1) = confidenceClamped;
            }
            auto skeletonEdgesCpy = skeletonEdges;
            outputMessage = std::make_shared<Keypoints3D3C>(std::move(result), std::move(output), std::move(skeletonEdgesCpy));
        }
        const auto t2 = std::chrono::high_resolution_clock::now();
        pimpl->logger->trace("SimCCKeypointParser: Time taken: {}ms", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000);
        out.send(std::static_pointer_cast<Buffer>(outputMessage));
    }
}

}  // namespace dai::node