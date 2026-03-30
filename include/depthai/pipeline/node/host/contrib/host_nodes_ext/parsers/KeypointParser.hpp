//
// Created by thwdpc on 7/25/25.
//

#pragma once
#include "BaseParser.hpp"
#include "messages/Keypoints.hpp"

namespace dai::node {

struct KeypointsOutputExpanded {
    std::string name;
    std::vector<int> dimensions;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(KeypointsOutputExpanded, name, dimensions)
};


class KeypointParser : virtual public CustomParser<KeypointParser> {
   public:
    constexpr static const char* NAME = "KeypointParser";

   protected:
    void buildImpl(const nn_archive::v1::Head& head, const nn_archive::v1::Model& model) override;
    void run() override;

    std::vector<nn_archive::v1::Output> keypointsOutputs{};
    // Maybe the same as keypointsOutputs
    std::vector<KeypointsOutputExpanded> keypointsOutputsSpatial{};
    std::vector<std::pair<xt::xkeep_slice<size_t>, xt::xkeep_slice<size_t>>> keypointSpatialSlicers{};
    std::vector<KeypointsOutputExpanded> confidenceOutputs{};
    std::vector<std::pair<xt::xkeep_slice<size_t>, xt::xkeep_slice<size_t>>> confidenceSlicers{};
    uint16_t nKeypoints = 17;
    // dimensionality: 2D or 3D
    std::vector<std::string> keypointNames{};
    std::optional<std::vector<std::pair<uint16_t, uint16_t>>> skeletonEdges = std::nullopt;

    std::function<std::shared_ptr<Buffer>(std::shared_ptr<const NNData>,  xt::xtensor<float_t, 2>&&, std::optional<xt::xtensor<float_t, 2>>, std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&&)>
        makeOutputMessage;
};
}  // namespace dai::node