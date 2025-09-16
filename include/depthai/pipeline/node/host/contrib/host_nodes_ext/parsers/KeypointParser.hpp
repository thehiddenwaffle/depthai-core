//
// Created by thwdpc on 7/25/25.
//

#pragma once
#include "messages/Keypoints.hpp"
#include "BaseParser.hpp"

namespace dai::node {


class KeypointParser : virtual public CustomParser<KeypointParser> {
public:
    constexpr static const char* NAME = "KeypointParser";

protected:
    void buildImpl(const nn_archive::v1::Head& head, const nn_archive::v1::Model& model) override;
    void run() override;

    std::vector<nn_archive::v1::Output> keypointsOutputs{};
    uint16_t nKeypoints = 17;
    // dimensionality: 2D or 3D
    singlekp::ValuesPerKeypoint valuesPerKeypoint = singlekp::ValuesPerKeypoint::Two;
    std::vector<std::string> keypointNames{};
    std::optional<std::vector<std::pair<uint16_t, uint16_t>>> skeletonEdges = std::nullopt;
};
}