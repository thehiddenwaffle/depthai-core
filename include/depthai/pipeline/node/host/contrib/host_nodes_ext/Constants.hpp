//
// Created by thwdpc on 9/3/25.
//

// ReSharper disable CppIdenticalOperandsInBinaryExpression
#pragma once
#include "depthai/common/Color.hpp"

const dai::Color PRIMARY_COLOR = dai::Color(21 / 255, 127 / 255, 88 / 255, 1.0);
const dai::Color TRANSPARENT_PRIMARY_COLOR = dai::Color(21 / 255, 127 / 255, 88 / 255, 0.2);
const dai::Color SECONDARY_COLOR = dai::Color(240 / 255, 240 / 255, 240 / 255, 1.0);
const dai::Color FONT_COLOR = dai::Color(255 / 255, 255 / 255, 255 / 255, 1.0);
constexpr float_t FONT_SIZE_PER_HEIGHT = 1.0 / 30;
constexpr float_t SMALLER_FONT_SIZE_PER_HEIGHT = FONT_SIZE_PER_HEIGHT / 2;
const dai::Color FONT_BACKGROUND_COLOR = dai::Color(0.0, 0.0, 0.0, 0.0);
const dai::Color KEYPOINT_COLOR = dai::Color(0.0, 1.0, 1.0, 1.0);
constexpr float_t KEYPOINT_THICKNESS_PER_RESOLUTION = 1.0 / 1000;
constexpr float_t SMALLER_KEYPOINT_THICKNESS_PER_RESOLUTION = KEYPOINT_THICKNESS_PER_RESOLUTION / 2;
const dai::Color DETECTION_COLOR = dai::Color(200 / 255, 200 / 255, 0.0, 1.0);
constexpr float_t DETECTION_CORNER_SIZE = 0.04;
constexpr float_t DETECTION_BORDER_THICKNESS_PER_RESOLUTION = 1 / 1000;
constexpr float_t SMALLER_DETECTION_THRESHOLD = DETECTION_CORNER_SIZE * 3;
constexpr float_t SMALLER_DETECTION_CORNER_SIZE = DETECTION_CORNER_SIZE / 2;
constexpr float_t SMALLER_DETECTION_BORDER_THICKNESS_PER_RESOLUTION = (DETECTION_BORDER_THICKNESS_PER_RESOLUTION / 2);

static float_t keypointThickness(const float_t& width, const float_t& height) {
    return KEYPOINT_THICKNESS_PER_RESOLUTION * (width + height);
}

static float_t detectionThickness(const float_t& width, const float_t& height) {
    return DETECTION_BORDER_THICKNESS_PER_RESOLUTION * (width + height);
}
