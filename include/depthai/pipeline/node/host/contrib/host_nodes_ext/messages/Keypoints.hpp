//
// Created by thwdpc on 7/28/25.
//

#pragma once
#include <depthai/common/ImgTransformations.hpp>
#include <depthai/common/Point3f.hpp>
#include <depthai/pipeline/datatype/Buffer.hpp>
#include <depthai/pipeline/datatype/NNData.hpp>

#include "xtensor/views/xview.hpp"

namespace dai {

namespace singlekp {

enum class ValuesPerKeypoint : uint8_t { Two = 2, Three = 3 };

struct ValueWithConfidence {
    float_t value;
    float_t confidence;
};

template <ValuesPerKeypoint D, const bool InterleavedConfidences>
struct Keypoint;

// Primary template may be defined elsewhere or left as a declaration if only the
// two partial specializations are intended to be used.
template <ValuesPerKeypoint D>
struct Keypoint<D, false> {
    float_t values[static_cast<uint8_t>(D)];
    float_t confidence;
    static constexpr uint8_t value = static_cast<uint8_t>(D) + 1;  // D values + 1 confidence
    static constexpr ValuesPerKeypoint dimensionality = D;
    static constexpr bool interleavedConfidences = false;
};

template <ValuesPerKeypoint D>
struct Keypoint<D, true> {
    ValueWithConfidence data[static_cast<uint8_t>(D)];             // value, confidence, value, confidence, ...
    static constexpr uint8_t value = 2 * static_cast<uint8_t>(D);  // value, conf x D
    static constexpr ValuesPerKeypoint dimensionality = D;
    static constexpr bool interleavedConfidences = true;
};


};  // namespace singlekp

template <typename KP, typename = void>
struct has_interleaved_confidences_defined : std::false_type {};

template <typename KP>
struct has_interleaved_confidences_defined<KP, std::void_t<decltype(KP::interleavedConfidences)>> : std::true_type {};


template <typename KP, typename = void>
struct has_dimensionality_defined : std::false_type {};

template <typename KP>
struct has_dimensionality_defined<KP, std::void_t<decltype(KP::dimensionality)>> : std::true_type {};

// Helpers to compute index positions for keypoint coordinates in the layout generically.
// - For 2D: y is at index 1 (whether interleaved or not).
// - For 3D:
//     non-interleaved: y at 1, z at 2
//     interleaved: y at 2, z at 4
template <typename kp>
constexpr uint8_t keypoint_y_index() {
    if constexpr(has_interleaved_confidences_defined<kp>::value) {
        // if the type defines interleavedconfidences, use it to decide the index
        return kp::interleavedConfidences ? 2 : 1;
    }
    // else default to the false case
    return 1;
}

template <typename KP>
constexpr std::optional<uint8_t> keypoint_z_index() {
    // If we have dimensionality
    if constexpr(has_dimensionality_defined<KP>::value) {
        // And it's 3
        if constexpr(KP::dimensionality == singlekp::ValuesPerKeypoint::Three) {
            // Then check whether interleaved exists and is true
            if constexpr(has_interleaved_confidences_defined<KP>::value) {
                // If the type defines interleavedConfidences, use it to decide the index
                return KP::interleavedConfidences ? 4 : 2;
            }
        }
    }
    // Don't attempt to guess anything
    return std::nullopt;
}

template <singlekp::ValuesPerKeypoint D, const bool I>
class Keypoints : public Buffer {
   public:
    std::optional<ImgTransformation> transformation;

    // Return a constant reference to the contained floats. Only valid for the lifetime of this object
    [[nodiscard]] std::pair<const float_t*, size_t> getFlatData() const {
        constexpr uint8_t floatsPerKp = singlekp::Keypoint<D, I>::value;
        assert(keypointsXT.size() == floatsPerKp * distilledConfidence.size());
        return std::make_pair(keypointsXT.data(), keypointsXT.size());
    }

    template <typename T = singlekp::Keypoint<D, I>, std::enable_if_t<T::dimensionality == singlekp::ValuesPerKeypoint::Three, int> = 0>
    void toRHCS() {
        if(LHCS) {
            xt::view(keypointsXT, xt::all(), keypoint_z_index<T>().value()) *= -1;
            LHCS = false;
        }

    }
    template <typename T = singlekp::Keypoint<D, I>, std::enable_if_t<T::dimensionality == singlekp::ValuesPerKeypoint::Three, int> = 0>
    void toLHCS() {
        if(!LHCS) {
            xt::view(keypointsXT, xt::all(), keypoint_z_index<T>().value()) *= -1;
            LHCS = true;
        }
    }

    void toMeters() {
        if(mm) {
            auto yIndex = keypoint_y_index<singlekp::Keypoint<D, I>>();
            auto maybeZIndex = keypoint_z_index<singlekp::Keypoint<D, I> >();
            auto keeps = maybeZIndex.has_value() ? xt::keep(0, yIndex, maybeZIndex.value()) : xt::keep(0, yIndex);
            xt::view(keypointsXT, xt::all(), keeps) *= 0.001f;
            mm = false;
        }
    }

    void toMillimeters() {
        if(!mm) {
            auto yIndex = keypoint_y_index<singlekp::Keypoint<D, I>>();
            auto maybeZIndex = keypoint_z_index<singlekp::Keypoint<D, I> >();
            auto keeps = maybeZIndex.has_value() ? xt::keep(0, yIndex, maybeZIndex.value()) : xt::keep(0, yIndex);
            xt::view(keypointsXT, xt::all(), keeps) *= 1000.0f;
            mm = true;
        }
    }

    [[nodiscard]] std::shared_ptr<const std::vector<float_t>> getValuesOnly() const {
        auto yIndex = keypoint_y_index<singlekp::Keypoint<D, I>>();
        auto maybeZIndex = keypoint_z_index<singlekp::Keypoint<D, I> >();
        auto keeps = maybeZIndex.has_value() ? xt::keep(0, yIndex, maybeZIndex.value()) : xt::keep(0, yIndex);
        auto view = xt::view(keypointsXT, xt::all(), keeps);
        auto ret = std::make_shared<std::vector<float_t>>();
        ret->reserve(view.size());
        ret->assign(view.begin(), view.end());
        return ret;
    }

    Keypoints(std::shared_ptr<const NNData> other, xt::xarray<float>&& planarStackedKeypoints, std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&& skeletonEdges);
    Keypoints(std::shared_ptr<Keypoints> other);

    template <typename T = singlekp::Keypoint<D, I>, std::enable_if_t<T::dimensionality == singlekp::ValuesPerKeypoint::Three, int> = 0>
    void localizeAndDenormalizeFromAnchorPoint(const Point3f& anchorPoint, float xToZScalingFactor = 1.0f) {
        if(auto& transformChecked = transformation) {
            return locAndDenormAnchorErased(anchorPoint, transformChecked.value(), xToZScalingFactor, keypoint_y_index<T>(), keypoint_z_index<T>().value());
        }
        throw std::runtime_error("Keypoint message does not have an attached transformation, unable to localize");
    }

    VisualizeType getVisualizationMessage() const override;

    xt::xtensor<float, 2> keypointsXT;
    xt::xtensor<float, 2> displayKeypointsXT;
private:
    void locAndDenormAnchorErased(
        const Point3f& anchorPoint, const ImgTransformation& transformation, float xToZScalingFactor = 1.0f, size_t yIndex = 0, size_t zIndex = 0);

    std::optional<std::vector<std::pair<uint16_t, uint16_t>>> skeletonEdges = std::nullopt;

    xt::xtensor<float_t, 1>  distilledConfidence;
    // whether this is in the left-hand-coordinate-system, which is depthai's default
    bool LHCS = true;
    bool mm = true;
};

template class Keypoints<singlekp::ValuesPerKeypoint::Two, false>;
typedef Keypoints<singlekp::ValuesPerKeypoint::Two, false> Keypoints2D;
template class Keypoints<singlekp::ValuesPerKeypoint::Two, true>;
typedef Keypoints<singlekp::ValuesPerKeypoint::Two, true> Keypoints2D2C;

template class Keypoints<singlekp::ValuesPerKeypoint::Three, false>;
typedef Keypoints<singlekp::ValuesPerKeypoint::Three, false> Keypoints3D1C;
template class Keypoints<singlekp::ValuesPerKeypoint::Three, true>;
typedef Keypoints<singlekp::ValuesPerKeypoint::Three, true> Keypoints3D3C;
}  // namespace dai