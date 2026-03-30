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

struct NoSpatialData {};

template <const bool HasSpatial>
class Keypoints : public Buffer {
    using SpatialStorage = std::conditional_t<HasSpatial, xt::xtensor<float, 2>, NoSpatialData>;

   public:
    std::optional<ImgTransformation> transformation;

    // Return a constant reference to the contained floats. Only valid for the lifetime of this object
    template <bool HS = HasSpatial, std::enable_if_t<HS, int> = 0>
    [[nodiscard]] std::pair<const float_t*, size_t> getSpatialValues() const {
        assert(keypointsSpatialXT.size() == 3 * confidence.size());
        return std::make_pair(keypointsSpatialXT.data(), keypointsSpatialXT.size());
    }

    std::pair<const float_t*, size_t> getConfidenceValues() const {
        return std::make_pair(confidence.data(), confidence.size());
    }

    template <bool HS = HasSpatial, std::enable_if_t<HS, int> = 0>
    void toRHCS() {
        if(LHCS) {
            xt::view(keypointsSpatialXT, xt::all(), 2) *= -1;
            LHCS = false;
        }
    }


    template <bool HS = HasSpatial, std::enable_if_t<HS, int> = 0>
    void toLHCS() {
        if(!LHCS) {
            xt::view(keypointsSpatialXT, xt::all(), 2) *= -1;
            LHCS = true;
        }
    }

    template <bool HS = HasSpatial, std::enable_if_t<HS, int> = 0>
    void toMeters() {
        if(mm) {
            keypointsSpatialXT *= 0.001f;
            mm = false;
        }
    }

    template <bool HS = HasSpatial, std::enable_if_t<HS, int> = 0>
    void toMillimeters() {
        if(!mm) {
            keypointsSpatialXT *= 1000.0f;
            mm = true;
        }
    }

    // template <bool HS = HasSpatial, std::enable_if_t<HS, int> = 0>
    // [[nodiscard]] std::shared_ptr<const std::vector<float_t>> getSpatialValues() const {
    //     auto keeps = maybeZIndex.has_value() ? xt::keep(0, yIndex, maybeZIndex.value()) : xt::keep(0, yIndex);
    //     auto view = xt::view(*keypointsSpatialXT, xt::all(), keeps);
    //     auto ret = std::make_shared<std::vector<float_t>>();
    //     ret->reserve(view.size());
    //     ret->assign(view.begin(), view.end());
    //     return ret;
    // }


    template<bool HS = HasSpatial, std::enable_if_t<HS, int> = 0>
    Keypoints(std::shared_ptr<const NNData> other,
              xt::xtensor<float_t, 2>&& displayKeypoints,
              xt::xtensor<float_t, 2>&& keypoints3d,
              xt::xtensor<float_t, 1>&& confidenceValues,
              std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&& skeletonEdges);

    template <bool HS = HasSpatial, std::enable_if_t<!HS, int> = 0>
    Keypoints(std::shared_ptr<const NNData> other,
              xt::xtensor<float_t, 2>&& planarStackedKeypoints,
              xt::xtensor<float_t, 1>&& confidenceValues,
              std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&& skeletonEdges);

    Keypoints(std::shared_ptr<Keypoints> other);

    // template <typename T = singlekp::Keypoint<D, I>, std::enable_if_t<T::dimensionality == singlekp::ValuesPerKeypoint::Three, int> = 0>
    // void localizeAndDenormalizeFromAnchorPoint(const Point3f& anchorPoint, float xToZScalingFactor = 1.0f) {
    //     if(auto& transformChecked = transformation) {
    //         return locAndDenormAnchorErased(anchorPoint, transformChecked.value(), xToZScalingFactor, keypoint_y_index<T>(), keypoint_z_index<T>().value());
    //     }
    //     throw std::runtime_error("Keypoint message does not have an attached transformation, unable to localize");
    // }

    VisualizeType getVisualizationMessage() const override;

   protected:
   [[no_unique_address]] SpatialStorage keypointsSpatialXT;
    xt::xtensor<float, 2> displayKeypointsXT;

    // void locAndDenormAnchorErased(
    //     const Point3f& anchorPoint, const ImgTransformation& transformation, float xToZScalingFactor = 1.0f, size_t yIndex = 0, size_t zIndex = 0);

    std::optional<std::vector<std::pair<uint16_t, uint16_t>>> skeletonEdges = std::nullopt;

    xt::xtensor<float_t, 1> confidence;
    // whether this is in the left-hand-coordinate-system, which is depthai's default
    bool LHCS = true;
    bool mm = true;
};

template class Keypoints<false>;
typedef Keypoints<false> Keypoints2D;
template class Keypoints<true>;
typedef Keypoints<true> Keypoints3D;



}  // namespace dai