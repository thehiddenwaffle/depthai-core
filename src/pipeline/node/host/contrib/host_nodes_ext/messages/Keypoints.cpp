//
// Created by thwdpc on 7/28/25.
//

#include "messages/Keypoints.hpp"

#include <spdlog/spdlog.h>

#include <xtensor/views/xaxis_slice_iterator.hpp>
#include <xtensor/views/xview.hpp>

#include "Constants.hpp"
#include "fmt/ranges.h"
#include "pipeline/datatype/ImgAnnotations.hpp"
#include "pipeline/node/ColorCamera.hpp"
#include "utility/ErrorMacros.hpp"
#include "utility/Logging.hpp"
#include "xtensor/containers/xfixed.hpp"
#include "xtensor/misc/xpad.hpp"
#include "xtensor/reducers/xnorm.hpp"

// --- Macro: constexpr RGBA LUT for Reverse-Jet ---
// NAME: identifier of the LUT
// BINS: number of bins (e.g., 10, 64, 256)
#define DEFINE_REVERSE_JET_LUT_FLOATS(NAME, BINS)                                                  \
    constexpr std::array<std::array<float, 4>, (BINS)> NAME = []() constexpr {                     \
        static_assert(BINS > 1);                                                                   \
        std::array<std::array<float, 4>, (BINS)> lut{};                                            \
        for(std::size_t i = 0; i < (BINS); ++i) {                                                  \
            float t = (BINS) == 1 ? 0.0f : static_cast<float>(i) / static_cast<float>((BINS) - 1); \
            float v = 1.0f - t;                                                                    \
            float r = std::clamp(1.5f - std::fabs(4.0f * v - 3.0f), 0.0f, .95f);                   \
            float g = std::clamp(1.5f - std::fabs(4.0f * v - 2.0f), 0.0f, .95f);                   \
            float b = std::clamp(1.5f - std::fabs(4.0f * v - 1.0f), 0.0f, .95f);                   \
            lut[i] = {r, g, b, 1.0f};                                                              \
        }                                                                                          \
        return lut;                                                                                \
    }();

#define DEPTH_BINS 15

DEFINE_REVERSE_JET_LUT_FLOATS(RJetLUT, DEPTH_BINS);

namespace dai {

template <bool HasSpatial>
template <bool HS, std::enable_if_t<HS, int>>
Keypoints<HasSpatial>::Keypoints(std::shared_ptr<const NNData> other,
                                 xt::xtensor<float_t, 2>&& displayKeypoints,
                                 xt::xtensor<float_t, 2>&& keypoints3d,
                                 xt::xtensor<float_t, 1>&& confidenceValues,
                                 std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&& skeletonEdges)
    : confidence(std::move(xt::eval(confidenceValues))),
      keypointsSpatialXT(std::move(xt::eval(keypoints3d))),
      displayKeypointsXT(std::move(xt::eval(displayKeypoints))),
      skeletonEdges(std::move(skeletonEdges)) {
    bool allKeypointDimsEqual = displayKeypoints.shape()[0] == keypoints3d.shape()[0] && displayKeypoints.shape()[0] == confidenceValues.shape()[0];
    DAI_CHECK_V(allKeypointDimsEqual,
                "Trying to build keypoints with different dimensions: displayKeypoints: {}, keypoints3d: {}, confidenceValues: {}",
                displayKeypoints.shape()[0],
                keypoints3d.shape()[0],
                confidenceValues.shape()[0]);

    transformation = other->transformation;
    ts = other->ts;
    tsDevice = other->tsDevice;
    sequenceNum = other->sequenceNum;
}
template <bool HasSpatial>
template <bool HS, std::enable_if_t<!HS, int>>
Keypoints<HasSpatial>::Keypoints(std::shared_ptr<const NNData> other,
                                 xt::xtensor<float_t, 2>&& planarStackedKeypoints,
                                 xt::xtensor<float_t, 1>&& confidenceValues,
                                 std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&& skeletonEdges)
    : confidence(std::move(xt::eval(confidenceValues))),
      skeletonEdges(std::move(skeletonEdges)),
      displayKeypointsXT(std::move(xt::eval(planarStackedKeypoints))) {
    transformation = other->transformation;
    ts = other->ts;
    tsDevice = other->tsDevice;
    sequenceNum = other->sequenceNum;
}

// template <bool HasSpatial>
// template<bool HS, std::enable_if_t<HS, int>>
// Keypoints<HasSpatial>::Keypoints(std::shared_ptr<const NNData> other,
//                             xt::xtensor<float_t, 2>&& planarStackedKeypoints,
//                             xt::xtensor<float_t, 1>&& confidenceValues,
//                             std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&& skeletonEdges)
//     : skeletonEdges(std::move(skeletonEdges)), displayKeypointsXT(std::move(planarStackedKeypoints)) {
//     transformation = other->transformation;
//     ts = other->ts;
//     tsDevice = other->tsDevice;
//     sequenceNum = other->sequenceNum;
// }

// template <singlekp::ValuesPerKeypoint D, const bool I>
// Keypoints<D, I>::Keypoints(std::shared_ptr<const NNData> other,
//                            xt::xtensor<float_t, 2>&& planarStackedKeypoints,
//                            std::optional<xt::xtensor<float_t, 2>> keypoints3d,
//                            std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&& skeletonEdges)
//     : skeletonEdges(skeletonEdges), displayKeypointsXT(planarStackedKeypoints), keypointsSpatialXT(keypoints3d) {
//     // KP#, dim
//     const size_t numKeypoints = planarStackedKeypoints.shape()[0], numDimsFound = planarStackedKeypoints.shape()[1];
//
//     constexpr uint8_t expected = singlekp::Keypoint<D, I>::value;
//     DAI_CHECK_V(numDimsFound == expected, "Trying to build {} dimensional keypoints, got {} sets of keypoints/confidence values", expected, numDimsFound);
//
//     // Direct copy into the vec
//     assert(sizeof(singlekp::Keypoint<D, I>) == sizeof(float) * expected);
//     assert(planarStackedKeypoints.size() == numKeypoints * expected);
//
//     auto drops = D == singlekp::ValuesPerKeypoint::Two
//                      ? xt::drop(0, keypoint_y_index<singlekp::Keypoint<D, I>>())
//                      : xt::drop(0, keypoint_y_index<singlekp::Keypoint<D, I>>(), keypoint_z_index<singlekp::Keypoint<D, I>>().value());
//     // Whatever is left has to be the confidences
//     auto confidences = xt::view(keypointsSpatialXT, xt::all(), drops);
//     if(confidences.shape()[1] >= 1) {
//         distilledConfidence = xt::mean(xt::view(confidences, xt::all(), xt::keep(0, 1)), -1);
//     } else {
//         distilledConfidence = confidences;
//     }
//
//     transformation = other->transformation;
//     ts = other->ts;
//     tsDevice = other->tsDevice;
//     sequenceNum = other->sequenceNum;
// }

template <bool HasSpatial>
Keypoints<HasSpatial>::Keypoints(std::shared_ptr<Keypoints> other) : transformation(other->transformation), displayKeypointsXT(other->displayKeypointsXT), keypointsSpatialXT(other->keypointsSpatialXT), skeletonEdges(other->skeletonEdges), confidence(other->confidence) {
    ts = other->ts;
    tsDevice = other->tsDevice;
    sequenceNum = other->sequenceNum;
}

template <class EX, class EM>
auto matVecMul(const xt::xexpression<EX>& x_in, const xt::xexpression<EM>& M_in) {
    const auto& x = x_in.derived_cast();  // (N,3)
    const auto& M = M_in.derived_cast();  // (3,3)
    DAI_CHECK(x.shape()[1] == 3, "vector must be (N, 3) but is {}");
    DAI_CHECK(M.shape().size() == 2 && M.shape()[0] == 3 && M.shape()[1] == 3, "matrix must be (3, 3)");

    // Get N and prep broadcast shape
    const std::size_t N = x.shape()[0];
    std::vector<std::size_t> M_shape = {N, 3u, 3u};

    // Expand x to (N,1,3) so it can broadcast against M (N,3,3)
    auto xN13 = xt::view(x, xt::all(), xt::newaxis(), xt::all());

    // Broadcast M across the batch: (N,3,3)
    auto Mb = xt::broadcast(M, M_shape);

    // Elementwise multiply -> (N,3,3), then sum over the last axis (j) -> (N,3)
    // If you don't eval the multiplication, it segfaults with no error............................
    return xt::sum(xt::eval(Mb * xN13), {2});
}

template <bool HasSpatial>
VisualizeType Keypoints<HasSpatial>::getVisualizationMessage() const {
    auto retAnnts = std::make_shared<ImgAnnotations>();
    retAnnts->ts = this->ts;
    retAnnts->tsDevice = this->tsDevice;
    retAnnts->sequenceNum = this->sequenceNum;

    auto annotation = std::make_shared<ImgAnnotation>();
    // TODO JUST A TEST
    // annotation->texts.push_back(
    //     TextAnnotation{.position = Point2f{.5f, .5f}, .text = "test", .fontSize = 15.0, .textColor = FONT_COLOR, .backgroundColor = FONT_BACKGROUND_COLOR});
    // retAnnts->annotations.push_back(*annotation);
    // return retAnnts;

    float_t srcWidth, srcHeight;
    // Transformation else just (450+450)/300=3.0f
    std::tie(srcWidth, srcHeight) = transformation.has_value() ? transformation->getSize() : std::make_pair(450ul, 450ul);
    if(srcHeight > 1500 || srcWidth > 1500) {
        // Just assume 600 * ratio
        srcWidth = 600.0;
        srcHeight = srcWidth * (srcHeight / srcWidth);
    }
    const float_t thickness = keypointThickness(srcWidth, srcHeight);

    std::vector<Point2f> pts;
    pts.reserve(displayKeypointsXT.shape()[0]);
    const std::size_t N = displayKeypointsXT.shape()[0];
    auto xNorm = xt::flatten(xt::view(displayKeypointsXT, xt::all(), 0)) / srcWidth;
    auto yNorm = xt::flatten(xt::view(displayKeypointsXT, xt::all(), 1)) / srcHeight;
    for(std::size_t i = 0; i < N; ++i) {
        // vis seems to only work with norms
        pts.emplace_back(xNorm(i), yNorm(i));
    }

    if constexpr(HasSpatial) {
        // Throw the keypoints into some number of bins based on normalized depth
        auto zVals = xt::view(keypointsSpatialXT, xt::all(), 2);  // N
        // TODO unlock range here using min/max or some configured range from archive config.json
        const float_t zMin = 0.5f, zMax = 2.5f;
        const float_t zRange = zMax - zMin;
        const float_t zStep = zRange / DEPTH_BINS;

        constexpr float_t eps = 1e-6f;
        for(uint8_t i = 0; i < DEPTH_BINS; ++i) {
            // Split by 1/num range increments
            const float_t min = (static_cast<float_t>(i) * zStep) + zMin, max = min + zStep;
            auto mask = (zVals >= min - eps) && (zVals < max + eps);
            xt::xtensor<size_t, 1> idxsInRange = xt::ravel_indices(xt::argwhere(mask), zVals.shape());
            std::vector<Point2f> majorPtsInRange;
            std::vector<Point2f> minorPtsInRange;
            for(const size_t idx : idxsInRange) {
                // if(confidence[i] < .60) {
                //     continue;
                // }
                if(idx <= 17) {
                    majorPtsInRange.push_back(pts[idx]);
                } else {
                    minorPtsInRange.push_back(pts[idx]);
                }
            }
            const auto heatColor = dai::Color(RJetLUT[i][0], RJetLUT[i][1], RJetLUT[i][2], RJetLUT[i][3]);
            annotation->texts.push_back(TextAnnotation{.position = Point2f{.025f, .025f * static_cast<float_t>(i + 1)},
                                                       .text = fmt::format("{:.2f}-{:.2f}", min, max),
                                                       .fontSize = thickness * 2.0f,
                                                       .textColor = heatColor,
                                                       .backgroundColor = FONT_BACKGROUND_COLOR});
            if(!majorPtsInRange.empty()) {
                annotation->points.emplace_back(PointsAnnotation{.points = majorPtsInRange, KEYPOINT_COLOR, {}, {}, thickness});
                annotation->points.emplace_back(PointsAnnotation{.points = majorPtsInRange, heatColor, {}, {}, thickness * 0.75f});
            }
            if(!minorPtsInRange.empty()) {
                annotation->points.emplace_back(PointsAnnotation{.points = minorPtsInRange, KEYPOINT_COLOR, {}, {}, thickness / 3.0f});
            }
        }
    } else {
        annotation->points.emplace_back(PointsAnnotation{.points = pts, KEYPOINT_COLOR, {}, SECONDARY_COLOR, thickness});
    }

    if(skeletonEdges.has_value() && !skeletonEdges->empty()) {
        std::vector<Point2f> skeletonEdgesFlat;
        for(auto& [kp1Idx, kp2Idx] : *skeletonEdges) {
            skeletonEdgesFlat.emplace_back(pts[kp1Idx]);
            skeletonEdgesFlat.emplace_back(pts[kp1Idx]);
        }

        auto skeletonEdgesAnnt = PointsAnnotation{
            .type = PointsAnnotationType::LINE_LIST, .points = skeletonEdgesFlat, .outlineColor = PRIMARY_COLOR, .fillColor = PRIMARY_COLOR, .thickness = 1.0f};
    }

    retAnnts->annotations.push_back(*annotation);
    return retAnnts;
}

// float_t vecMagnitude(const Point3f& vec) {
//     return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
// }
//
// template <singlekp::ValuesPerKeypoint D, const bool I>
// void Keypoints<D, I>::locAndDenormAnchorErased(
//     const Point3f& anchorPoint, const ImgTransformation& transformation, float xToZScalingFactor, size_t yIndex, size_t zIndex) {
//     if(keypointsSpatialXT.has_value()) {
//         return;
//     }
//
//     // TODO expand bounding box to force root Z to be at SimCC Z mid point(keeps Z realistic to what spatial location will output)
//     const float_t anchorWorldZ = anchorPoint.z;
//     float_t bboxWidth, bboxHeight;
//     std::tie(bboxWidth, bboxHeight) = transformation.getSize();
//     float_t srcWidth, srcHeight;
//     std::tie(srcWidth, srcHeight) = transformation.getSourceSize();
//     // TODO camera skew? Doesn't seem like the dai cams have any, so that's nice
//     auto originFrameIntrinsics = transformation.getSourceIntrinsicMatrix();
//     const float_t fx = originFrameIntrinsics[0][0], cx = originFrameIntrinsics[0][2], fy = originFrameIntrinsics[1][1], cy = originFrameIntrinsics[1][2];
//
//     // Calculates the depth of the box related to the width. This finds how wide the box is in meters based on how far away vs. pixel width
//     // ReSharper disable once CppRedundantParentheses
//     float_t magnitudeBoundingBoxDepthMeters = (srcWidth * xToZScalingFactor / fx) * anchorWorldZ;
//
//     /// TEST -> it worked right
//     // xt::xtensor<double, 2> x = xt::xtensor<double, 2>({2, 3}, 4.0f);  // N=2
//     // xt::xtensor<double, 2> M = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
//     // auto y = matVecMul(x, M);     // shape (2,3)
//     // std::cout << y << std::endl;  // should print {{4,8,12}{4,8,12}} i think -> it did
//     ///
//
//     // How is this not a free conversion......
//     // xt::xtensor_fixed<float_t, xt::fixed_shape<3, 3>> transformationMatrixInv = transformation.getMatrixInv();
//     std::array<std::array<float_t, 3>, 3> roiToSrcMat = transformation.getMatrixInv();
//     xt::xtensor_fixed<float, xt::xshape<3, 3>> transformationMatrixInv{};
//     std::copy(roiToSrcMat[0].begin(), roiToSrcMat[2].end(), transformationMatrixInv.begin());
//
//     const size_t nKeypoints = displayKeypointsXT.shape()[0];
//     // Fill the 3rd component of the look direction with 1 so that scaling works right later
//     std::vector kpByOneShape = {nKeypoints};
//     xt::xtensor<float_t, 1> numKPByOneFillOne = xt::ones<float_t>(kpByOneShape);
//
//     // Denormalize UV(bbox normalized) to Upx Vpx in original frame
//     auto xPxROI = xt::view(displayKeypointsXT, xt::all(), 0) * bboxWidth;
//     auto yPxROI = xt::view(displayKeypointsXT, xt::all(), yIndex) * bboxHeight;
//
//     // Same as ImgTransformation::invTransformPoint but with broadcasting
//     xt::xtensor<float_t, 2> uvSrc = matVecMul(xt::stack(xt::xtuple(xPxROI, yPxROI, numKPByOneFillOne), 1), transformationMatrixInv);
//     auto zScale = xt::view(uvSrc, xt::all(), 2);
//     auto xPxInFullFrame = xt::view(uvSrc, xt::all(), 0) / zScale;
//     auto yPxInFullFrame = xt::view(uvSrc, xt::all(), 1) / zScale;
//
//     // This is, for each keypoint, the absolute Euclidean world distance from the camera in meters
//     auto dWorld = vecMagnitude(anchorPoint) + (xt::view(displayKeypointsXT, xt::all(), zIndex, xt::newaxis()) - 0.5f) * magnitudeBoundingBoxDepthMeters;
//
//     xt::view(displayKeypointsXT, xt::all(), xt::keep(0, yIndex)) = xt::stack(xt::xtuple(xPxInFullFrame / srcWidth, yPxInFullFrame / srcHeight), 1);
//
//     constexpr int yCorrectionForLHCS = -1;  // +Y is up in the left-hand-coordinate-system but not in pixels (higher v value is lower)
//     // Pixel "look direction" vector #KP, 3. Unitless vector that "points" with the pixel ray
//     auto lookDirection = xt::stack(xt::xtuple((xPxInFullFrame - cx) / fx, (yPxInFullFrame - cy) / fy * yCorrectionForLHCS, numKPByOneFillOne), 1);  // (#KP,
//     3) auto lookDirectionNormalized = lookDirection / xt::expand_dims(xt::norm_l2(lookDirection, {1}), 1);
//     // Scale direction vectors up and you have 3D
//     xt::view(keypointsSpatialXT, xt::all(), xt::keep(0, yIndex, zIndex)) = xt::eval(lookDirectionNormalized * dWorld);
// }

// At the bottom of Keypoints.cpp
template Keypoints<true>::Keypoints(std::shared_ptr<const NNData>,
                                    xt::xtensor<float_t, 2>&&,
                                    xt::xtensor<float_t, 2>&&,
                                    xt::xtensor<float_t, 1>&&,
                                    std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&&);
template Keypoints<false>::Keypoints(std::shared_ptr<const NNData>,
                                     xt::xtensor<float_t, 2>&&,
                                     xt::xtensor<float_t, 1>&&,
                                     std::optional<std::vector<std::pair<uint16_t, uint16_t>>>&&);

}  // namespace dai
