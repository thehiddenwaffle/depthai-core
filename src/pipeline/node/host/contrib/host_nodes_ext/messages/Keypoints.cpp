//
// Created by thwdpc on 7/28/25.
//

#include "messages/Keypoints.hpp"

#include <spdlog/spdlog.h>

#include <xtensor/views/xaxis_slice_iterator.hpp>
#include <xtensor/views/xview.hpp>

#include "Constants.hpp"
#include "pipeline/datatype/ImgAnnotations.hpp"
#include "pipeline/node/ColorCamera.hpp"
#include "utility/ErrorMacros.hpp"
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
            float r = std::clamp(1.5f - std::fabs(7.0f * v - 3.0f), 0.0f, .80f);                   \
            float g = std::clamp(1.5f - std::fabs(7.0f * v - 2.0f), 0.0f, .80f);                   \
            float b = std::clamp(1.5f - std::fabs(7.0f * v - 1.0f), 0.0f, .80f);                   \
            lut[i] = {r, g, b, 1.0f};                                                              \
        }                                                                                          \
        return lut;                                                                                \
    }();

#define DEPTH_BINS 10

DEFINE_REVERSE_JET_LUT_FLOATS(RJetLUT, DEPTH_BINS);

namespace dai {
template <singlekp::ValuesPerKeypoint D, const bool I>
Keypoints<D, I>::Keypoints(std::shared_ptr<const NNData> other,
                           xt::xarray<float>&& planarStackedKeypoints,
                           std::optional<std::vector<std::pair<uint16_t, uint16_t> > >&& skeletonEdges)
    : skeletonEdges(skeletonEdges), keypointsXT(planarStackedKeypoints), displayKeypointsXT(planarStackedKeypoints) {
    // KP#, dim
    const size_t numKeypoints = planarStackedKeypoints.shape()[0], numDimsFound = planarStackedKeypoints.shape()[1];

    constexpr uint8_t expected = singlekp::Keypoint<D, I>::value;
    DAI_CHECK_V(numDimsFound == expected, "Trying to build {} dimensional keypoints, got {} sets of keypoints/confidence values", expected, numDimsFound);

    // Direct copy into the vec
    assert(sizeof(singlekp::Keypoint<D, I>) == sizeof(float) * expected);
    assert(planarStackedKeypoints.size() == numKeypoints * expected);

    auto drops = D == singlekp::ValuesPerKeypoint::Two
                     ? xt::drop(0, keypoint_y_index<singlekp::Keypoint<D, I> >())
                     : xt::drop(0, keypoint_y_index<singlekp::Keypoint<D, I> >(), keypoint_z_index<singlekp::Keypoint<D, I> >().value());
    // Whatever is left has to be the confidences
    auto confidences = xt::view(keypointsXT, xt::all(), drops);
    if(confidences.shape()[1] >= 1) {
        distilledConfidence = xt::mean(xt::view(confidences, xt::all(), xt::keep(0, 1)), -1);
    } else {
        distilledConfidence = confidences;
    }

    transformation = other->transformation;
    ts = other->ts;
    tsDevice = other->tsDevice;
    sequenceNum = other->sequenceNum;
}

template <singlekp::ValuesPerKeypoint D, const bool I>
Keypoints<D, I>::Keypoints(std::shared_ptr<Keypoints> other) {
    transformation = other->transformation;
    ts = other->ts;
    tsDevice = other->tsDevice;
    sequenceNum = other->sequenceNum;
    keypointsXT = other->keypointsXT;
    displayKeypointsXT = other->displayKeypointsXT;
    skeletonEdges = other->skeletonEdges;
    distilledConfidence = other->distilledConfidence;
}

template <class EX, class EM>
auto matVecMul(const xt::xexpression<EX>& x_in, const xt::xexpression<EM>& M_in) {
    const auto& x = x_in.derived_cast(); // (N,3)
    const auto& M = M_in.derived_cast(); // (3,3)
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

template <singlekp::ValuesPerKeypoint D, const bool I>
VisualizeType Keypoints<D, I>::getVisualizationMessage() const {
    auto retAnnts = std::make_shared<ImgAnnotations>();
    retAnnts->ts = this->ts;
    retAnnts->tsDevice = this->tsDevice;
    retAnnts->sequenceNum = this->sequenceNum;

    auto annotation = std::make_shared<ImgAnnotation>();

    const int yIdx = keypoint_y_index<singlekp::Keypoint<D, I> >();

    xt::xtensor<float_t, 2> xyKps = xt::view(displayKeypointsXT, xt::all(), xt::keep(0, yIdx)); // N, 2

    std::vector<Point2f> pts;
    pts.reserve(xyKps.shape()[0]);
    const std::size_t N = xyKps.shape()[0];
    for(std::size_t i = 0; i < N; ++i) {
        pts.emplace_back(xyKps(i, 0), xyKps(i, 1), true);
    }

    // TODO base on the src?
    float_t srcWidth, srcHeight;
    // Transformation else just (450+450)/300=3.0f
    std::tie(srcWidth, srcHeight) = transformation.has_value() ? transformation->getSourceSize() : std::make_pair(450ul, 450ul);
    if(srcHeight > 1500 || srcWidth > 1500) {
        // Just assume 600 * ratio
        srcWidth = 600.0;
        srcHeight = srcWidth * (srcHeight / srcWidth);
    }
    const float_t thickness = keypointThickness(srcWidth, srcHeight);

    if(auto zIsSome = keypoint_z_index<singlekp::Keypoint<D, I> >()) {
        // Throw the keypoints into some number of bins based on normalized depth
        auto zNormVals = xt::view(displayKeypointsXT, xt::all(), static_cast<size_t>(*zIsSome)); // N

        constexpr float_t eps = 1e-6f;
        for(uint8_t i = 0; i < DEPTH_BINS; ++i) {
            // Split by 1/num range increments
            const float_t min = static_cast<float_t>(i) / static_cast<float_t>(DEPTH_BINS),
                max = static_cast<float_t>(i + 1) / static_cast<float_t>(DEPTH_BINS);
            auto mask = (zNormVals >= min - eps) && (zNormVals < max + eps);
            xt::xtensor<size_t, 1> idxsInRange = xt::ravel_indices(xt::argwhere(mask), zNormVals.shape());
            std::vector<Point2f> majorPtsInRange;
            std::vector<Point2f> minorPtsInRange;
            for(const size_t idx : idxsInRange) {
                if(distilledConfidence[i] < .60) {
                    continue;
                }
                if(idx <= 17) {
                    majorPtsInRange.push_back(pts[idx]);
                } else {
                    minorPtsInRange.push_back(pts[idx]);
                }
            }
            const auto heatColor = dai::Color(RJetLUT[i][0], RJetLUT[i][1], RJetLUT[i][2], RJetLUT[i][3]);
            if(!majorPtsInRange.empty()) {
                annotation->points.emplace_back(PointsAnnotation{.points = majorPtsInRange, KEYPOINT_COLOR, {}, {}, thickness});
                annotation->points.emplace_back(PointsAnnotation{.points = majorPtsInRange, heatColor, {}, {}, thickness * 0.75f});
            }
            if(!minorPtsInRange.empty()) {
                annotation->points.emplace_back(PointsAnnotation{.points = minorPtsInRange, KEYPOINT_COLOR, {}, {}, thickness / 2.0f});
            }
        }

        annotation->texts.push_back(
            TextAnnotation{.position = pts[10], .text = fmt::format("{}", zNormVals(10)), .textColor = FONT_COLOR, .backgroundColor = {}});
    } else {
        annotation->points.emplace_back(PointsAnnotation{.points = pts, KEYPOINT_COLOR, {}, SECONDARY_COLOR, thickness});
    }

    if(skeletonEdges.has_value() && !skeletonEdges->empty()) {
        std::vector<Point2f> skeletonEdgesFlat;
        for(auto& [kp1Idx, kp2Idx] : *skeletonEdges) {
            skeletonEdgesFlat.emplace_back(pts[kp1Idx]);
            skeletonEdgesFlat.emplace_back(pts[kp1Idx]);
        }

        auto skeletonEdgesAnnt =
            PointsAnnotation{.type = PointsAnnotationType::LINE_LIST, .points = skeletonEdgesFlat, .outlineColor = PRIMARY_COLOR, .fillColor = PRIMARY_COLOR,
                             .thickness = 1.0f};
    }

    retAnnts->annotations.push_back(*annotation);
    return retAnnts;
}

float_t vecMagnitude(const Point3f& vec) {
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

template <singlekp::ValuesPerKeypoint D, const bool I>
void Keypoints<D, I>::locAndDenormAnchorErased(
    const Point3f& anchorPoint,
    const ImgTransformation& transformation,
    float xToZScalingFactor,
    size_t yIndex,
    size_t zIndex) {
    // TODO expand bounding box to force root Z to be at SimCC Z mid point(keeps Z realistic to what spatial location will output)
    const float_t anchorWorldZ = anchorPoint.z;
    float_t bboxWidth, bboxHeight;
    std::tie(bboxWidth, bboxHeight) = transformation.getSize();
    float_t srcWidth, srcHeight;
    std::tie(srcWidth, srcHeight) = transformation.getSourceSize();
    // TODO camera skew? Doesn't seem like the dai cams have any, so that's nice
    auto originFrameIntrinsics = transformation.getSourceIntrinsicMatrix();
    const float_t fx = originFrameIntrinsics[0][0], cx = originFrameIntrinsics[0][2], fy = originFrameIntrinsics[1][1], cy = originFrameIntrinsics[1][2];

    // Calculates the depth of the box related to the width. This finds how wide the box is in meters based on how far away vs. pixel width
    // ReSharper disable once CppRedundantParentheses
    float_t magnitudeBoundingBoxDepthMeters = (srcWidth * xToZScalingFactor / fx) * anchorWorldZ;

    /// TEST -> it worked right
    // xt::xtensor<double, 2> x = xt::xtensor<double, 2>({2, 3}, 4.0f);  // N=2
    // xt::xtensor<double, 2> M = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
    // auto y = matVecMul(x, M);     // shape (2,3)
    // std::cout << y << std::endl;  // should print {{4,8,12}{4,8,12}} i think -> it did
    ///

    // How is this not a free conversion......
    // xt::xtensor_fixed<float_t, xt::fixed_shape<3, 3>> transformationMatrixInv = transformation.getMatrixInv();
    std::array<std::array<float_t, 3>, 3> roiToSrcMat = transformation.getMatrixInv();
    xt::xtensor_fixed<float, xt::xshape<3, 3> > transformationMatrixInv{};
    std::copy(roiToSrcMat[0].begin(), roiToSrcMat[2].end(), transformationMatrixInv.begin());

    const size_t nKeypoints = keypointsXT.shape()[0];
    // Fill the 3rd component of the look direction with 1 so that scaling works right later
    std::vector kpByOneShape = {nKeypoints};
    xt::xtensor<float_t, 1> numKPByOneFillOne = xt::ones<float_t>(kpByOneShape);

    // Denormalize UV(bbox normalized) to Upx Vpx in original frame
    auto xPxROI = xt::view(displayKeypointsXT, xt::all(), 0) * bboxWidth;
    auto yPxROI = xt::view(displayKeypointsXT, xt::all(), yIndex) * bboxHeight;

    // Same as ImgTransformation::invTransformPoint but with broadcasting
    xt::xtensor<float_t, 2> uvSrc = matVecMul(xt::stack(xt::xtuple(xPxROI, yPxROI, numKPByOneFillOne), 1), transformationMatrixInv);
    auto zScale = xt::view(uvSrc, xt::all(), 2);
    auto xPxInFullFrame = xt::view(uvSrc, xt::all(), 0) / zScale;
    auto yPxInFullFrame = xt::view(uvSrc, xt::all(), 1) / zScale;

    // This is, for each keypoint, the absolute Euclidean world distance from the camera in meters
    auto dWorld = vecMagnitude(anchorPoint) + (xt::view(displayKeypointsXT, xt::all(), zIndex, xt::newaxis()) - 0.5f) * magnitudeBoundingBoxDepthMeters;

    xt::view(displayKeypointsXT, xt::all(), xt::keep(0, yIndex)) = xt::stack(xt::xtuple(xPxInFullFrame / srcWidth, yPxInFullFrame / srcHeight), 1);

    constexpr int yCorrectionForLHCS = -1; // +Y is up in the left-hand-coordinate-system but not in pixels (higher v value is lower)
    // Pixel "look direction" vector #KP, 3. Unitless vector that "points" with the pixel ray
    auto lookDirection = xt::stack(xt::xtuple((xPxInFullFrame - cx) / fx, (yPxInFullFrame - cy) / fy * yCorrectionForLHCS, numKPByOneFillOne), 1); // (#KP, 3)
    auto lookDirectionNormalized = lookDirection / xt::expand_dims(xt::norm_l2(lookDirection, {1}), 1);
    // Scale direction vectors up and you have 3D
    xt::view(keypointsXT, xt::all(), xt::keep(0, yIndex, zIndex)) = xt::eval(lookDirectionNormalized * dWorld);
}
} // namespace dai