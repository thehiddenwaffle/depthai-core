#pragma once
#include <xtensor/views/xview.hpp>

#include "depthai/pipeline/datatype/ImgAnnotations.hpp"
#include "depthai/pipeline/datatype/SpatialImgDetections.hpp"
#include "depthai/pipeline/datatype/Tracklets.hpp"
#include "depthai/pipeline/node/ObjectTracker.hpp"
#include "depthai/pipeline/node/SpatialDetectionNetwork.hpp"
#include "depthai/pipeline/node/host/HostNode.hpp"
#include "messages/Keypoints.hpp"
#include "xtensor/io/xio.hpp"

// template <typename T>
// using is_spatial = std::integral_constant<bool, std::is_same<T, dai::SpatialImgDetections>::value || std::is_same<T, dai::Tracklets>::value>;
//
// // Function type to get the required pieces for spatial manipulation without caring what the message was (Tracklets/SpatialDetections)
// template <dai::singlekp::ValuesPerKeypoint D, const bool I>
// using SpatialExtractorFunc = std::function<std::tuple<dai::Point3f, std::shared_ptr<dai::Keypoints<D, I>>>(std::shared_ptr<dai::MessageGroup>)>;
//
// namespace dai::node {
//
// class KeypointLocalizer : public CustomNode<KeypointLocalizer> {
//    public:
//     ~KeypointLocalizer() override = default;
//
//     // TODO it's unlikely that the keypoint network and the spatial are the same, so some state tracker would be nice to take for example
//     //  1 SpatialDetections package with 5 detections and then map them to the 5 incoming keypoint buffers, note that we can't use sync for this(1!=5)
//     Input& kpInput = inputs[kpInName];
//     Input& spatialInput = inputs[tlInName];
//
//     // Enable only when KP is a Keypoints specialization or derives from it
//     template <singlekp::ValuesPerKeypoint D, const bool I, std::enable_if_t<D == singlekp::ValuesPerKeypoint::Three, int> = 0>
//     std::shared_ptr<KeypointLocalizer> build(Output& incomingKeypoints, const std::shared_ptr<ObjectTracker>& incomingTracklets);
//
//     // Enable only when KP is a Keypoints specialization or derives from it
//     template <singlekp::ValuesPerKeypoint D, const bool I, std::enable_if_t<D == singlekp::ValuesPerKeypoint::Three, int> = 0>
//     std::shared_ptr<KeypointLocalizer> build(Output& incomingKeypoints, const std::shared_ptr<SpatialDetectionNetwork>& incomingDetections);
//
//     template <singlekp::ValuesPerKeypoint D, const bool I, std::enable_if_t<D == singlekp::ValuesPerKeypoint::Three, int> = 0>
//     std::shared_ptr<KeypointLocalizer> build(Output& incomingKeypoints, Output& splitSingularTracklets);
//
//     std::shared_ptr<Buffer> processGroup(std::shared_ptr<MessageGroup> in) override;
//
//    private:
//     template <singlekp::ValuesPerKeypoint D, const bool I, std::enable_if_t<D == singlekp::ValuesPerKeypoint::Three, int> = 0>
//     std::shared_ptr<KeypointLocalizer> sharedBuildParts(Output& kp, Output& tl, SpatialExtractorFunc<D, I> genericExtractor) {
//         kp.link(kpInput);
//         tl.link(spatialInput);
//         // sync->setSyncAttempts(0);
//         dynamic_cast<Sync&>(kpInput.getParent()).setSyncAttempts(0);
//         out.setName("KpLocalizerOut");
//         // Install a type-erased processor that knows KPMsg at compile time
//         processFn = [genericExtractor, this](std::shared_ptr<MessageGroup>& in) {
//             auto [anchorPoint, keypointsMsg] = genericExtractor(in);
//
//             // auto po = xt::print_options::precision(4);
//             // std::ostringstream keypointsXT_ss;
//             // keypointsXT_ss << po << xt::eval(xt::view(keypointsMsg->keypointsXT, xt::range(0, 18), xt::all()));
//             // std::ostringstream displayKeypointsXT_ss;
//             // displayKeypointsXT_ss << po << xt::eval(xt::view(keypointsMsg->displayKeypointsXT, xt::range(0, 18), xt::all()));
//             // pimpl->logger->trace("Start localizing:\nkeypointsXT: {}\ndisplayKeypointsXT: {}", keypointsXT_ss.str(), displayKeypointsXT_ss.str());
//
//             //{
//             //    VisualizeType var = keypointsMsg->getVisualizationMessage();
//             //    auto vis = *std::get_if<std::shared_ptr<ImgAnnotations>>(&var);
//             //    nlohmann::json j = *vis;
//             //    pimpl->logger->trace("Done localizing, display: \n{}", j.dump());
//             //}
//
//             auto kpMsgCopy = std::make_shared<Keypoints<D, I>>(keypointsMsg);
//             kpMsgCopy->localizeAndDenormalizeFromAnchorPoint(anchorPoint);
//
//             //{
//             //    VisualizeType var = kpMsgCopy->getVisualizationMessage();
//             //    auto vis = *std::get_if<std::shared_ptr<ImgAnnotations>>(&var);
//             //    nlohmann::json j = *vis;
//             //    pimpl->logger->trace("Done localizing, display: \n{}", j.dump());
//             //}
//
//             return kpMsgCopy;
//         };
//         return std::static_pointer_cast<KeypointLocalizer>(shared_from_this());
//     }
//
//     std::optional<std::function<std::shared_ptr<Buffer>(std::shared_ptr<MessageGroup>&)>> processFn = std::nullopt;
//     static constexpr const char* kpInName = "kp";
//     static constexpr const char* tlInName = "tl";
// };
// }  // namespace dai::node