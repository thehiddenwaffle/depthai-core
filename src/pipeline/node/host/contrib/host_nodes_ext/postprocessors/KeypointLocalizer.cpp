#include "postprocessors/KeypointLocalizer.hpp"

#include <spdlog/spdlog.h>

#include <messages/Keypoints.hpp>
#include <pipeline/datatype/SpatialImgDetections.hpp>
#include <pipeline/datatype/Tracklets.hpp>
#include <pipeline/node/ObjectTracker.hpp>
#include <pipeline/node/SpatialDetectionNetwork.hpp>
#include <utility/ErrorMacros.hpp>

#include "pipeline/ThreadedNodeImpl.hpp"
#include "utility/Logging.hpp"

namespace dai::node {

template std::shared_ptr<KeypointLocalizer> KeypointLocalizer::build<singlekp::ValuesPerKeypoint::Three, true>(Output& incomingKeypoints,
                                                                                                               const std::shared_ptr<ObjectTracker>& t);
template std::shared_ptr<KeypointLocalizer> KeypointLocalizer::build<singlekp::ValuesPerKeypoint::Three, true>(
    Output& incomingKeypoints, const std::shared_ptr<SpatialDetectionNetwork>& t);
template std::shared_ptr<KeypointLocalizer> KeypointLocalizer::build<singlekp::ValuesPerKeypoint::Three, true>(Output& incomingKeypoints,
                                                                                                               Output& splitSingularTracklets);

template <singlekp::ValuesPerKeypoint D, const bool I, std::enable_if_t<D == singlekp::ValuesPerKeypoint::Three, int>>
std::shared_ptr<KeypointLocalizer> KeypointLocalizer::build(Output& incomingKeypoints, const std::shared_ptr<ObjectTracker>& incomingTracklets) {
    // TODO message syncing and state for 1 to N where N is the number of ROI's in the tracker
    return sharedBuildParts(
        incomingKeypoints, incomingTracklets->out, SpatialExtractorFunc<D, I>([kp = kpInName, tl = tlInName](const std::shared_ptr<MessageGroup>& in) {
            auto kpMsg = in->get<Keypoints<D, I>>(kp);
            auto tracklets = in->get<Tracklets>(tl);
            DAI_CHECK(kpMsg && tracklets, "One or more received messages were unable to be cast into expected type");
            DAI_CHECK(!tracklets->tracklets.empty(), "Empty tracklet message")
            Tracklet first = tracklets->tracklets.front();
            return std::make_tuple(first.spatialCoordinates, kpMsg);
        }));
}

template <singlekp::ValuesPerKeypoint D, const bool I, std::enable_if_t<D == singlekp::ValuesPerKeypoint::Three, int>>
std::shared_ptr<KeypointLocalizer> KeypointLocalizer::build(Output& incomingKeypoints, const std::shared_ptr<SpatialDetectionNetwork>& incomingDetections) {
    // TODO message syncing and state for 1 to N where N is the number of ROI's in the det net
    return sharedBuildParts(
        incomingKeypoints, incomingDetections->out, SpatialExtractorFunc<D, I>([kp = kpInName, tl = tlInName](const std::shared_ptr<MessageGroup>& in) {
            auto kpMsg = in->get<Keypoints<D, I>>(kp);
            auto detections = in->get<SpatialImgDetections>(tl);
            DAI_CHECK(kpMsg && detections, "One or more received messages were unable to be cast into expected type");
            DAI_CHECK(!detections->detections.empty(), "Empty detection message");
            // TODO internal matching?
            SpatialImgDetection first = detections->detections.front();
            return std::make_tuple(first.spatialCoordinates, kpMsg);
        }));
}

template <singlekp::ValuesPerKeypoint D, const bool I, std::enable_if_t<D == singlekp::ValuesPerKeypoint::Three, int>>
std::shared_ptr<KeypointLocalizer> KeypointLocalizer::build(Output& incomingKeypoints, Output& splitSingularTracklets) {
    return sharedBuildParts(
        incomingKeypoints, splitSingularTracklets, SpatialExtractorFunc<D, I>([kp = kpInName, tl = tlInName](const std::shared_ptr<MessageGroup>& in) {
            auto kpMsg = in->get<Keypoints<D, I>>(kp);
            auto tracklets = in->get<Tracklets>(tl);
            DAI_CHECK(kpMsg && tracklets, "One or more received messages were unable to be cast into expected type");
            DAI_CHECK(!tracklets->tracklets.empty(), "Empty tracklet message")
            if(tracklets->tracklets.size() > 1) {
                logger::warn("KeypointLocalizer was built to process split singular tracklets, received {} tracklets, ignoring all but the first",
                             tracklets->tracklets.size());
            }
            Tracklet first = tracklets->tracklets.front();
            return std::make_tuple(first.spatialCoordinates, kpMsg);
        }));
}

std::shared_ptr<Buffer> KeypointLocalizer::processGroup(std::shared_ptr<MessageGroup> in) {
    const auto t1 = std::chrono::high_resolution_clock::now();
    DAI_CHECK(processFn.has_value(), "internal processing function missing, was any build(...) method called?");
    DAI_CHECK_IN(in != nullptr);
    const uint8_t found = in->group.count(kpInName) + in->group.count(tlInName);
    DAI_CHECK_IN(found <= 2);  // This is violated only if MessageGroup was refactored to use multimap, in which case everything here needs to change
    DAI_CHECK_V(found == 2, "%u missing messages in message group, was the sync modified manually?", 2 - found);
    auto msg = (*processFn)(in);
    DAI_CHECK_IN(msg);
    const auto t2 = std::chrono::high_resolution_clock::now();
    pimpl->logger->trace("KeypointLocalizer: Time taken: {} ms", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000);
    return msg;
}

}  // namespace dai::node
