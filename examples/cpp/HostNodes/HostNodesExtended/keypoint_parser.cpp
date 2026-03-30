#include "ParsingNeuralNetwork.hpp"
#include "absl/time/internal/cctz/include/cctz/time_zone.h"
#include "depthai/depthai.hpp"
#include "depthai/remote_connection/RemoteConnection.hpp"
#include "messages/Keypoints.hpp"
#include "postprocessors/KeypointLocalizer.hpp"
#include "xtensor/io/xio.hpp"

// Global flag for graceful shutdown
std::atomic<bool> quitEvent(false);

// Signal handler
void signalHandler(int signum) {
    quitEvent = true;
}

constexpr float FPS = 25.0f;

int main() {
    // 1. Create device and pipeline
    auto device = std::make_shared<dai::Device>();

    device->setLogLevel(dai::LogLevel::TRACE);
    device->setLogOutputLevel(dai::LogLevel::WARN);

    dai::Pipeline pipeline(device);
    if(!device->isNeuralDepthSupported()) {
        std::cout << "Exiting NeuralAssistedStereo example: device doesn't support NeuralDepth.\n";
        return 0;
    }

    dai::RemoteConnection remoteConnector{};

    // auto archive = dai::NNArchive(
    //     dai::getModelFromZoo(dai::NNModelDescription{.model = "pedestl/depth-fusion-rtmpose3d-l-adapose:rtmpose3d-l-adapose-no-lstm-384x288",
    //                                                  .platform = pipeline.getDefaultDevice()->getPlatformAsString()}));
    auto archive = dai::NNArchive(".depthai_cached_models/rtm_ada_fusion_fpdepth.rvc4.tar.xz");

    // 2. Define nodes
    auto rgb = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_A, std::nullopt, FPS);
    auto monoLeft = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_B, std::nullopt, FPS);
    auto monoRight = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_C, std::nullopt, FPS);
    auto monoRightOut = monoRight->requestFullResolutionOutput();
    auto monoLeftOut = monoLeft->requestFullResolutionOutput();

    auto neuralAssistedStereo = pipeline.create<dai::node::NeuralAssistedStereo>()->build(*monoLeftOut, *monoRightOut, dai::DeviceModelZoo::NEURAL_DEPTH_SMALL);

    auto imageAlign = pipeline.create<dai::node::ImageAlign>();
    auto rgb_feed = rgb->requestOutput(std::make_pair(1280, 960), dai::ImgFrame::Type::BGR888i, dai::ImgResizeMode::STRETCH, FPS);
    rgb_feed->link(imageAlign->inputAlignTo);
    neuralAssistedStereo->depth.link(imageAlign->input);

    auto rgbManip = pipeline.create<dai::node::ImageManip>();
    rgbManip->initialConfig->setOutputSize(288, 384, dai::ImageManipConfig::ResizeMode::STRETCH);
    rgbManip->initialConfig->setFrameType(dai::ImgFrame::Type::BGR888i);
    rgb_feed->link(rgbManip->inputImage);

    auto depthManip = pipeline.create<dai::node::ImageManip>();
    depthManip->initialConfig->setOutputSize(288, 384, dai::ImageManipConfig::ResizeMode::STRETCH);
    depthManip->initialConfig->setFrameType(dai::ImgFrame::Type::RAW16);
    imageAlign->outputAligned.link(depthManip->inputImage);

    auto calibExtractor = pipeline.create<dai::node::Script>();
    calibExtractor->setScript(R"(
import numpy as np
while True:
    img = node.io["image_in"].get()
    depth = node.io["depth"].get().getCvFrame()
    mat = img.getTransformation().getIntrinsicMatrixInv()
    fx_recip, fy_recip, neg_cx_over_fx, neg_cy_over_fy = mat[0][0], mat[1][1], mat[0][2], mat[1][2]
    nn_cal = NNData()
    nn_cal.addTensor("camera_K_inv", np.array([fx_recip, neg_cx_over_fx, fy_recip, neg_cy_over_fy], dtype=np.float16).reshape((1,2,2)), TensorInfo.DataType.FP16)
    node.io["inverse_calib_out"].send(nn_cal)
    nn_depth = NNData()
    nn_depth.addTensor("depth", depth.astype(dtype=np.float16).reshape((1, 384, 288, 1)), TensorInfo.DataType.FP16)
    node.io["depth_out"].send(nn_depth)
)");
    // auto calQueue = calibExtractor->outputs["inverse_calib_out"].createOutputQueue();
    rgbManip->out.link(calibExtractor->inputs["image_in"]);
    depthManip->out.link(calibExtractor->inputs["depth"]);

    auto parsedNN = pipeline.create<dai::node::ParsingNeuralNetwork>()->build_no_link(archive);
    dai::Node::InputMap& nnInputs = parsedNN->getInputs().value().get();
    rgbManip->out.link(nnInputs["rtm_input"]);
    calibExtractor->outputs["depth_out"].link(nnInputs["depth"]);
    calibExtractor->outputs["inverse_calib_out"].link(nnInputs["camera_K_inv"]);

    // Create an output queue, note that parsed_nn->out only exists if the NN has only a single parser head
    auto& outNN = parsedNN->getOut().value().get();

    remoteConnector.addTopic("rgb", rgbManip->out, "img");
    remoteConnector.addTopic("track", outNN, "img");

    // 6. Get output queue
    // auto disparityQueue = imageAlign->outputAligned.createOutputQueue();

    pipeline.start();
    while(pipeline.isRunning()) {
        int key = remoteConnector.waitKey(1);
        // auto alignedDepth = disparityQueue->get<dai::ImgFrame>();
        // alignedDepth->setType(dai::ImgFrame::Type::GRAYF16);
        // cv::Mat mat = alignedDepth->getCvFrame();
        // // Convert from fp16 meters to uint16 mm
        // mat *= 1000;
        // mat.convertTo(mat, CV_16UC1);
        // // cv::imshow("Depth", mat);
        // int key = cv::waitKey(1);
        if(key == 'q') {
            break;
        }
    }
    return 0;
}