/**
 * Minimal GPUStereo example — disparity from a stereo camera pair (RVC4 only).
 */
#include <depthai/depthai.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    dai::Device device;

    if(!device.isGpuAvailable()) {
        std::cout << "Exiting GPUStereo example: GPU not available on this device.\n";
        return 0;
    }

    dai::Pipeline pipeline(device);
    auto camLeft = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_B);
    auto camRight = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_C);

    auto gpu = pipeline.create<dai::node::GPUStereo>();
    camLeft->requestOutput({1280, 800}, dai::ImgFrame::Type::GRAY8, 30)->link(gpu->left);
    camRight->requestOutput({1280, 800}, dai::ImgFrame::Type::GRAY8, 30)->link(gpu->right);
    gpu->setRectification(true);
    gpu->setConfidenceThreshold(25);

    auto dispQ = gpu->disparity.createOutputQueue();

    pipeline.start();
    while(pipeline.isRunning()) {
        auto frame = dispQ->get<dai::ImgFrame>();
        cv::Mat disp(frame->getHeight(), frame->getWidth(), CV_16UC1, frame->getData().data());
        cv::Mat dispF;
        disp.convertTo(dispF, CV_32F);
        cv::Mat disp8;
        cv::normalize(dispF, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat dispColor;
        cv::applyColorMap(disp8, dispColor, cv::COLORMAP_JET);
        cv::imshow("GPUStereo Disparity", dispColor);
        if(cv::waitKey(1) == 'q') break;
    }

    return 0;
}
