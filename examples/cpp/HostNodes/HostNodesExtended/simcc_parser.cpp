#include "ParsingNeuralNetwork.hpp"
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

int main() {
    dai::RemoteConnection remoteConnector;

    // Create device
    std::shared_ptr<dai::Device> device = std::make_shared<dai::Device>();
    device->setLogLevel(dai::LogLevel::TRACE);
    device->setLogOutputLevel(dai::LogLevel::TRACE);

    // Create pipeline
    dai::Pipeline pipeline(device);

    // Create nodes
    auto camera = pipeline.create<dai::node::Camera>()->build();

    auto archive = dai::NNArchive(dai::getModelFromZoo(dai::NNModelDescription{.model = "pedestl/rtmpose3d-open-mmlab-mmpose-large:v1-0-0:latest",
                                                                               .platform = pipeline.getDefaultDevice()->getPlatformAsString()}));

    auto nn_feed = camera->requestOutput(std::make_pair(288, 384), dai::ImgFrame::Type::BGR888i, dai::ImgResizeMode::STRETCH);

    auto trackletRepeater = pipeline.create<dai::node::Script>();
    trackletRepeater->setScript(R"(
t = Tracklets()
t.tracklets = [Tracklet()]
t.tracklets[0].spatialCoordinates = Point3f(0.0, 0.0, 1000.0)
while True:
    nn_msg = node.inputs["img"].get()
    t.setTimestamp(nn_msg.getTimestamp())
    node.outputs["t"].send(t)
    node.outputs["to_nn"].send(nn_msg)
)");
    nn_feed->link(trackletRepeater->inputs["img"]);

    auto parsedNN = pipeline.create<dai::node::ParsingNeuralNetwork>()->build(trackletRepeater->outputs["to_nn"], archive);
    // Create an output queue, note that parsed_nn->out only exists if the NN has only a single parser head
    auto& outNN = parsedNN->getOut().value().get();


    auto localizer = pipeline.create<dai::node::KeypointLocalizer>()->build<dai::singlekp::ValuesPerKeypoint::Three, true>(outNN, trackletRepeater->outputs["t"]);

    // remoteConnector.addTopic("Passthrough", *camera->requestOutput(std::make_pair(600, 400)), "img");
    // remoteConnector.addTopic("Keypoints", localizer->out, "img");
    auto q = outNN.createOutputQueue();

    // Start a pipeline
    pipeline.start();

    // Main loop
    while(pipeline.isRunning() && !quitEvent) {
        auto kp = q->get<dai::Keypoints3D3C>();
        std::cout << xt::view(kp->displayKeypointsXT, xt::range(91, xt::placeholders::_), xt::all()) << std::endl;
        {
            dai::VisualizeType var = kp->getVisualizationMessage();
            auto vis = *std::get_if<std::shared_ptr<dai::ImgAnnotations>>(&var);
            nlohmann::json j = *vis;
            std::cout << j.dump() << std::endl;
        }
        if(remoteConnector.waitKey(1) == 'q') {
            pipeline.stop();
            break;
        }
    }

    // Cleanup
    pipeline.stop();
    pipeline.wait();
}
