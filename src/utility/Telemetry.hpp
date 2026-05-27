#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace dai {
class DeviceBase;
class Pipeline;
}  // namespace dai

namespace dai {
namespace utility {

class Telemetry {
   public:
    static Telemetry& getInstance();
    static std::string getTemporaryTelemetryHostId();
    static std::string getTemporaryTelemetryDeviceId(const std::string& mxid);
    static std::string getTelemetrySessionId();
    static std::string getTelemetryHostOS();
    static std::string getTelemetryHostOSVersion();
    static bool isTelemetryEnabled();
    static void setTelemetryUsesPython(bool value);
    static void emitDepthaiTelemetryLoadEvent();

    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;
    Telemetry(Telemetry&&) = delete;
    Telemetry& operator=(Telemetry&&) = delete;

    /**
     * Always prefix depthai only events with depthai_
     * Don't prefix global events (shared across software stack)
     */
    void event(std::string eventName, nlohmann::json properties);

    void event(const DeviceBase& device, std::string eventName, nlohmann::json properties);

    void event(const Pipeline& pipeline, std::string eventName, nlohmann::json properties);

   private:
    Telemetry();
    ~Telemetry();

    class Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace utility
}  // namespace dai
