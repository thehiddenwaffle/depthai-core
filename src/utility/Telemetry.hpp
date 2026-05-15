#pragma once

#include <map>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace dai {
class DeviceBase;
class Pipeline;
}

namespace dai {
namespace utility {

class Telemetry {
   public:
    static Telemetry& getInstance();

    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;
    Telemetry(Telemetry&&) = delete;
    Telemetry& operator=(Telemetry&&) = delete;

    void event(std::string eventName, std::map<std::string, std::string> properties);

    template <typename T>
    void event(std::string eventName, std::map<std::string, T> properties) {
        nlohmann::json jsonProperties = nlohmann::json::object();
        for(auto& [key, value] : properties) {
            jsonProperties[key] = value;
        }
        event(std::move(eventName), std::move(jsonProperties));
    }

    void event(std::string eventName, nlohmann::json properties);

    void event(const DeviceBase& device, std::string eventName, std::map<std::string, std::string> properties);

    template <typename T>
    void event(const DeviceBase& device, std::string eventName, std::map<std::string, T> properties) {
        nlohmann::json jsonProperties = nlohmann::json::object();
        for(auto& [key, value] : properties) {
            jsonProperties[key] = value;
        }
        event(device, std::move(eventName), std::move(jsonProperties));
    }

    void event(const DeviceBase& device, std::string eventName, nlohmann::json properties);

    void event(const Pipeline& pipeline, std::string eventName, std::map<std::string, std::string> properties);

    template <typename T>
    void event(const Pipeline& pipeline, std::string eventName, std::map<std::string, T> properties) {
        nlohmann::json jsonProperties = nlohmann::json::object();
        for(auto& [key, value] : properties) {
            jsonProperties[key] = value;
        }
        event(pipeline, std::move(eventName), std::move(jsonProperties));
    }

    void event(const Pipeline& pipeline, std::string eventName, nlohmann::json properties);

   private:
    Telemetry();
    ~Telemetry();

    class Impl;
    std::unique_ptr<Impl> impl;
};

std::string getTemporaryTelemetryHostId();
std::string getTemporaryTelemetryDeviceId(const std::string& mxid);
std::string getTelemetrySessionId();
std::string getTelemetryHostOS();
std::string getTelemetryHostOSVersion();
void emitDepthaiTelemetryLoadEvent();

}  // namespace utility
}  // namespace dai
