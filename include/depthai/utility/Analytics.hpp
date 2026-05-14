#pragma once

#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace dai {
namespace utility {

class Analytics {
   public:
    Analytics();
    ~Analytics();

    Analytics(const Analytics&) = delete;
    Analytics& operator=(const Analytics&) = delete;
    Analytics(Analytics&&) = delete;
    Analytics& operator=(Analytics&&) = delete;

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

   private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

std::string getTemporaryHostId();
std::string getTemporaryDeviceId(const std::string& mxid);
std::string getAnalyticsHostOS();
std::string getAnalyticsHostOSVersion();
void emitDepthaiLoadEvent();

}  // namespace utility
}  // namespace dai
