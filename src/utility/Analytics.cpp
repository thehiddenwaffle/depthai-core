#include "depthai/utility/Analytics.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "build/version.hpp"
#include "utility/Logging.hpp"

#if defined(TARGET_DEVICE_RVC4)
    #include <XLink/XLink.h>

    #include "depthai/xlink/XLinkConstants.hpp"
#endif

#ifdef DEPTHAI_ENABLE_CURL
    #include <cpr/cpr.h>
#endif

namespace dai {
namespace utility {

namespace {

using Clock = std::chrono::system_clock;

constexpr std::size_t DEFAULT_FLUSH_AT = 20;
constexpr std::size_t DEFAULT_MAX_QUEUE_SIZE = 1000;
constexpr std::size_t DEFAULT_MAX_BATCH_SIZE = 50;
constexpr std::chrono::seconds DEFAULT_FLUSH_INTERVAL{30};
constexpr std::chrono::seconds RETRY_DELAY{5};
constexpr std::chrono::seconds MAX_RETRY_DELAY{30};
constexpr char DEFAULT_POSTHOG_HOST[] = "https://eu.i.posthog.com";
constexpr char DEFAULT_POSTHOG_API_KEY[] = "phc_navwoWmBZEUeN5UH2sFBbQJSJw6DwEUkFa8QTq9W4Mji";

std::string readEnv(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string{};
}

std::string trim(std::string value) {
    auto isSpace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](unsigned char ch) { return !isSpace(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char ch) { return !isSpace(ch); }).base(), value.end());
    return value;
}

bool analyticsEnabledByDefault() {
    auto value = trim(readEnv("DEPTHAI_ANALYTICS"));
    if(value.empty()) {
        return true;
    }

    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return !(value == "0" || value == "false" || value == "off");
}

std::filesystem::path homeDirectory() {
    auto home = readEnv("HOME");
    if(!home.empty()) {
        return home;
    }

#ifdef _WIN32
    home = readEnv("USERPROFILE");
    if(!home.empty()) {
        return home;
    }

    const auto drive = readEnv("HOMEDRIVE");
    const auto path = readEnv("HOMEPATH");
    if(!drive.empty() && !path.empty()) {
        return drive + path;
    }
#endif

    std::error_code ec;
    const auto tempDir = std::filesystem::temp_directory_path(ec);
    if(!ec) {
        return tempDir;
    }
    return std::filesystem::current_path();
}

std::filesystem::path defaultStorageDir() {
    return homeDirectory() / ".depthai" / "posthog";
}

std::string randomAlphaNumeric(std::size_t length) {
    static std::mt19937_64 generator(std::random_device{}());
    static const std::string alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static std::uniform_int_distribution<std::size_t> distribution(0, alphabet.size() - 1);

    std::string result;
    result.reserve(length);
    for(std::size_t i = 0; i < length; ++i) {
        result.push_back(alphabet[distribution(generator)]);
    }
    return result;
}

std::string generateUuidV4() {
    static std::mt19937_64 generator(std::random_device{}());
    static std::uniform_int_distribution<int> distribution(0, 255);

    std::array<std::uint8_t, 16> bytes{};
    for(auto& byte : bytes) {
        byte = static_cast<std::uint8_t>(distribution(generator));
    }

    bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0F) | 0x40);
    bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3F) | 0x80);

    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for(std::size_t index = 0; index < bytes.size(); ++index) {
        stream << std::setw(2) << static_cast<int>(bytes[index]);
        if(index == 3 || index == 5 || index == 7 || index == 9) {
            stream << '-';
        }
    }
    return stream.str();
}

std::string toIso8601(const Clock::time_point& timePoint) {
    const auto time = Clock::to_time_t(timePoint);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timePoint.time_since_epoch()) % std::chrono::seconds(1);

    std::tm utcTime{};
#ifdef _WIN32
    gmtime_s(&utcTime, &time);
#else
    gmtime_r(&time, &utcTime);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utcTime, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << milliseconds.count() << 'Z';
    return stream.str();
}

std::string makeQueueFilename() {
    const auto timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
    std::ostringstream stream;
    stream << std::setw(20) << std::setfill('0') << timestampMs << '_' << generateUuidV4() << ".json";
    return stream.str();
}

}  // namespace

#if defined(TARGET_DEVICE_RVC4)

struct AnalyticsDeviceSharedState {
    std::mutex mutex;
    streamId_t streamId{INVALID_STREAM_ID};
    bool enabled{analyticsEnabledByDefault()};

    ~AnalyticsDeviceSharedState() {
        std::lock_guard<std::mutex> lock(mutex);
        if(streamId != INVALID_STREAM_ID) {
            XLinkCloseStream(streamId);
            streamId = INVALID_STREAM_ID;
        }
    }

    void event(std::string eventName, nlohmann::json properties) {
        if(!enabled) {
            return;
        }

        eventName = trim(std::move(eventName));
        if(eventName.empty()) {
            return;
        }

        if(!properties.is_object()) {
            properties = nlohmann::json::object();
        }

        const auto payload = nlohmann::json{
            {"event", eventName},
            {"properties", std::move(properties)},
            {"timestamp", toIso8601(Clock::now())},
        };
        const auto serialized = payload.dump();

        std::lock_guard<std::mutex> lock(mutex);
        if(streamId == INVALID_STREAM_ID) {
            streamId = XLinkOpenStream(0, device::XLINK_CHANNEL_ANALYTICS, static_cast<int>(std::max<std::size_t>(serialized.size(), 4096)));
            if(streamId == INVALID_STREAM_ID) {
                return;
            }
        }

        const auto status = XLinkWriteData(streamId, reinterpret_cast<const std::uint8_t*>(serialized.data()), serialized.size());
        if(status != X_LINK_SUCCESS) {
            XLinkCloseStream(streamId);
            streamId = INVALID_STREAM_ID;
        }
    }
};

AnalyticsDeviceSharedState& analyticsSharedState() {
    static AnalyticsDeviceSharedState state;
    return state;
}

class Analytics::Impl {
   public:
    void event(std::string eventName, nlohmann::json properties) {
        analyticsSharedState().event(std::move(eventName), std::move(properties));
    }
};

#else

struct AnalyticsSharedState {
    bool enabled{false};
    std::string apiKey;
    std::string host;
    std::string distinctId;
    std::string sessionKey{randomAlphaNumeric(32)};

    std::filesystem::path queueDir;

    std::size_t flushAt{DEFAULT_FLUSH_AT};
    std::size_t maxQueueSize{DEFAULT_MAX_QUEUE_SIZE};
    std::size_t maxBatchSize{DEFAULT_MAX_BATCH_SIZE};
    std::chrono::seconds flushInterval{DEFAULT_FLUSH_INTERVAL};

    std::deque<std::string> queuedFiles;

    std::mutex mutex;
    std::condition_variable condition;
    std::thread worker;
    bool stopRequested{false};
    bool flushRequested{false};
    std::size_t retryCount{0};
    std::optional<Clock::time_point> pausedUntil;

    AnalyticsSharedState();
    ~AnalyticsSharedState();

    void event(std::string eventName, nlohmann::json properties);
    void loadQueueFromDisk();
    void workerLoop();
    void flushOneBatch();
    void removeFileFromQueueLocked(const std::string& filename);
    void deleteFilesLocked(const std::vector<std::string>& filenames);
    void deleteFileOnDisk(const std::string& filename);
};

AnalyticsSharedState& analyticsSharedState() {
    static AnalyticsSharedState state;
    return state;
}

class Analytics::Impl {
   public:
    void event(std::string eventName, nlohmann::json properties) {
        analyticsSharedState().event(std::move(eventName), std::move(properties));
    }
};

AnalyticsSharedState::AnalyticsSharedState() {
    const auto storageDir = defaultStorageDir();
    queueDir = storageDir / "queue";

    host = DEFAULT_POSTHOG_HOST;
    apiKey = DEFAULT_POSTHOG_API_KEY;
    distinctId = sessionKey;

    try {
        std::filesystem::create_directories(queueDir);
        loadQueueFromDisk();
    } catch(const std::exception& ex) {
        logger::warn("Failed to initialize analytics storage at '{}': {}", queueDir.string(), ex.what());
        return;
    }

    #ifdef DEPTHAI_ENABLE_CURL
    enabled = analyticsEnabledByDefault();
    if(!enabled) {
        logger::debug("Analytics disabled via DEPTHAI_ANALYTICS");
        return;
    }

    worker = std::thread([this]() { workerLoop(); });
    #else
    logger::debug("Analytics disabled: depthai-core was built without CURL support");
    #endif
}

AnalyticsSharedState::~AnalyticsSharedState() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopRequested = true;
        flushRequested = true;
    }
    condition.notify_all();

    if(worker.joinable()) {
        worker.join();
    }
}

void AnalyticsSharedState::event(std::string eventName, nlohmann::json properties) {
    if(!enabled) {
        return;
    }

    eventName = trim(std::move(eventName));
    if(eventName.empty()) {
        logger::debug("Skipping analytics event with an empty name");
        return;
    }

    if(!properties.is_object()) {
        logger::warn("Analytics properties for '{}' must be a JSON object. Dropping invalid properties.", eventName);
        properties = nlohmann::json::object();
    }

    std::string eventDistinctId = distinctId;
    if(properties.contains("__analytics_distinct_id")) {
        if(properties["__analytics_distinct_id"].is_string()) {
            eventDistinctId = properties["__analytics_distinct_id"].get<std::string>();
        }
        properties.erase("__analytics_distinct_id");
    }

    if(!properties.contains("$lib")) {
        properties["$lib"] = "depthai-core";
    }
    if(!properties.contains("$lib_version")) {
        properties["$lib_version"] = build::VERSION;
    }
    properties["$session_id"] = sessionKey;

    const auto payload = nlohmann::json{
        {"event", eventName},
        {"distinct_id", eventDistinctId},
        {"properties", std::move(properties)},
        {"timestamp", toIso8601(Clock::now())},
        {"uuid", generateUuidV4()},
    };

    std::string filename;
    {
        std::lock_guard<std::mutex> lock(mutex);

        if(queuedFiles.size() >= maxQueueSize && !queuedFiles.empty()) {
            const auto oldest = queuedFiles.front();
            queuedFiles.pop_front();
            deleteFileOnDisk(oldest);
            logger::debug("Analytics queue is full, dropping oldest event '{}'", oldest);
        }

        filename = makeQueueFilename();
        const auto eventPath = queueDir / filename;
        std::ofstream output(eventPath, std::ios::binary | std::ios::trunc);
        if(!output) {
            logger::warn("Failed to open analytics queue file '{}'", eventPath.string());
            return;
        }

        output << payload.dump();
        if(!output.good()) {
            output.close();
            std::error_code ec;
            std::filesystem::remove(eventPath, ec);
            logger::warn("Failed to write analytics queue file '{}'", eventPath.string());
            return;
        }

        queuedFiles.push_back(filename);
        if(queuedFiles.size() >= flushAt) {
            flushRequested = true;
        }
    }

    condition.notify_one();
}

void AnalyticsSharedState::loadQueueFromDisk() {
    std::vector<std::string> filenames;
    for(const auto& entry : std::filesystem::directory_iterator(queueDir)) {
        if(entry.is_regular_file()) {
            filenames.push_back(entry.path().filename().string());
        }
    }
    std::sort(filenames.begin(), filenames.end());

    if(filenames.size() > maxQueueSize) {
        const auto toDrop = filenames.size() - maxQueueSize;
        for(std::size_t index = 0; index < toDrop; ++index) {
            deleteFileOnDisk(filenames[index]);
        }
        filenames.erase(filenames.begin(), filenames.begin() + static_cast<std::ptrdiff_t>(toDrop));
    }

    queuedFiles.assign(filenames.begin(), filenames.end());
}

void AnalyticsSharedState::workerLoop() {
    auto nextFlushAt = Clock::now() + flushInterval;

    std::unique_lock<std::mutex> lock(mutex);
    while(!stopRequested) {
        auto wakeAt = nextFlushAt;

        condition.wait_until(lock, wakeAt, [this]() { return stopRequested || flushRequested; });
        if(stopRequested) {
            break;
        }

        const auto now = Clock::now();
        if(pausedUntil && now < *pausedUntil) {
            flushRequested = false;
            if(now >= nextFlushAt) {
                nextFlushAt = now + flushInterval;
            }
            continue;
        }

        if(!flushRequested && now < nextFlushAt) {
            continue;
        }

        flushRequested = false;
        nextFlushAt = now + flushInterval;

        lock.unlock();
        flushOneBatch();
        lock.lock();
    }
    lock.unlock();

    // Best effort flush on shutdown. This intentionally mirrors the one-shot
    // behavior of the iOS queue rather than draining the entire backlog.
    flushOneBatch();
}

void AnalyticsSharedState::flushOneBatch() {
    std::vector<std::string> batchFiles;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(queuedFiles.empty()) {
            return;
        }

        const auto batchSize = std::min<std::size_t>(maxBatchSize, queuedFiles.size());
        batchFiles.assign(queuedFiles.begin(), queuedFiles.begin() + static_cast<std::ptrdiff_t>(batchSize));
    }

    std::vector<std::string> invalidFiles;
    nlohmann::json events = nlohmann::json::array();

    for(const auto& filename : batchFiles) {
        const auto filePath = queueDir / filename;

        std::ifstream input(filePath, std::ios::binary);
        if(!input) {
            logger::warn("Analytics queue file '{}' is missing", filePath.string());
            invalidFiles.push_back(filename);
            continue;
        }

        std::stringstream buffer;
        buffer << input.rdbuf();

        try {
            const auto eventPayload = nlohmann::json::parse(buffer.str());
            if(!eventPayload.is_object()) {
                throw std::runtime_error("payload is not a JSON object");
            }
            events.push_back(eventPayload);
        } catch(const std::exception& ex) {
            logger::warn("Failed to parse analytics queue file '{}': {}", filePath.string(), ex.what());
            invalidFiles.push_back(filename);
        }
    }

    if(!invalidFiles.empty()) {
        std::lock_guard<std::mutex> lock(mutex);
        deleteFilesLocked(invalidFiles);
        batchFiles.erase(
            std::remove_if(batchFiles.begin(),
                           batchFiles.end(),
                           [&](const std::string& filename) { return std::find(invalidFiles.begin(), invalidFiles.end(), filename) != invalidFiles.end(); }),
            batchFiles.end());
    }

    if(events.empty()) {
        return;
    }

    std::vector<std::string> deletableFiles;
    bool shouldRetry = false;

    #ifdef DEPTHAI_ENABLE_CURL
    try {
        for(std::size_t index = 0; index < events.size(); ++index) {
            const auto& queuedEvent = events[index];
            auto properties = queuedEvent.value("properties", nlohmann::json::object());
            if(!properties.is_object()) {
                properties = nlohmann::json::object();
            }

            properties["distinct_id"] = queuedEvent.value("distinct_id", distinctId);

            nlohmann::json requestBody = {
                {"api_key", apiKey},
                {"event", queuedEvent.value("event", "")},
                {"properties", std::move(properties)},
            };

            if(queuedEvent.contains("timestamp")) {
                requestBody["timestamp"] = queuedEvent["timestamp"];
            }

            auto response = cpr::Post(cpr::Url{host + "/capture/"},
                                      cpr::Body{requestBody.dump()},
                                      cpr::Header{{"Content-Type", "application/json"}, {"User-Agent", std::string("depthai-core/") + build::VERSION}},
                                      cpr::Timeout{10000},
                                      cpr::VerifySsl(true));

            const auto statusCode = response.status_code;
            const bool retryable = static_cast<bool>(response.error) || (statusCode >= 300 && statusCode <= 399) || statusCode <= 0;

            if(response.error) {
                logger::debug("Analytics capture upload error {}: {}", static_cast<int>(response.error.code), response.error.message);
            } else if(statusCode < 200 || statusCode > 299) {
                logger::debug("Analytics capture upload failed with status {}: {}", statusCode, response.text);
            }

            if(retryable) {
                shouldRetry = true;
                break;
            }

            deletableFiles.push_back(batchFiles[index]);
        }
    } catch(const std::exception& ex) {
        logger::debug("Analytics capture upload threw an exception: {}", ex.what());
        shouldRetry = true;
    }
    #endif

    std::lock_guard<std::mutex> lock(mutex);
    deleteFilesLocked(deletableFiles);

    if(shouldRetry) {
        retryCount += 1;
        const auto delaySeconds =
            std::min<std::size_t>(retryCount * static_cast<std::size_t>(RETRY_DELAY.count()), static_cast<std::size_t>(MAX_RETRY_DELAY.count()));
        pausedUntil = Clock::now() + std::chrono::seconds(delaySeconds);
        logger::debug("Pausing analytics uploads for {} seconds after retryable failure", delaySeconds);
        return;
    }

    retryCount = 0;
    pausedUntil.reset();
    deleteFilesLocked(batchFiles);
}

void AnalyticsSharedState::removeFileFromQueueLocked(const std::string& filename) {
    const auto iterator = std::find(queuedFiles.begin(), queuedFiles.end(), filename);
    if(iterator != queuedFiles.end()) {
        queuedFiles.erase(iterator);
    }
}

void AnalyticsSharedState::deleteFilesLocked(const std::vector<std::string>& filenames) {
    for(const auto& filename : filenames) {
        removeFileFromQueueLocked(filename);
        deleteFileOnDisk(filename);
    }
}

void AnalyticsSharedState::deleteFileOnDisk(const std::string& filename) {
    std::error_code ec;
    std::filesystem::remove(queueDir / filename, ec);
    if(ec) {
        logger::debug("Failed to delete analytics queue file '{}': {}", filename, ec.message());
    }
}

#endif

Analytics::Analytics() : impl(std::make_unique<Impl>()) {}

Analytics::~Analytics() = default;

void Analytics::event(std::string eventName, std::map<std::string, std::string> properties) {
    nlohmann::json jsonProperties = nlohmann::json::object();
    for(auto& [key, value] : properties) {
        jsonProperties[key] = value;
    }
    event(std::move(eventName), std::move(jsonProperties));
}

void Analytics::event(std::string eventName, nlohmann::json properties) {
    if(impl) {
        impl->event(std::move(eventName), std::move(properties));
    }
}

}  // namespace utility
}  // namespace dai
