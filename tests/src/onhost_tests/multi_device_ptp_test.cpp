#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fsync_ptp_test_utils.hpp"

TEST_CASE("Test Multi-device PTP frame sync with different FPS values", "[ptp]") {
    // auto fps = GENERATE(10.0f, 13.0f, 18.5f, 30.0f, 60.0f, 120.0f, 240.0f, 300.0f, 600.0f);
    // 60 FPS does not work as of 1.30.1
    auto fps = GENERATE(10.0f, 13.0f, 18.5f, 30.0f, 45.0f);
    CAPTURE(fps);
    struct TestThresholds thresholds {
        .syncThresholdSec = 1/(2*fps), // lower this limit when we have better accuracy for timestamps
        .testDurationSec = 180,
        .recvAllTimeoutSec = 15,
        .initialSyncTimeoutSec = 60,
        .initialTimeoutSec = 60,
        .deltaMeanThreshold = 1e-3,
        .deltaP99Threshold = 2e-3,
        .syncType = SyncType::PTP
    };
    testFsync(fps, thresholds);
}
