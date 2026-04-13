#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fsync_ptp_test_utils.hpp"

TEST_CASE("Test Multi-device external frame sync with different FPS values", "[fsync]") {
    // auto fps = GENERATE(10.0f, 13.0f, 18.5f, 30.0f, 60.0f, 120.0f, 240.0f, 300.0f, 600.0f);
    // 60 FPS has a issue. STM looses sync on 60 FPS
    auto fps = GENERATE(10.0f, 13.0f, 18.5f, 30.0f, 45.0f);
    CAPTURE(fps);
    struct TestThresholds thresholds {
        .syncThresholdSec = 1/(2*fps), // lower this limit when we have better accuracy for timestamps
        .testDurationSec = 180,
        .recvAllTimeoutSec = 10,
        .initialSyncTimeoutSec = 4,
        .initialTimeoutSec = 0,
        .deltaMeanThreshold = 1e-3,
        .deltaP99Threshold = 2e-3,
        .syncType = SyncType::EXTERNAL
    };
    testFsync(fps, thresholds);
}
