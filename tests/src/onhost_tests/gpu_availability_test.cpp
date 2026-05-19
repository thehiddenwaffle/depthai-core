#include <catch2/catch_all.hpp>

#include <cctype>
#include <string>

#include "depthai/device/Device.hpp"
#include "depthai/common/EepromData.hpp"
#include "depthai/device/Platform.hpp"

namespace {

int parseBoardRevisionNumber(const std::string& boardRev) {
    std::size_t pos = boardRev.find_first_of("Rr");
    if(pos == std::string::npos) pos = 0;
    else pos += 1;

    while(pos < boardRev.size() && !std::isdigit(static_cast<unsigned char>(boardRev[pos]))) {
        pos++;
    }
    if(pos >= boardRev.size()) return 0;

    int value = 0;
    while(pos < boardRev.size() && std::isdigit(static_cast<unsigned char>(boardRev[pos]))) {
        value = value * 10 + (boardRev[pos] - '0');
        pos++;
    }
    return value;
}

bool expectedHasGPU(dai::Device& device) {
    if(device.getPlatform() != dai::Platform::RVC4) return false;

    const auto product = device.getProductName();  // Uppercase + hyphenated
    const bool isOak4D = product.rfind("OAK4-D", 0) == 0;
    const bool isOak4S = product.rfind("OAK4-S", 0) == 0;
    if(!isOak4D && !isOak4S) return false;

    const auto eepromFactory = device.readFactoryCalibration().getEepromData();
    const auto eeprom = device.readCalibration().getEepromData();
    const std::string& boardRev = !eepromFactory.boardRev.empty() ? eepromFactory.boardRev : eeprom.boardRev;
    return parseBoardRevisionNumber(boardRev) >= 9;
}

}  // namespace

TEST_CASE("Device.hasGPU matches expected policy", "[onhost]") {
    dai::Device device;
    const bool expected = expectedHasGPU(device);
    const bool actual = device.hasGPU();

    REQUIRE(actual == expected);
}
