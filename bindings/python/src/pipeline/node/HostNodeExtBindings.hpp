#pragma once

#include "pybind11_common.hpp"
#include <deque>

struct HostNodeExtBindings {
    static void addToCallstack(std::deque<StackFunction>& callstack);

};
