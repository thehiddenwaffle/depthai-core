#include "HostNodeExtBindings.hpp"


void bind_parsingneuralnetwork(pybind11::module& m, void* pcallstack);
void bind_keypointlocalizer(pybind11::module& m, void* pcallstack);

void HostNodeExtBindings::addToCallstack(std::deque<StackFunction>& callstack) {
    callstack.push_front(&bind_parsingneuralnetwork);
    // callstack.push_front(&bind_keypointlocalizer);
}