#include "utils.h"

#include <iostream>
#include <chrono>

static uint64_t _last_time = 0;

double current_time() 
{
    using namespace std::chrono;
    return std::chrono::duration<double>(system_clock::now().time_since_epoch()).count();
}

uint64_t get_elapsed_time() 
{
    uint64_t prev_time = _last_time;
    _last_time = static_cast<uint64_t>(1000 * current_time());
    return _last_time - prev_time; 
}

void print_elapsed_time(const std::string &source) 
{
    std::cout << "Time [" << source << "]:" << get_elapsed_time() << std::endl;
}

void print_vector(const std::vector<int>& v) {
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << ", ";
    }
    std::cout << std::endl;
}