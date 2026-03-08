#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

double current_time();
uint64_t get_elapsed_time();
void print_elapsed_time(const std::string& source);
void print_vector(const std::vector<int>& v);

#endif // UTILS_H