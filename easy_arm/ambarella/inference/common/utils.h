#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/time.h>
#include <thread>
#include <mutex>
#include <future>

#define ROUND_UP_32(x) ((x)&0x1f ? (((x)&0xffffffe0) + 32) : (x))
#define LAYER_P(w) (ROUND_UP_32(4 * w))/4

unsigned long get_current_time(void);

void ListPath(std::string const &path, std::vector<std::string> &paths);

void ListImages(std::string const &path, std::vector<std::string> &images);

#endif //__UTILS_H__