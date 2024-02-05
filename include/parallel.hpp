#pragma once

#include <cstdint>
#include <semaphore>

#include <thread>
#include <functional>
#include <queue>

#include "config.hpp"

namespace snn
{   

    void thread_worker(std::counting_semaphore<MAX_THREAD_POOL>* semp,const std::function<void(size_t)>& fun,size_t N);
    
    /*
        Run function fun in parralel with N.
    */
    void parallel(const std::function<void(size_t)>& fun,size_t N);
    
}