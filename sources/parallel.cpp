#include <iostream>
#include <cstdint>
#include <semaphore>

#include <thread>
#include <functional>
#include <queue>

#include "config.hpp"

namespace snn
{   

    void thread_worker(std::counting_semaphore<MAX_THREAD_POOL>* semp,const std::function<void(size_t)>& fun,size_t N)
    {
        try
        {
            fun(N);
            semp->release();
        }
        catch(...)
        {
            semp->release();
        }
        
    }

    /*
        Run function fun in parralel with N.
    */
    void parallel(const std::function<void(size_t)>& fun,size_t N)
    {
        if(N==0)
        {
            return;
        }

        std::counting_semaphore<MAX_THREAD_POOL> semp(MAX_THREAD_POOL);
        std::queue<std::thread> thread_queue;

        for(size_t i=0;i<N;++i)
        {
            semp.acquire();
            thread_queue.push(std::thread(thread_worker,&semp,fun,i));

            if(thread_queue.size()>MAX_THREAD_POOL)
            {
                thread_queue.front().join();
                thread_queue.pop();
            }

        }

        while(!thread_queue.empty())
        {
            std::thread& th=thread_queue.front();

            if(th.joinable())
            {
                th.join();
            }

            thread_queue.pop();
        }
               
    }
    
}