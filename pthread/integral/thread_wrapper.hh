#pragma once

#include <iostream>
#include <thread>
#include <string>
#include <functional>
#include <limits>
#include <vector>
#include <queue>
#include <mutex>
#include <future>
#include <unordered_set>

template<typename data_type>
class threads_holder final
{
    //! NOTE: pool of threads for job
    std::vector<std::thread> m_threads{};

    //! NOTE: queue of tasks to execute
    //! pair -> job & job_id
    std::queue<std::pair<std::future<data_type>, int64_t>> m_queue{};
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_flag{};

    //! NOTE: set for done job
    std::unordered_set<int64_t> m_done_jobs_id{};
    std::condition_variable m_flag_done{};
    std::mutex m_mutex_done{};

    //! NOTE: flag of threads_holder confition
    std::atomic<bool> m_quite{false};

    //! NOTE: id which will give next job
    std::atomic<int64_t> m_last_job_id{};

public:
    threads_holder(size_t nthreads, std::function<void(size_t, data_type)> &results_sum) {
        m_threads.reserve(nthreads);
        for (size_t i = 0; i < nthreads; ++i)
            m_threads.emplace_back(&threads_holder::run, this, i, std::ref(results_sum));
    }   

    ~threads_holder() {
        m_quite = true;
        for (uint32_t i = 0; i < m_threads.size(); ++i) {
            m_queue_flag.notify_all();
            m_threads[i].join();
        }
    }

public:
    void run(size_t thr_id, std::function<void(size_t, data_type)> &results_sum) {
        while (!m_quite) {
            std::unique_lock<std::mutex> lock(m_queue_mutex);

            m_queue_flag.wait(lock, [this]() -> bool
                           { return !m_queue.empty() || m_quite; });

            if (!m_queue.empty()) {
                auto elem = std::move(m_queue.front());
                m_queue.pop();
                lock.unlock();

                //! NOTE: process local result
                results_sum(thr_id, elem.first.get());

                std::lock_guard<std::mutex> lock(m_mutex_done);
                m_done_jobs_id.insert(elem.second);
                m_flag_done.notify_all();
            }
        }
    }

    template <typename Func, typename... Args>
    size_t add_job(const Func &task, Args &&...args) {
        auto job_id = m_last_job_id++;

        std::lock_guard<std::mutex> q_lock(m_queue_mutex);
        m_queue.emplace(std::async(std::launch::deferred, task, args...), job_id);

        m_queue_flag.notify_one();
        return job_id;
    }

    void wait(int64_t task_id) {
        std::unique_lock<std::mutex> lock(m_mutex_done);

        m_flag_done.wait(lock, [this, task_id]() -> bool
                           { return m_done_jobs_id.find(task_id) != m_done_jobs_id.end(); });
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        //! NOTE: waiting for notify in run function
        m_flag_done.wait(lock, [this]() -> bool
                           {
        std::lock_guard<std::mutex> task_lock(m_mutex_done);
        return m_queue.empty() && m_last_job_id == m_done_jobs_id.size(); });
    }

    bool calculated(int64_t task_id) {
        std::lock_guard<std::mutex> lock(m_mutex_done);
        if (m_done_jobs_id.find(task_id) != m_done_jobs_id.end())
            return true;

        return false;
    }
    
    size_t last_job_id() const { return m_last_job_id; }
};