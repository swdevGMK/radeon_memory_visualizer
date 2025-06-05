//=============================================================================
// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
/// @author AMD Developer Tools Team
/// @file
/// @brief  Implementation for a job system to run work on multiple threads.
//=============================================================================

#include "rmt_job_system.h"
#include "rmt_platform.h"
#include "rmt_assert.h"
#include <string.h>  // for memset()

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

// Job handle type
using RmtJobHandle = uint64_t;

// Definition of the job function executed by worker threads.
using JobFunction = std::function<void(int32_t thread_id, int32_t job_index, void* input)>;

// Job: A structure representing a job to be processed by the worker threads.
// It contains the function to be executed, input data and counters for tracking status.
struct Job
{
    JobFunction          function;
    void*                input      = nullptr;
    int32_t              base_index = 0;
    int32_t              count      = 0;
    std::atomic<int32_t> run_count{0};
    std::atomic<int32_t> completed_count{0};
};

// A helper function that stops a thread and blocks until thread has completed.
void StopAndJoinThread(std::jthread& thread)
{
    if (thread.joinable())
    {
        thread.request_stop();
        thread.join();
    }
}

// JobQueue: A class that manages a pool of worker threads and a queue of jobs to be processed.
// It uses a condition variable to notify worker threads when new jobs are available and to signal when all jobs are done.
class JobQueue
{
public:
    /// Constructor for the JobQueue.
    ///
    /// @param worker_thread_count The number of worker threads to create.
    JobQueue(int32_t worker_thread_count);

    /// Destructor for the JobQueue.
    ~JobQueue();

    /// Set the adapter for the JobQueue.  Allows the JobQueue to interact with external RMT APIs.
    ///
    /// @param wrapper Pointer to the IJobQueueAdapter that will handle job management.
    void SetAdapter(IJobQueueAdapter* wrapper);

    /// Add a job to the queue.
    ///
    /// @param job A shared pointer to the Job to be added to the queue.
    void AddJob(const std::shared_ptr<Job>& job);

    /// Wait for all jobs in the queue to complete.
    void WaitForAllJobs();

    /// Shutdown the JobQueue, stopping all worker threads and clearing the job queue.
    void Shutdown();

private:
    /// Worker thread function that processes jobs from the queue.
    ///
    /// @param thread_id The ID of the worker thread.
    /// @param stop_token The stop token to check for stop requests.
    void WorkerThreadFunc(int32_t thread_id, std::stop_token stop_token);

    /// Validate a job before adding it to the queue.
    void NotifyWhenAllJobsDone();

    /// Validate a job function and count.
    /// @brief Note: The caller needs to lock the queue mutex before calling this function.
    std::shared_ptr<Job> PopJobFromQueue();

    /// Get the next job from the queue, blocking until a job is available or a stop token is signaled.
    ///
    /// @param stop_token The stop token to check for stop requests.
    std::shared_ptr<Job> GetNextJob(std::stop_token stop_token);

    /// Check if there are pending jobs in the queue.
    ///
    /// @param stop_token The stop token to check for stop requests.
    bool IsJobPending(std::stop_token stop_token) const;

private:
    std::vector<std::jthread>        workers_threads_;           ///< Vector of worker threads that will process jobs.
    std::queue<std::shared_ptr<Job>> jobs_;                      ///< Queue of jobs to be processed by the worker threads.
    std::mutex                       queue_mutex_;               ///< Mutex to protect access to the job queue.
    std::condition_variable          queue_condition_;           ///< Condition variable to notify worker threads when new jobs are available.
    std::condition_variable          all_jobs_done_condition_;   ///< Condition variable to notify when all jobs are done.
    std::atomic<bool>                terminate_flag_ = {0};      ///< Flag to indicate if the job queue is terminating.
    int32_t                          active_jobs_    = {0};      ///< Counter for active jobs being processed.
    IJobQueueAdapter*                wrapper_        = nullptr;  ///< Pointer to the JobQueue adapter that handles job management.
};

// Interface for the JobQueue Wrapper structure that handles mapping from external APIs.
// -------------------------------------------------------------------------------------
struct IJobQueueAdapter
{
public:
    virtual ~IJobQueueAdapter() = default;

    virtual std::mutex& GetHandleMutex() = 0;

    /// Add a single job.
    ///
    /// @param func The function to be executed by the job.
    /// @param input Pointer to the input data for the job.
    virtual RmtErrorCode AddSingleJob(JobFunction func, void* input, RmtJobHandle* out_handle) = 0;

    /// Add multiple jobs.
    ///
    /// @param func The function to be executed by the jobs.
    /// @param input Pointer to the input data for the jobs.
    /// @param base_index The base index for the jobs.
    /// @param count The number of jobs to add.
    /// @param out_handle Pointer to store the job handle for the first job added.
    virtual RmtErrorCode AddMultipleJobs(JobFunction func, void* input, int32_t base_index, int32_t count, RmtJobHandle* out_handle) = 0;

    /// Wait for all jobs to complete.
    virtual void WaitForAllJobs() = 0;

    /// Wait for a specific job to complete.
    ///
    /// @param handle The handle of the job to wait for.
    virtual RmtErrorCode WaitForJobCompletion(RmtJobHandle handle) = 0;

    /// Shutdown the job queue.
    virtual void Shutdown() = 0;

    /// Condition variable to notify when a job is done.
    virtual void NotifyJobDone() = 0;

    /// Condition variable to notify all threads all job are done.
    virtual void NotifyAllJobsDone() = 0;

    /// Get the next job from the queue.
    ///
    /// @param handle The handle of the job to retrieve.
    ///
    /// @returns A shared pointer to the job if found, or nullptr if not found.
    virtual std::shared_ptr<Job> GetJobByHandle(RmtJobHandle handle) = 0;
};

// JobQueue implementation: A class that manages a pool of worker threads and a queue of jobs to be processed.
// -----------------------------------------------------------------------------------------------------------

// JobQueue constructor: Initializes the job queue with a specified number of worker threads.
JobQueue::JobQueue(int32_t worker_thread_count)
{
    assert(worker_thread_count > 0);
    for (int32_t i = 0; i < worker_thread_count; ++i)
    {
        workers_threads_.emplace_back([this, i](std::stop_token stop_token) { WorkerThreadFunc(i, stop_token); });
    }
}

// JobQueue destructor: Cleans up the job queue and stops all worker threads.
JobQueue::~JobQueue()
{
    Shutdown();
}

// AddJob: Adds a job to the queue and notifies a worker thread that a job is available.
void JobQueue::AddJob(const std::shared_ptr<Job>& job)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    jobs_.push(job);
    ++active_jobs_;
    queue_condition_.notify_one();
}

// WaitForAllJobs: Blocks the calling thread until all jobs in the queue are completed.
void JobQueue::WaitForAllJobs()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    all_jobs_done_condition_.wait(lock, [this] { return jobs_.empty() && active_jobs_ == 0; });
}

// Shutdown: Stops all worker threads and clears the job queue.
void JobQueue::Shutdown()
{
    terminate_flag_ = true;
    queue_condition_.notify_all();
    std::ranges::for_each(workers_threads_, [](std::jthread& worker_thread) { StopAndJoinThread(worker_thread); });
    workers_threads_.clear();
}

// SetAdapter: Sets the adapter for the JobQueue, allowing it to interact with external RMT APIs.
void JobQueue::SetAdapter(IJobQueueAdapter* wrapper)
{
    wrapper_ = wrapper;
}

// WorkerThreadFunc: The function executed by each worker thread.
void JobQueue::WorkerThreadFunc(int32_t thread_id, std::stop_token stop_token)
{
    while (auto job = GetNextJob(stop_token))
    {
        for (int index_offset = 0; index_offset < job->count; ++index_offset)
        {
            if (stop_token.stop_requested() || terminate_flag_)
                break;

            job->function(thread_id, job->base_index + index_offset, job->input);
            if (wrapper_)
            {
                ++job->completed_count;
                wrapper_->NotifyAllJobsDone();
            }
            else
            {
                ++job->completed_count;
            }
        }
    }
}

// Isolated decision making functionality for the JobQueue.
// --------------------------------------------------------

// ValidateJob: Checks if the job function pointer is valid and the count is greater than zero.
static bool ValidateJob(const JobFunction func, const int32_t count)
{
    // Make sure the function pointer is valid and the count is more than zero.
    return (func && (count > 0)) ? true : false;
}

// IsJobPending: Checks if there are any jobs pending in the queue.
bool JobQueue::IsJobPending(std::stop_token stop_token) const
{
    // Perform the following checks (return true if all are valid, otherwise false):
    // 1. If the job queue is not empty.
    // 2. If the terminate flag is set.
    // 3. If the stop token has not requested to stop.
    return !jobs_.empty() || terminate_flag_ || stop_token.stop_requested();
}

// PopJobFromQueue: Removes a job from the queue and returns it.
// Note: The calling thread needs to hold the queue mutex lock before calling this function.
std::shared_ptr<Job> JobQueue::PopJobFromQueue()
{
    if (jobs_.empty())
    {
        return nullptr;
    }

    auto job = jobs_.front();
    jobs_.pop();
    --active_jobs_;

    RMT_ASSERT(active_jobs_ >= 0);

    // Notify waiting threads if all jobs are done.
    if (jobs_.empty() && active_jobs_ == 0)
        all_jobs_done_condition_.notify_all();

    return job;
}

// Get the next job from the queue, block until a job is available or a stop token is signaled.
std::shared_ptr<Job> JobQueue::GetNextJob(std::stop_token stop_token)
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_condition_.wait(lock, [this, &stop_token] { return IsJobPending(stop_token); });

    if (terminate_flag_ || stop_token.stop_requested())
    {
        return nullptr;
    }

    return PopJobFromQueue();
}

// Signal when all jobs are done.
void JobQueue::NotifyWhenAllJobsDone()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    all_jobs_done_condition_.wait(lock, [this] { return jobs_.empty() && active_jobs_ == 0; });
}

// Class definition for RmtJobQueueAdapter: A class that wraps the JobQueue and implements the IJobQueueAdapter interface.
class RmtJobQueueAdapter : public IJobQueueAdapter
{
public:
    /// Constructor for the RmtJobQueueAdapter class. Initializes the job queue and sets up the adapter.
    ///
    /// @param job_queue A unique pointer to the JobQueue instance to be wrapped by this adapter.
    RmtJobQueueAdapter(std::unique_ptr<JobQueue> job_queue);

    /// Destructor for the RmtJobQueueAdapter class. Cleans up the job queue and resets handles.
    ~RmtJobQueueAdapter() override;

    /// Factory method to create a new RmtJobQueueAdapter instance.
    ///
    /// @param job_queue A unique pointer to the JobQueue instance to be wrapped by this adapter.
    static IJobQueueAdapter* CreateJobQueueAdapter(std::unique_ptr<JobQueue> job_queue);

    /// Add a single job to the job queue.
    ///
    /// @param func The function to be executed by the job.
    /// @param input Pointer to the input data for the job.
    RmtErrorCode AddSingleJob(JobFunction func, void* input, RmtJobHandle* out_handle) override;

    /// Add multiple jobs to the job queue.
    ///
    /// @param func The function to be executed by the jobs.
    /// @param input Pointer to the input data for the jobs.
    /// @param base_index The base index for the jobs.
    /// @param count The number of jobs to be added.
    /// @param out_handle Pointer to store the handle of the first job added.
    ///
    /// @retval
    /// kRmtOk                                  The operation completed successfully.
    /// @retval
    /// kRmtErrorInvalidPointer 			    The operation failed because <c><i>func</i></c> was <c><i>NULL</i></c>.
    RmtErrorCode AddMultipleJobs(JobFunction func, void* input, int32_t base_index, int32_t count, RmtJobHandle* out_handle) override;

    /// Get the mutex used for synchronizing access to job handles.
    ///
    /// @returns A reference to the mutex used for synchronizing access to job handles.
    std::mutex& GetHandleMutex() override;

    /// Notify that a job has been completed.
    void NotifyJobDone() override;

    /// Notify that all jobs are done.
    void NotifyAllJobsDone() override;

    /// Wait for a specific job to complete.
    ///
    /// @param handle The handle of the job to wait for.
    ///
    /// @retval
    /// kRmtErrorInvalidPointer The operation failed because <c><i>handle</i></c> is associated with a <c><i>NULL</i></c> job pointer.
    RmtErrorCode WaitForJobCompletion(RmtJobHandle handle) override;

    /// Wait for all jobs in the queue to complete.
    void WaitForAllJobs() override;

    /// Shutdown the job queue, stopping all worker threads and clearing the job queue.
    void Shutdown() override;

    /// Get a job by its handle.
    ///
    /// @param handle The handle of the job to retrieve.
    std::shared_ptr<Job> GetJobByHandle(RmtJobHandle handle) override;

private:
    /// Create a new job handle for the given job.
    ///
    /// @param job A shared pointer to the Job for which to create a handle.
    RmtJobHandle CreateNewHandle(const std::shared_ptr<Job>& job);

    std::unique_ptr<JobQueue>                              job_queue_;                                     ///< The job queue that this adapter wraps.
    std::mutex                                             handle_mutex_;                                  ///< Mutex to protect access to job handles.
    std::condition_variable                                job_done_condition_;                            ///< Condition variable to notify when a job is done.
    static constexpr RmtJobHandle                          kStartingHandleCount_ = 1;                      ///< Starting value for job handles.
    RmtJobHandle                                           next_handle_          = kStartingHandleCount_;  ///< The next job handle to be assigned.
    std::unordered_map<RmtJobHandle, std::shared_ptr<Job>> handle_to_job_;  ///< Map to associate job handles with their corresponding jobs.
};

// RmtJobQueueAdapter constructor: Initializes the job queue and sets the adapter.
RmtJobQueueAdapter::RmtJobQueueAdapter(std::unique_ptr<JobQueue> job_queue)
    : job_queue_(std::move(job_queue))
{
    if (this->job_queue_)
    {
        this->job_queue_->SetAdapter(this);
    }
}

// Destructor to clean up the job queue and reset handles.
RmtJobQueueAdapter::~RmtJobQueueAdapter()
{
    Shutdown();
}

// Factory method to create a new RMT job queue adapter.
IJobQueueAdapter* RmtJobQueueAdapter::CreateJobQueueAdapter(std::unique_ptr<JobQueue> job_queue)
{
    return new RmtJobQueueAdapter(std::move(job_queue));
}

// AddSingleJob: Adds a single job to the job queue.
RmtErrorCode RmtJobQueueAdapter::AddSingleJob(JobFunction func, void* input, RmtJobHandle* out_handle)
{
    return AddMultipleJobs(func, input, 0, 1, out_handle);
}

// AddMultipleJobs: Adds multiple jobs to the job queue.
RmtErrorCode RmtJobQueueAdapter::AddMultipleJobs(JobFunction func, void* input, int32_t base_index, int32_t count, RmtJobHandle* out_handle)
{
    if (!ValidateJob(func, count))
    {
        return kRmtErrorInvalidPointer;
    }

    auto job        = std::make_shared<Job>();
    job->function   = func;
    job->input      = input;
    job->base_index = base_index;
    job->count      = count;

    // Generate a new handle
    RmtJobHandle handle = CreateNewHandle(job);

    // Add the job to the queue
    job_queue_->AddJob(job);
    if (out_handle)
        *out_handle = handle;
    return kRmtOk;
}

// GetHandleMutex: Returns the mutex used for synchronizing access to job handles.
std::mutex& RmtJobQueueAdapter::GetHandleMutex()
{
    return handle_mutex_;
}

// NotifyJobDone: Notifies that a job has been completed.
void RmtJobQueueAdapter::NotifyJobDone()
{
    job_done_condition_.notify_one();
}

// Condition variable to notify all job are done.
void RmtJobQueueAdapter::NotifyAllJobsDone()
{
    job_done_condition_.notify_all();
}

// WaitForJobCompletion: Waits for a specific job to complete.
RmtErrorCode RmtJobQueueAdapter::WaitForJobCompletion(RmtJobHandle handle)
{
    auto job = GetJobByHandle(handle);
    if (!job)
        return kRmtErrorInvalidPointer;

    std::unique_lock<std::mutex> lock(handle_mutex_);

    job_done_condition_.wait(lock, [&job] { return job->completed_count.load() >= job->count; });
    return kRmtOk;
}

// WaitForAllJobs: Waits for all jobs in the queue to complete.
void RmtJobQueueAdapter::WaitForAllJobs()
{
    job_queue_->WaitForAllJobs();
    std::unique_lock<std::mutex> lock(handle_mutex_);
    job_done_condition_.wait(lock, [this] { return handle_to_job_.empty(); });
}

// Shutdown: Shuts down the job queue, stopping all worker threads and clearing the job handles.
void RmtJobQueueAdapter::Shutdown()
{
    job_queue_->Shutdown();
    std::lock_guard<std::mutex> lock(handle_mutex_);
    handle_to_job_.clear();
    next_handle_ = kStartingHandleCount_;  // Reset the handle counter
}

// Get the job by handle
std::shared_ptr<Job> RmtJobQueueAdapter::GetJobByHandle(RmtJobHandle handle)
{
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto                        it = handle_to_job_.find(handle);

    if (it != handle_to_job_.end())
    {
        return it->second;
    }
    return nullptr;  // Handle not found
}

// CreateNewHandle: Generates a new job handle for the given job.
RmtJobHandle RmtJobQueueAdapter::CreateNewHandle(const std::shared_ptr<Job>& job)
{
    std::lock_guard<std::mutex> lock(handle_mutex_);
    RmtJobHandle                handle = next_handle_++;
    handle_to_job_[handle]             = job;  // Directly store the job in the map.
    return handle;
}

// Helper function to get the wrapper from the external RMT job queue structure.
inline IJobQueueAdapter* GetWrapper(RmtJobQueue* job_queue)
{
    return job_queue->wrapper;
}

// Implementation for external API Functions
// -----------------------------------------

// Initialize the job queue
RmtErrorCode RmtJobQueueInitialize(RmtJobQueue* job_queue, int32_t worker_thread_count)
{
    if (!job_queue || worker_thread_count <= 0)
        return kRmtErrorInvalidPointer;

    job_queue->wrapper = RmtJobQueueAdapter::CreateJobQueueAdapter(std::make_unique<JobQueue>(worker_thread_count));
    return kRmtOk;
}

// Shutdown the job queue
RmtErrorCode RmtJobQueueShutdown(RmtJobQueue* job_queue)
{
    if (!job_queue)
        return kRmtErrorInvalidPointer;

    // Get the adapter used to wrap the RmtJobQueue.
    auto* wrapper = GetWrapper(job_queue);

    wrapper->Shutdown();
    delete wrapper;

    return kRmtOk;
}

// Add a single job
RmtErrorCode RmtJobQueueAddSingle(RmtJobQueue* job_queue, RmtJobFunction func, void* input, RmtJobHandle* out_handle)
{
    return RmtJobQueueAddMultiple(job_queue, func, input, 0, 1, out_handle);
}

// Add multiple jobs
RmtErrorCode RmtJobQueueAddMultiple(RmtJobQueue* job_queue, RmtJobFunction func, void* input, int32_t base_index, int32_t count, RmtJobHandle* out_handle)
{
    if (!job_queue || !func || count <= 0)
        return kRmtErrorInvalidPointer;

    // Wrap the job queue function pointer in a std::function
    JobFunction job_func = [func](int32_t thread_id, int32_t job_index, void* input) { func(thread_id, job_index, input); };

    // Get the adapter used to wrap the RmtJobQueue.
    auto* wrapper = GetWrapper(job_queue);

    // Have the adapter create the job object
    return wrapper->AddMultipleJobs(job_func, input, base_index, count, out_handle);
}

// Wait for a job to complete
RmtErrorCode RmtJobQueueWaitForCompletion(RmtJobQueue* job_queue, RmtJobHandle handle)
{
    if (!job_queue)
        return kRmtErrorInvalidPointer;

    // Get the adapter used to wrap the RmtJobQueue.
    auto* wrapper = GetWrapper(job_queue);

    // Wait for the job to complete.
    RmtErrorCode result = wrapper->WaitForJobCompletion(handle);

    if (result != kRmtOk)
        return result;
    return kRmtOk;
}
