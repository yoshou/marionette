#pragma once

#include "frame.hpp"
#include <mutex>
#include <condition_variable>
#include <memory>

class frame_cursor
{
    frame_data_t frame;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::shared_ptr<frame_cursor> next;

public:
    frame_cursor(const frame_data_t& frame)
        : frame(frame)
    {}

    const frame_data_t& get_frame() const
    {
        return frame;
    }

    void set_next(const std::shared_ptr<frame_cursor>& next)
    {
        assert(next.get() != this);
        {
            std::lock_guard lock(mtx);
            this->next = next;
        }
        cv.notify_all();
    }

    std::shared_ptr<frame_cursor> get_next() const
    {
        std::lock_guard lock(mtx);
        return next;
    }

    void wait_next(std::atomic_bool& waiting)
    {
        std::unique_lock lock(mtx);
        cv.wait(lock, [&] { return !waiting || next != nullptr; });
    }
};

class frame_queue
{
public:
    size_t max_frame;
    std::deque<std::shared_ptr<frame_cursor>> frames;
    mutable std::mutex mtx;

    frame_queue(size_t max_frame)
        : max_frame(max_frame)
    {}

    void push(const frame_data_t& frame)
    {
        std::lock_guard lock(mtx);
        if (frames.size() >= max_frame)
        {
            frames.pop_front();
        }
        const auto cursor = std::make_shared<frame_cursor>(frame);
        if (frames.size() > 0)
        {
            frames.back()->set_next(cursor);
        }
        frames.push_back(cursor);
    }

    std::shared_ptr<frame_cursor> find(uint64_t frame_number) const
    {
        std::lock_guard lock(mtx);
        const auto found = std::find_if(frames.begin(), frames.end(), [frame_number](const auto& frame) { return frame->get_frame().frame_number == frame_number; });
        if (found != frames.end())
        {
            return *found;
        }
        return nullptr;
    }
};
