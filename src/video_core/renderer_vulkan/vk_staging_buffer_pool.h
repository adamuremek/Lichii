// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_policy.h"

#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

struct StagingBufferRef {
    VkBuffer buffer;
    VkDeviceAddress device_address;
    VkDeviceSize offset;
    std::span<u8> mapped_span;
    MemoryUsage usage;
    u32 size_class;
    u64 index;
};

using LargeUploadStaging =
    boost::container::static_vector<StagingBufferRef, StagingBufferPolicy::UploadArenaBlockCount + 1>;

class StagingBufferPool {
public:
    static constexpr size_t NUM_SYNCS = 16;

    explicit StagingBufferPool(const Device& device, MemoryAllocator& memory_allocator,
                               Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferRef Request(size_t size, MemoryUsage usage, bool deferred = false);
    LargeUploadStaging TryLargeUpload(size_t size);
    void FreeDeferred(StagingBufferRef& ref);

    [[nodiscard]] VkBuffer StreamBuf() const noexcept {
        return *stream_buffer;
    }

    void TickFrame();

    /// Bytes of the cached staging buffers of a kind, idle or in use (EDEN_NX2_MEM_TRACE).
    [[nodiscard]] u64 CachedBytes(MemoryUsage usage);

private:
    struct StreamBufferCommit {
        size_t upper_bound;
        u64 tick;
    };

    struct StagingBuffer {
        vk::Buffer buffer;
        VkDeviceAddress device_address;
        std::span<u8> mapped_span;
        MemoryUsage usage;
        u32 size_class;
        u64 index;
        u64 tick = 0;
        u64 last_use_frame = 0;
        bool deferred{};

        StagingBufferRef Ref() const noexcept {
            return {
                .buffer = *buffer,
                .device_address = device_address,
                .offset = 0,
                .mapped_span = mapped_span,
                .usage = usage,
                .size_class = size_class,
                .index = index,
            };
        }
    };

    struct StagingBuffers {
        std::vector<StagingBuffer> entries;
        size_t delete_index = 0;
        size_t iterate_index = 0;
    };

    struct UploadArenaBlock {
        vk::Buffer buffer;
        VkDeviceAddress device_address{};
        StagingBufferPolicy::UploadArenaCursor cursor;

        StagingBufferRef Slice(size_t offset, size_t size) {
            return {
                .buffer = *buffer,
                .device_address = device_address,
                .offset = offset,
                .mapped_span = buffer.Mapped().subspan(offset, size),
                .usage = MemoryUsage::Upload,
                .size_class{},
                .index{},
            };
        }
    };

    using StagingBuffersCache = std::array<StagingBuffers, StagingBufferPolicy::NumClasses>;

    StagingBufferRef GetStreamBuffer(size_t size);

    bool AreRegionsActive(size_t region_begin, size_t region_end) const;

    StagingBufferRef GetStagingBuffer(size_t size, MemoryUsage usage, bool deferred = false);

    std::optional<StagingBufferRef> TryGetUploadSlice(size_t size);
    bool AddUploadArenaBlock(size_t capacity);

    std::optional<StagingBufferRef> TryGetReservedBuffer(u32 size_class, MemoryUsage usage,
                                                         bool deferred);

    StagingBufferRef CreateStagingBuffer(u32 size_class, MemoryUsage usage, bool deferred);

    vk::Buffer CreateBuffer(VkDeviceSize size, MemoryUsage usage);
    [[nodiscard]] VkDeviceAddress GetBufferAddress(VkBuffer buffer) const;

    StagingBuffersCache& GetCache(MemoryUsage usage);

    void ReleaseCache(MemoryUsage usage);

    void ReleaseLevel(StagingBuffersCache& cache, size_t log2);
    void ReleaseInactive(StagingBuffersCache& cache, size_t& deletions_left);
    size_t Region(size_t iter) const noexcept {
        return iter / region_size;
    }

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    const bool reuse_staging;
    const bool upload_arena_enabled;
    const size_t upload_arena_alignment;
    bool upload_arena_can_grow = true;
    std::vector<UploadArenaBlock> upload_arena;

    vk::Buffer stream_buffer;
    VkDeviceAddress stream_buffer_address{};
    std::span<u8> stream_pointer;
    VkDeviceSize stream_buffer_size;
    VkDeviceSize region_size;

    size_t iterator = 0;
    size_t used_iterator = 0;
    size_t free_iterator = 0;
    std::array<u64, NUM_SYNCS> sync_ticks{};

    StagingBuffersCache device_local_cache;
    StagingBuffersCache upload_cache;
    StagingBuffersCache download_cache;

    size_t current_delete_level = 0;
    u64 frame = 0;
    u64 buffer_index = 0;
    u64 unique_ids{};
};

} // namespace Vulkan
