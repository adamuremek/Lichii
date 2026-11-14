// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/ranges.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
namespace {

using namespace Common::Literals;

// Maximum potential alignment of a Vulkan buffer
constexpr VkDeviceSize MAX_ALIGNMENT = 256;

// Stream buffer size in bytes
// *NIX drivers are more sensitive to increased buffers for streaming.
// Windows ones however, can intake bigger buffers and generally do not OOM.
// - GTX 960 on Windows will not OOM with 256mib
// - GT 1030 on ^NIX will OOM with 256mib
#if defined(__FreeBSD__)
constexpr VkDeviceSize MAX_STREAM_BUFFER_SIZE = 128_MiB;
#else
// 128 MiB on Windows too: 256 MiB sat mostly unused in device memory on Switch 2 titles.
constexpr VkDeviceSize MAX_STREAM_BUFFER_SIZE = 128_MiB;
#endif

// Unused space after the end of the stream buffer. Constant buffers are bound from it and the
// shaders declare each one at its full 64 KiB, so Metal checks 64 KiB from the binding offset
// against the end of the buffer ("argument c3[0] ... has space for 26624 bytes, but argument has a
// length(65536)"). The tail is never handed out, so every binding has the full 64 KiB behind it.
constexpr VkDeviceSize STREAM_BUFFER_TAIL = 64_KiB;

bool WarmUploadArenaRequested() {
    const char* value = std::getenv("EDEN_NX2_WARM_UPLOAD_ARENA");
    return value && std::string_view{value} == "1";
}

size_t GetStreamBufferSize(const Device& device) {
    if (!device.HasDebuggingToolAttached()) {
        if (device.IsIntegrated()) {
            // An integrated GPU's device-local host-visible heap can be no bigger than the stream
            // buffer (Radeon 780M: 256 MiB, 243 MiB budget), which then sat over that budget.
            // Leave a quarter of it free.
            VkDeviceSize heap_size = 0;
            ForEachDeviceLocalHostVisibleHeap(
                device, [&heap_size](size_t, VkMemoryHeap& heap) {
                    heap_size = (std::max)(heap_size, heap.size);
                });
            if (heap_size != 0 && heap_size <= MAX_STREAM_BUFFER_SIZE) {
                return Common::AlignUp(heap_size * 3 / 4, MAX_ALIGNMENT);
            }
        }
        return MAX_STREAM_BUFFER_SIZE;
    }

    VkDeviceSize size{0};
    bool has_device_local_host_visible_heap{};
    ForEachDeviceLocalHostVisibleHeap(device, [&size, &has_device_local_host_visible_heap](
                                                  size_t index, VkMemoryHeap& heap) {
        has_device_local_host_visible_heap = true;
        size = (std::max)(size, heap.size);
    });
    if (has_device_local_host_visible_heap) {
        // If rebar is not supported, cut the max heap size to 40%. This will allow 2 captures to be
        // loaded at the same time in RenderDoc. If rebar is supported, this shouldn't be an issue
        // as the heap will be much larger.
        if (size <= MAX_STREAM_BUFFER_SIZE) {
            size = size * 40 / 100;
        }
    } else {
        size = MAX_STREAM_BUFFER_SIZE;
    }
    return (std::min)(Common::AlignUp(size, MAX_ALIGNMENT), MAX_STREAM_BUFFER_SIZE);
}
} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Device& device_, MemoryAllocator& memory_allocator_,
                                     Scheduler& scheduler_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      reuse_staging{[] {
          const char* value = std::getenv("EDEN_NX2_STAGING_REUSE");
          return value && std::string_view{value} == "1";
      }()},
      upload_arena_enabled{[] {
          const char* value = std::getenv("EDEN_NX2_UPLOAD_ARENA");
          return WarmUploadArenaRequested() || (value && std::string_view{value} == "1");
      }()},
      upload_arena_alignment{3 * (std::max)({MAX_ALIGNMENT, device.GetStorageBufferAlignment(),
                                             device.GetUniformBufferAlignment()})},
      stream_buffer_size{GetStreamBufferSize(device)}, region_size{stream_buffer_size /
                                                                   StagingBufferPool::NUM_SYNCS} {
    VkBufferCreateInfo stream_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = stream_buffer_size + STREAM_BUFFER_TAIL,
        // Transfer destination too: GPU-written list words are copied over their placeholders
        // in the constant buffers streamed here (Nx2ListSlots::PatchUniform).
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    stream_buffer = memory_allocator.CreateBuffer(stream_ci, MemoryUsage::Stream);
    if (device.HasDebuggingToolAttached()) {
        stream_buffer.SetObjectNameEXT("Stream Buffer");
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_buffer_address = device.GetLogical().GetBufferDeviceAddress(*stream_buffer);
    }
    stream_pointer = stream_buffer.Mapped();
    ASSERT_MSG(!stream_pointer.empty(), "Stream buffer must be host visible!");

    if (WarmUploadArenaRequested() && !device.IsIntegrated() &&
        Common::GetMemInfo().TotalPhysicalMemory >= 16_GiB) {
        // Allocate on startup so a scene transition can reuse mapped host buffers.
        for (size_t i = 0; i < StagingBufferPolicy::UploadArenaBlockCount; ++i) {
            constexpr size_t capacity = StagingBufferPolicy::WarmUploadArenaBlockSize;
            const auto commit_room = Common::GetSystemCommitRoom();
            if ((commit_room && *commit_room < 2_GiB + capacity) || !AddUploadArenaBlock(capacity)) {
                break;
            }
        }
        LOG_INFO(Render_Vulkan, "Upload arena reserved {} MiB at startup",
                 CachedBytes(MemoryUsage::Upload) >> 20);
    }
}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(size_t size, MemoryUsage usage, bool deferred) {
    if (!deferred && usage == MemoryUsage::Upload && size <= region_size) {
        return GetStreamBuffer(size);
    }
    return GetStagingBuffer(size, usage, deferred);
}

LargeUploadStaging StagingBufferPool::TryLargeUpload(size_t size) {
    namespace Policy = StagingBufferPolicy;
    if (size <= Policy::WarmUploadArenaBlockSize || upload_arena.empty()) {
        return {};
    }

    auto& semaphore = scheduler.GetMasterSemaphore();
    semaphore.Refresh();
    const u32 size_class = Policy::SizeClass(size, reuse_staging);
    const u32 last_class = Policy::LastReuseClass(size_class, reuse_staging);
    for (u32 candidate = size_class; candidate <= last_class; ++candidate) {
        if (auto ref = TryGetReservedBuffer(candidate, MemoryUsage::Upload, false)) {
            ref->mapped_span = ref->mapped_span.first(size);
            return {*ref};
        }
    }

    boost::container::static_vector<Policy::UploadArenaReservation, Policy::UploadArenaBlockCount>
        reservations;
    for (const auto& block : upload_arena) {
        reservations.push_back({block.cursor, block.buffer.Mapped().size()});
    }
    const size_t remaining =
        Policy::PlanLargeUpload(std::span{reservations.data(), reservations.size()}, size,
                                upload_arena_alignment, scheduler.CurrentTick(),
                                semaphore.KnownGpuTick(), reuse_staging);
    if (remaining == size) {
        return {};
    }

    LargeUploadStaging slices;
    for (size_t i = 0; i < reservations.size(); ++i) {
        const auto& reservation = reservations[i];
        if (reservation.size != 0) {
            auto& block = upload_arena[i];
            block.cursor = reservation.cursor;
            slices.push_back(block.Slice(reservation.offset, reservation.size));
        }
    }
    if (remaining != 0) {
        auto spill = GetStagingBuffer(remaining, MemoryUsage::Upload);
        spill.mapped_span = spill.mapped_span.first(remaining);
        slices.push_back(spill);
    }
    LOG_INFO(Render_Vulkan, "Large buffer upload: {} MiB from arena, {} MiB from staging",
             (size - remaining) / 1_MiB, remaining / 1_MiB);
    return slices;
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    auto& entries = GetCache(ref.usage)[ref.size_class].entries;
    const auto is_this_one = [&ref](const StagingBuffer& entry) {
        return entry.index == ref.index;
    };
    auto it = std::find_if(entries.begin(), entries.end(), is_this_one);
    ASSERT(it != entries.end());
    ASSERT(it->deferred);
    it->tick = scheduler.CurrentTick();
    it->last_use_frame = frame;
    it->deferred = false;
}

void StagingBufferPool::TickFrame() {
    if (reuse_staging) {
        ++frame;
        size_t deletions_left = StagingBufferPolicy::DeletionsPerFrame;
        ReleaseInactive(upload_cache, deletions_left);
        ReleaseInactive(download_cache, deletions_left);
        ReleaseInactive(device_local_cache, deletions_left);
        return;
    }
    current_delete_level = (current_delete_level + 1) % StagingBufferPolicy::SizeBits;

    ReleaseCache(MemoryUsage::DeviceLocal);
    ReleaseCache(MemoryUsage::Upload);
    ReleaseCache(MemoryUsage::Download);
}

StagingBufferRef StagingBufferPool::GetStreamBuffer(size_t size) {
    if (AreRegionsActive(Region(free_iterator) + 1,
                         (std::min)(Region(iterator + size) + 1, NUM_SYNCS))) {
        // Avoid waiting for the previous usages to be free
        return GetStagingBuffer(size, MemoryUsage::Upload);
    }
    const u64 current_tick = scheduler.CurrentTick();
    std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + Region(iterator),
              current_tick);
    used_iterator = iterator;
    free_iterator = (std::max)(free_iterator, iterator + size);

    if (iterator + size >= stream_buffer_size) {
        std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + NUM_SYNCS,
                  current_tick);
        used_iterator = 0;
        iterator = 0;
        free_iterator = size;

        if (AreRegionsActive(0, Region(size) + 1)) {
            // Avoid waiting for the previous usages to be free
            return GetStagingBuffer(size, MemoryUsage::Upload);
        }
    }
    const size_t offset = iterator;
    iterator = Common::AlignUp(iterator + size, MAX_ALIGNMENT);
    return StagingBufferRef{
        .buffer = *stream_buffer,
        .device_address = stream_buffer_address,
        .offset = static_cast<VkDeviceSize>(offset),
        .mapped_span = stream_pointer.subspan(offset, size),
        .usage{},
        .size_class{},
        .index{},
    };
}

bool StagingBufferPool::AreRegionsActive(size_t region_begin, size_t region_end) const {
    const u64 gpu_tick = scheduler.GetMasterSemaphore().KnownGpuTick();
    return std::any_of(sync_ticks.begin() + region_begin, sync_ticks.begin() + region_end,
                       [gpu_tick](u64 sync_tick) { return gpu_tick < sync_tick; });
};

StagingBufferRef StagingBufferPool::GetStagingBuffer(size_t size, MemoryUsage usage,
                                                     bool deferred) {
    const u32 size_class = StagingBufferPolicy::SizeClass(size, reuse_staging);
    const u32 last_class = StagingBufferPolicy::LastReuseClass(size_class, reuse_staging);
    for (u32 candidate = size_class; candidate <= last_class; ++candidate) {
        if (const auto ref = TryGetReservedBuffer(candidate, usage, deferred)) {
            return *ref;
        }
    }
    if (upload_arena_enabled && usage == MemoryUsage::Upload && !deferred && size != 0 &&
        size <= StagingBufferPolicy::WarmUploadArenaBlockSize) {
        if (const auto ref = TryGetUploadSlice(size)) {
            return *ref;
        }
    }
    return CreateStagingBuffer(size_class, usage, deferred);
}

std::optional<StagingBufferRef> StagingBufferPool::TryGetUploadSlice(size_t size) {
    const u64 tick = scheduler.CurrentTick();
    const u64 completed_tick = scheduler.GetMasterSemaphore().KnownGpuTick();
    const auto reserve = [&](UploadArenaBlock& block) -> std::optional<StagingBufferRef> {
        const auto mapped = block.buffer.Mapped();
        const auto offset = block.cursor.TryReserve(size, upload_arena_alignment, mapped.size(),
                                                    tick, completed_tick);
        if (!offset) {
            return std::nullopt;
        }
        return block.Slice(*offset, size);
    };
    for (auto& block : upload_arena) {
        if (const auto ref = reserve(block)) {
            return ref;
        }
    }
    if (size > StagingBufferPolicy::UploadArenaBlockSize ||
        !AddUploadArenaBlock(StagingBufferPolicy::UploadArenaBlockSize)) {
        return std::nullopt;
    }
    return reserve(upload_arena.back());
}

bool StagingBufferPool::AddUploadArenaBlock(size_t capacity) {
    if (!upload_arena_can_grow || upload_arena.size() == StagingBufferPolicy::UploadArenaBlockCount) {
        return false;
    }
    vk::Buffer buffer;
    try {
        buffer = CreateBuffer(capacity, MemoryUsage::Upload);
    } catch (const vk::Exception& error) {
        if (error.GetResult() != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
            error.GetResult() != VK_ERROR_OUT_OF_HOST_MEMORY) {
            throw;
        }
        upload_arena_can_grow = false;
        return false;
    }
    if (!buffer.IsHostCoherent() || buffer.Mapped().empty()) {
        upload_arena_can_grow = false;
        return false;
    }
    if (capacity > StagingBufferPolicy::UploadArenaBlockSize) {
        const auto properties = device.GetPhysical().GetMemoryProperties().memoryProperties;
        const u32 heap = properties.memoryTypes[buffer.Location().memory_type].heapIndex;
        if (properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            return false;
        }
    }
    const VkDeviceAddress device_address = GetBufferAddress(*buffer);
    upload_arena.push_back(UploadArenaBlock{
        .buffer = std::move(buffer),
        .device_address = device_address,
        .cursor{},
    });
    return true;
}

std::optional<StagingBufferRef> StagingBufferPool::TryGetReservedBuffer(u32 size_class,
                                                                        MemoryUsage usage,
                                                                        bool deferred) {
    StagingBuffers& cache_level = GetCache(usage)[size_class];

    const auto is_free = [this](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick);
    };
    auto& entries = cache_level.entries;
    const auto hint_it = entries.begin() + cache_level.iterate_index;
    auto it = std::find_if(entries.begin() + cache_level.iterate_index, entries.end(), is_free);
    if (it == entries.end()) {
        it = std::find_if(entries.begin(), hint_it, is_free);
        if (it == hint_it) {
            return std::nullopt;
        }
    }
    cache_level.iterate_index = std::distance(entries.begin(), it) + 1;
    it->tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick();
    it->last_use_frame = frame;
    ASSERT(!it->deferred);
    it->deferred = deferred;
    return it->Ref();
}

vk::Buffer StagingBufferPool::CreateBuffer(VkDeviceSize size, MemoryUsage usage) {
    const VideoCommon::StutterTrace::SiteScope site{VideoCommon::StutterTrace::Site::StagingCreate};
    VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, // mesh emulation dispatch and draw arguments
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    vk::Buffer buffer = memory_allocator.CreateBuffer(buffer_ci, usage);
    if (device.HasDebuggingToolAttached()) {
        ++buffer_index;
        buffer.SetObjectNameEXT(fmt::format("Staging Buffer {}", buffer_index).c_str());
    }
    return buffer;
}

VkDeviceAddress StagingBufferPool::GetBufferAddress(VkBuffer buffer) const {
    return device.IsBufferDeviceAddressSupported()
               ? device.GetLogical().GetBufferDeviceAddress(buffer)
               : VkDeviceAddress{};
}

StagingBufferRef StagingBufferPool::CreateStagingBuffer(u32 size_class, MemoryUsage usage,
                                                       bool deferred) {
    vk::Buffer buffer = CreateBuffer(StagingBufferPolicy::ClassSize(size_class, reuse_staging), usage);
    const std::span<u8> mapped_span = buffer.Mapped();
    const VkDeviceAddress buffer_address = GetBufferAddress(*buffer);
    StagingBuffer& entry = GetCache(usage)[size_class].entries.emplace_back(StagingBuffer{
        .buffer = std::move(buffer),
        .device_address = buffer_address,
        .mapped_span = mapped_span,
        .usage = usage,
        .size_class = size_class,
        .index = unique_ids++,
        .tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick(),
        .last_use_frame = frame,
        .deferred = deferred,
    });
    return entry.Ref();
}

StagingBufferPool::StagingBuffersCache& StagingBufferPool::GetCache(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
        return device_local_cache;
    case MemoryUsage::Upload:
        return upload_cache;
    case MemoryUsage::Download:
        return download_cache;
    default:
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return upload_cache;
    }
}

u64 StagingBufferPool::CachedBytes(MemoryUsage usage) {
    u64 bytes = 0;
    if (usage == MemoryUsage::Upload) {
        for (const auto& block : upload_arena) {
            bytes += block.buffer.Mapped().size();
        }
    }
    const StagingBuffersCache& cache = GetCache(usage);
    const u32 level_count = reuse_staging ? StagingBufferPolicy::NumClasses
                                         : StagingBufferPolicy::SizeBits + 1;
    for (u32 level = 0; level < level_count; ++level) {
        if (!cache[level].entries.empty()) {
            bytes += cache[level].entries.size() * StagingBufferPolicy::ClassSize(level, reuse_staging);
        }
    }
    return bytes;
}

void StagingBufferPool::ReleaseCache(MemoryUsage usage) {
    ReleaseLevel(GetCache(usage), current_delete_level);
}

void StagingBufferPool::ReleaseInactive(StagingBuffersCache& cache, size_t& deletions_left) {
    const u64 gpu_tick = scheduler.GetMasterSemaphore().KnownGpuTick();
    for (auto level = cache.rbegin(); level != cache.rend() && deletions_left; ++level) {
        auto& entries = level->entries;
        std::erase_if(entries, [this, gpu_tick, &deletions_left](const StagingBuffer& entry) {
            if (!deletions_left ||
                !StagingBufferPolicy::CanRelease(frame, entry.last_use_frame, entry.deferred,
                                                 entry.tick <= gpu_tick)) {
                return false;
            }
            --deletions_left;
            return true;
        });
        if (level->iterate_index > entries.size()) {
            level->iterate_index = 0;
        }
    }
}

void StagingBufferPool::ReleaseLevel(StagingBuffersCache& cache, size_t log2) {
    constexpr size_t deletions_per_tick = 16;
    auto& staging = cache[log2];
    auto& entries = staging.entries;
    const size_t old_size = entries.size();

    // Keep two idle upload buffers per level up to 32 MiB (under 128 MiB in all). Each level is
    // swept every 64 frames, so after a pause each upload over 8 MiB (past the stream buffer's
    // regions) got a new dedicated allocation, written to for the first time (16 MiB on an
    // RTX 4080, idle system: 0.4 ms to create, 1.2 ms more for the first write).
    constexpr size_t KeepMaxLog2 = 25;
    constexpr size_t KeepPerLevel = 2;
    // EDEN_NX2_STAGING_KEEP_MIB experiment: also keep one idle upload buffer per level up to
    // that size, so a loading burst's texture uploads over 32 MiB reuse them.
    static const size_t keep_large_log2 = [] {
        const char* value = std::getenv("EDEN_NX2_STAGING_KEEP_MIB");
        const u64 mib = value ? std::strtoull(value, nullptr, 10) : 0;
        return mib > 32 && mib <= 1024 ? Common::Log2Ceil<u64>(mib << 20) : size_t{0};
    }();
    const size_t keep_per_level = log2 <= KeepMaxLog2 ? KeepPerLevel : 1;
    if (&cache == &upload_cache && (log2 <= KeepMaxLog2 || log2 <= keep_large_log2)) {
        size_t idle = 0;
        std::erase_if(entries, [this, &idle, keep_per_level](const StagingBuffer& entry) {
            return scheduler.IsFree(entry.tick) && ++idle > keep_per_level;
        });
        staging.delete_index = 0;
        if (staging.iterate_index > entries.size()) {
            staging.iterate_index = 0;
        }
        return;
    }

    const auto is_deletable = [this](const StagingBuffer& entry) {
        return scheduler.IsFree(entry.tick);
    };
    const size_t begin_offset = staging.delete_index;
    const size_t end_offset = (std::min)(begin_offset + deletions_per_tick, old_size);
    const auto begin = entries.begin() + begin_offset;
    const auto end = entries.begin() + end_offset;
    entries.erase(std::remove_if(begin, end, is_deletable), end);

    const size_t new_size = entries.size();
    staging.delete_index += deletions_per_tick;
    if (staging.delete_index >= new_size) {
        staging.delete_index = 0;
    }
    if (staging.iterate_index > new_size) {
        staging.iterate_index = 0;
    }
}

} // namespace Vulkan
