// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/filesystem/fsp/fs_read_trace.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <array>
#include <map>
#include <mutex>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <zstd.h>

#include "common/logging.h"
#include "core/arm/debug.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/service/service.h"

namespace Service::FileSystem {
namespace {

using Clock = std::chrono::steady_clock;

struct ReadStats {
    const char* label;
    std::mutex mutex{};
    Clock::time_point window_start{};
    std::vector<Clock::duration> latencies{};
    u64 window_bytes{};
    u64 total_reads{};
    u64 total_bytes{};
    Clock::duration total_time{};
};

std::array<ReadStats, 2> stats{{{"storage"}, {"file"}}};

double Micros(Clock::duration duration) {
    return std::chrono::duration<double, std::micro>(duration).count();
}

double MiB(u64 bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

/// Per file this window: bytes read, bytes re-read, reads, and the distinct read offsets.
struct FileWindow {
    u64 bytes{};
    u64 reread{};
    u64 reads{};
    std::unordered_set<u64> offsets;
    u64 min_size{~0ULL};
    u64 max_size{};
};

} // namespace

// The build has no RTTI, so this parses the table itself rather than casting ExtractRomFS's files.
std::vector<RomFsEntry> IndexRomFs(const FileSys::VirtualFile& file) {
    struct Header {
        u64 header_size, dir_hash_offset, dir_hash_size, dir_meta_offset, dir_meta_size,
            file_hash_offset, file_hash_size, file_meta_offset, file_meta_size, data_offset;
    } header{};
    static_assert(sizeof(Header) == 0x50);
    std::vector<RomFsEntry> out;
    if (file->ReadObject(&header) != sizeof(Header) || header.header_size != sizeof(Header) ||
        header.dir_meta_size > (64ULL << 20) || header.file_meta_size > (256ULL << 20)) {
        return out;
    }
    const auto dirs = file->ReadBytes(header.dir_meta_size, header.dir_meta_offset);
    const auto files = file->ReadBytes(header.file_meta_size, header.file_meta_offset);
    if (dirs.size() != header.dir_meta_size || files.size() != header.file_meta_size) {
        return out;
    }
    const auto u32_at = [](const std::vector<u8>& v, u64 at) {
        u32 value;
        std::memcpy(&value, v.data() + at, sizeof(value));
        return value;
    };
    const auto u64_at = [](const std::vector<u8>& v, u64 at) {
        u64 value;
        std::memcpy(&value, v.data() + at, sizeof(value));
        return value;
    };
    constexpr u32 Empty = 0xFFFFFFFF;
    std::vector<bool> dir_seen(dirs.size()), file_seen(files.size());
    // Each entry: a directory and its parent's path.
    std::vector<std::pair<u32, std::string>> stack{{0, std::string{}}};
    while (!stack.empty()) {
        auto [dir, parent] = std::move(stack.back());
        stack.pop_back();
        if (dir >= dirs.size() || dirs.size() - dir < 0x18 || dir_seen[dir] || parent.size() > 0x300) {
            continue;
        }
        dir_seen[dir] = true;
        const u32 dir_name_length =
            std::min<u32>(u32_at(dirs, dir + 0x14), static_cast<u32>(dirs.size() - dir - 0x18));
        const std::string path =
            dir == 0 ? std::string{"/"}
                     : parent + std::string(reinterpret_cast<const char*>(dirs.data() + dir + 0x18),
                                            dir_name_length) + "/";
        for (u32 at = u32_at(dirs, dir + 12); at != Empty;) {
            if (at >= files.size() || files.size() - at < 0x20 || file_seen[at]) {
                break;
            }
            file_seen[at] = true;
            const u32 name_length = std::min<u32>(u32_at(files, at + 0x1c),
                                                  static_cast<u32>(files.size() - at - 0x20));
            out.push_back({header.data_offset + u64_at(files, at + 8), u64_at(files, at + 16),
                           path + std::string(reinterpret_cast<const char*>(files.data() + at + 0x20),
                                              name_length)});
            at = u32_at(files, at + 4);
        }
        if (const u32 sibling = u32_at(dirs, dir + 4); sibling != Empty) {
            stack.emplace_back(sibling, parent);
        }
        if (const u32 child = u32_at(dirs, dir + 8); child != Empty) {
            stack.emplace_back(child, path);
        }
    }
    return out;
}

namespace {

/// Decodes every zstd frame of data, with ddict when it is set. Returns the error, or empty.
std::string ZstdDecodeAll(const std::vector<u8>& data, const ZSTD_DDict* ddict, std::vector<u8>* keep,
                          u64& decoded) {
    constexpr u64 KeepLimit = 256ULL << 20;
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (ddict) {
        ZSTD_DCtx_refDDict(dctx, ddict);
    }
    std::vector<u8> scratch(1 << 20);
    ZSTD_inBuffer in{data.data(), data.size(), 0};
    decoded = 0;
    size_t ret = 1;
    std::string error;
    for (;;) {
        ZSTD_outBuffer out{scratch.data(), scratch.size(), 0};
        ret = ZSTD_decompressStream(dctx, &out, &in);
        if (ZSTD_isError(ret)) {
            error = fmt::format("error at input +{:#x}: {}", in.pos, ZSTD_getErrorName(ret));
            break;
        }
        decoded += out.pos;
        if (keep && keep->size() + out.pos <= KeepLimit) {
            keep->insert(keep->end(), scratch.begin(), scratch.begin() + out.pos);
        }
        if (in.pos == in.size && out.pos < out.size) {
            break; // All input consumed and nothing left buffered.
        }
    }
    ZSTD_freeDCtx(dctx);
    if (error.empty() && ret != 0) {
        error = fmt::format("truncated ({} more bytes expected)", ret);
    }
    return error;
}

/// Zstd dictionaries in one RomFS file: the file itself, or the files of a (zstd-compressed) SARC.
std::vector<std::vector<u8>> FindDictionaries(std::vector<u8> data) {
    constexpr u32 ZstdMagic = 0xFD2FB528;
    constexpr u32 DictMagic = 0xEC30A437;
    const auto u32_at = [&](const std::vector<u8>& v, std::size_t at) {
        u32 value = 0;
        if (at + 4 <= v.size()) {
            std::memcpy(&value, v.data() + at, 4);
        }
        return value;
    };
    if (u32_at(data, 0) == ZstdMagic) {
        std::vector<u8> plain;
        u64 decoded = 0;
        if (!ZstdDecodeAll(data, nullptr, &plain, decoded).empty()) {
            return {};
        }
        data = std::move(plain);
    }
    if (u32_at(data, 0) == DictMagic) {
        return {std::move(data)};
    }
    std::vector<std::vector<u8>> dicts;
    if (data.size() < 0x20 || std::memcmp(data.data(), "SARC", 4) != 0) {
        return dicts;
    }
    u16 header_size;
    std::memcpy(&header_size, data.data() + 4, 2);
    const u32 data_offset = u32_at(data, 0x0c);
    const std::size_t sfat = header_size;
    if (sfat + 0x0c > data.size() || std::memcmp(data.data() + sfat, "SFAT", 4) != 0) {
        return dicts;
    }
    u16 nodes;
    std::memcpy(&nodes, data.data() + sfat + 6, 2);
    for (u16 i = 0; i < nodes; ++i) {
        const std::size_t node = sfat + 0x0c + i * 0x10u;
        const u64 begin = u64{data_offset} + u32_at(data, node + 8);
        const u64 end = u64{data_offset} + u32_at(data, node + 12);
        if (begin < end && end <= data.size() && u32_at(data, begin) == DictMagic) {
            dicts.emplace_back(data.begin() + begin, data.begin() + end);
        }
    }
    return dicts;
}

/// Once per file: whether one whole read matches reads in the game's piece size, and whether host
/// zstd decodes the file with the game's own dictionary. A failure here means the bytes are bad.
void CheckZstdFile(const FileSys::VirtualFile& romfs, const std::vector<RomFsEntry>& files,
                   const RomFsEntry& entry, u64 piece) {
    const auto whole = romfs->ReadBytes(entry.size, entry.offset);
    std::vector<u8> pieces;
    for (u64 at = 0; at < entry.size; at += piece) {
        const auto part = romfs->ReadBytes(std::min(piece, entry.size - at), entry.offset + at);
        pieces.insert(pieces.end(), part.begin(), part.end());
    }
    std::string pieces_result = "match";
    if (pieces.size() != whole.size()) {
        pieces_result = fmt::format("size {} vs {}", pieces.size(), whole.size());
    } else if (const auto diff = std::mismatch(whole.begin(), whole.end(), pieces.begin());
               diff.first != whole.end()) {
        pieces_result = fmt::format("differ from +{:#x}", diff.first - whole.begin());
    }
    const unsigned dict_id = ZSTD_getDictID_fromFrame(whole.data(), whole.size());
    std::string dict_source = "none needed";
    ZSTD_DDict* ddict = nullptr;
    if (dict_id != 0) {
        dict_source = "not found";
        std::string searched;
        for (const auto& candidate : files) {
            std::string lower = candidate.path;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower.find("zsdic") == std::string::npos || candidate.size > (64ULL << 20)) {
                continue;
            }
            searched += " " + candidate.path;
            for (auto& dict : FindDictionaries(romfs->ReadBytes(candidate.size, candidate.offset))) {
                if (ZSTD_getDictID_fromDict(dict.data(), dict.size()) == dict_id) {
                    ddict = ZSTD_createDDict(dict.data(), dict.size());
                    dict_source = candidate.path;
                    break;
                }
            }
            if (ddict) {
                break;
            }
        }
        if (!ddict) {
            dict_source += searched.empty() ? std::string{" (no zsdic files)"}
                                            : " (searched" + searched + ")";
        }
    }
    u64 decoded = 0;
    std::string zstd_result = dict_id != 0 && !ddict
                                  ? std::string{"skipped"}
                                  : ZstdDecodeAll(whole, ddict, nullptr, decoded);
    if (zstd_result.empty()) {
        zstd_result = "ok";
    }
    if (ddict) {
        ZSTD_freeDDict(ddict);
    }
    LOG_INFO(Service_FS,
             "Guest read check {}: {} bytes, reads of {:#x} {}, zstd dictionary {} from {}, host "
             "zstd {} ({} bytes decoded)",
             entry.path, whole.size(), piece, pieces_result, dict_id, dict_source, zstd_result,
             decoded);
}

void LogReadBacktrace(const FileSys::VirtualFile* romfs, const std::vector<RomFsEntry>& files,
                      const RomFsEntry& entry, u64 offset, u64 bytes) {
    static u64 hits{};
    ++hits;
    if (hits == 1 && romfs && *romfs) {
        try {
            CheckZstdFile(*romfs, files, entry, std::max<u64>(bytes, 0x1000));
        } catch (...) {
            LOG_INFO(Service_FS, "Guest read check {}: failed to run", entry.path);
        }
    }
    if (hits > 12 && hits % 200 != 0) {
        return;
    }
    const Kernel::KThread* thread = Service::CurrentRequestThread();
    if (thread == nullptr || thread->GetOwnerProcess() == nullptr) {
        LOG_INFO(Service_FS, "Guest read backtrace #{} {} +{:#x} ({} bytes): no requesting thread",
                 hits, entry.path, offset - entry.offset, bytes);
        return;
    }
    std::string frames;
    for (const auto& frame : Core::GetBacktrace(thread)) {
        frames += fmt::format(" {}+{:#x}", frame.module, frame.offset);
        if (!frame.name.empty()) {
            frames += fmt::format("({})", frame.name);
        }
    }
    LOG_INFO(Service_FS, "Guest read backtrace #{} {} +{:#x} ({} bytes) thread {} lr={:#x}:{}", hits,
             entry.path, offset - entry.offset, bytes, thread->GetThreadId(),
             thread->GetContext().lr, frames.empty() ? std::string{" none"} : frames);
}

struct StreamDetail {
    std::string name;
    ReadTraceSource source{};
    ReadReuse reuse;
    bool indexed{};
    std::vector<RomFsEntry> files; ///< Sorted by offset; empty unless the stream is a RomFS.
    std::unordered_map<std::size_t, FileWindow> file_window;
    u64 reads{};
    u64 bytes{};
    u64 reread{};
    u64 total_bytes{};
    u64 total_reread{};
    /// Re-read bytes this window by 4 MiB region.
    std::map<u64, u64> regions;
};

constexpr u64 RegionSize = 4ULL << 20;

struct DetailState {
    std::mutex mutex;
    Clock::time_point window_start{};
    std::unordered_map<const void*, StreamDetail> streams;
};

DetailState detail;

} // namespace

u64 ReadReuse::Add(u64 offset, u64 size) {
    u64 reread = 0;
    for (u64 at = offset, end = offset + size; at < end;) {
        const u64 chunk = at / ChunkSize;
        const u64 chunk_end = std::min(end, (chunk + 1) * ChunkSize);
        if (!seen.insert(chunk).second) {
            reread += chunk_end - at;
        }
        at = chunk_end;
    }
    return reread;
}

void RecordDetail(ReadTraceSource source, const void* stream, const std::string& name, u64 offset,
                  u64 bytes, const FileSys::VirtualFile* romfs) {
    std::scoped_lock lock{detail.mutex};
    const auto now = Clock::now();
    if (detail.window_start == Clock::time_point{}) {
        detail.window_start = now;
    }
    auto& s = detail.streams[stream];
    if (s.reads == 0 && s.total_bytes == 0) {
        s.name = name.empty() ? std::string{"(unnamed)"} : name;
        s.source = source;
    }
    if (!s.indexed && romfs && *romfs) {
        s.indexed = true;
        try {
            s.files = IndexRomFs(*romfs);
        } catch (...) {
            s.files.clear();
        }
        std::sort(s.files.begin(), s.files.end(),
                  [](const RomFsEntry& a, const RomFsEntry& b) { return a.offset < b.offset; });
        LOG_INFO(Service_FS, "Guest read reuse: indexed {} RomFS files of {}", s.files.size(),
                 s.name);
    }
    const u64 reread = s.reuse.Add(offset, bytes);
    if (!s.files.empty()) {
        auto it = std::upper_bound(s.files.begin(), s.files.end(), offset,
                                   [](u64 value, const RomFsEntry& e) { return value < e.offset; });
        if (it != s.files.begin() && offset < std::prev(it)->offset + std::prev(it)->size) {
            auto& f = s.file_window[static_cast<std::size_t>(std::prev(it) - s.files.begin())];
            f.bytes += bytes;
            f.reread += reread;
            ++f.reads;
            if (f.offsets.size() < 4096) {
                f.offsets.insert(offset);
            }
            f.min_size = std::min(f.min_size, bytes);
            f.max_size = std::max(f.max_size, bytes);
            const RomFsEntry& entry = *std::prev(it);
            if (const char* filter = BacktraceFilter();
                filter && entry.path.find(filter) != std::string::npos) {
                LogReadBacktrace(romfs, s.files, entry, offset, bytes);
            }
        }
    }
    ++s.reads;
    s.bytes += bytes;
    s.reread += reread;
    s.total_bytes += bytes;
    s.total_reread += reread;
    if (reread != 0) {
        s.regions[offset / RegionSize] += reread;
    }
    if (now - detail.window_start < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - detail.window_start).count();
    std::vector<StreamDetail*> active;
    for (auto& [key, value] : detail.streams) {
        if (value.reads != 0) {
            active.push_back(&value);
        }
    }
    std::sort(active.begin(), active.end(),
              [](const StreamDetail* a, const StreamDetail* b) { return a->bytes > b->bytes; });
    for (std::size_t i = 0; i < std::min<std::size_t>(active.size(), 6); ++i) {
        StreamDetail& d = *active[i];
        std::vector<std::pair<u64, u64>> hot(d.regions.begin(), d.regions.end());
        std::sort(hot.begin(), hot.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string regions;
        for (std::size_t r = 0; r < std::min<std::size_t>(hot.size(), 6); ++r) {
            regions += fmt::format(" {:#x}:{:.1f}", hot[r].first * RegionSize, MiB(hot[r].second));
        }
        LOG_INFO(Service_FS,
                 "Guest read reuse ({}) {}: {} reads in {:.1f} s, {:.2f} MiB/s, re-read {:.2f} "
                 "MiB/s ({:.0f}%), {:.1f} MiB distinct since boot, {:.1f} MiB re-read since boot "
                 "of {:.1f} MiB | re-read MiB by 4 MiB region:{}",
                 d.source == ReadTraceSource::Storage ? "storage" : "file", d.name, d.reads,
                 seconds, MiB(d.bytes) / seconds, MiB(d.reread) / seconds,
                 d.bytes ? 100.0 * static_cast<double>(d.reread) / static_cast<double>(d.bytes)
                         : 0.0,
                 MiB(d.reuse.DistinctChunks() * ReadReuse::ChunkSize), MiB(d.total_reread),
                 MiB(d.total_bytes), regions.empty() ? std::string{" none"} : regions);
        std::vector<std::pair<std::size_t, const FileWindow*>> files;
        for (const auto& [index, f] : d.file_window) {
            files.emplace_back(index, &f);
        }
        std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
            return a.second->reread != b.second->reread ? a.second->reread > b.second->reread
                                                        : a.second->bytes > b.second->bytes;
        });
        for (std::size_t r = 0; r < std::min<std::size_t>(files.size(), 8); ++r) {
            const auto& entry = d.files[files[r].first];
            const FileWindow& f = *files[r].second;
            LOG_INFO(Service_FS,
                     "Guest read reuse file {} ({:.2f} MiB at {:#x}): {} reads, {:.2f} MiB/s, re-read "
                     "{:.2f} MiB/s, {} distinct offsets, read size {}-{} bytes",
                     entry.path, MiB(entry.size), entry.offset, f.reads, MiB(f.bytes) / seconds,
                     MiB(f.reread) / seconds, f.offsets.size(), f.min_size, f.max_size);
        }
    }
    for (auto& [key, value] : detail.streams) {
        value.reads = value.bytes = value.reread = 0;
        value.regions.clear();
        value.file_window.clear();
    }
    detail.window_start = now;
}

void RecordRead(ReadTraceSource source, u64 bytes, Clock::duration elapsed) {
    auto& s = stats[static_cast<std::size_t>(source)];
    std::scoped_lock lock{s.mutex};
    const auto now = Clock::now();
    if (s.latencies.empty()) {
        s.window_start = now - elapsed;
    }
    s.latencies.push_back(elapsed);
    s.window_bytes += bytes;
    ++s.total_reads;
    s.total_bytes += bytes;
    s.total_time += elapsed;
    if (now - s.window_start < std::chrono::seconds{1}) {
        return;
    }

    auto& l = s.latencies;
    std::sort(l.begin(), l.end());
    const auto at = [&](double q) { return l[static_cast<std::size_t>(q * (l.size() - 1))]; };
    const auto over = [&](auto limit) {
        return l.end() - std::upper_bound(l.begin(), l.end(), Clock::duration{limit});
    };
    LOG_INFO(Service_FS,
             "Guest {} reads: {} in {:.2f} s, {:.2f} MiB, p50 {:.1f} us, p99 {:.1f} us, max {:.1f} "
             "us, {} over 1 ms, {} over 16 ms; since boot {} reads, {:.1f} MiB, {:.3f} s reading",
             s.label, l.size(), std::chrono::duration<double>(now - s.window_start).count(),
             MiB(s.window_bytes), Micros(at(0.50)), Micros(at(0.99)), Micros(l.back()),
             over(std::chrono::milliseconds{1}), over(std::chrono::milliseconds{16}), s.total_reads,
             MiB(s.total_bytes), std::chrono::duration<double>(s.total_time).count());
    l.clear();
    s.window_bytes = 0;
}

} // namespace Service::FileSystem
