// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace Service::FileSystem {

enum class ReadTraceSource { Storage, File };

void RecordRead(ReadTraceSource source, u64 bytes, std::chrono::steady_clock::duration elapsed);

/// EDEN_NX2_FS_READ_TRACE=2 also reports, every 10 s per guest storage or file, how many of the
/// bytes read had been read before, and where the re-reads were.
[[nodiscard]] inline bool DetailEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FS_READ_TRACE");
        return value && value[0] == '2' && value[1] == '\0';
    }();
    return enabled;
}

/// Which 64 KiB chunks of one stream have been read.
class ReadReuse {
public:
    static constexpr u64 ChunkSize = 64 * 1024;

    /// Returns the bytes of [offset, offset + size) in chunks read before, and marks the range read.
    u64 Add(u64 offset, u64 size);

    [[nodiscard]] std::size_t DistinctChunks() const {
        return seen.size();
    }

private:
    std::unordered_set<u64> seen;
};

/// A RomFS file's place in its storage.
struct RomFsEntry {
    u64 offset;
    u64 size;
    std::string path;
};

/// Reads a RomFS's file table into storage offsets and paths; empty for anything else.
std::vector<RomFsEntry> IndexRomFs(const FileSys::VirtualFile& file);

/// EDEN_NX2_FS_READ_BACKTRACE=<text>, with EDEN_NX2_FS_READ_TRACE=2: reads of a RomFS file whose
/// path contains the text log the requesting guest thread's backtrace (the first 12, then 1 in 200).
[[nodiscard]] inline const char* BacktraceFilter() {
    static const char* const filter = [] {
        const char* value = std::getenv("EDEN_NX2_FS_READ_BACKTRACE");
        return value && value[0] != '\0' ? value : nullptr;
    }();
    return filter;
}

/// stream identifies the guest object (its IStorage or IFile); name labels it in the log. For a
/// storage holding a RomFS, romfs lets the report name the files inside it.
void RecordDetail(ReadTraceSource source, const void* stream, const std::string& name, u64 offset,
                  u64 bytes, const FileSys::VirtualFile* romfs = nullptr);

/// EDEN_NX2_FS_READ_TRACE=1 times the backend read of a guest storage or file read; about once a
/// second while reads happen, the log gets their count, bytes and latency percentiles.
class ReadTraceScope {
public:
    ReadTraceScope(ReadTraceSource source_, s64 bytes_) : source{source_}, bytes{bytes_} {
        if (Enabled) {
            start = std::chrono::steady_clock::now();
        }
    }

    ~ReadTraceScope() {
        if (Enabled) {
            RecordRead(source, static_cast<u64>(bytes), std::chrono::steady_clock::now() - start);
        }
    }

    ReadTraceScope(const ReadTraceScope&) = delete;
    ReadTraceScope& operator=(const ReadTraceScope&) = delete;

private:
    static inline const bool Enabled = std::getenv("EDEN_NX2_FS_READ_TRACE") != nullptr;

    ReadTraceSource source;
    s64 bytes;
    std::chrono::steady_clock::time_point start{};
};

} // namespace Service::FileSystem
