// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstddef>
#include <string>
#include <fmt/format.h>

#include "dynarmic/backend/x64/perf_map.h"
#include "common/common_types.h"
#if defined(__linux__) && !defined(__ANDROID__)
#    include <cstdio>
#    include <cstdlib>
#    include <mutex>
#    include <sys/types.h>
#    include <unistd.h>

namespace Dynarmic::Backend::X64 {

namespace {
std::mutex mutex;
std::FILE* file = nullptr;

void OpenFile() {
    const char* perf_dir = std::getenv("PERF_BUILDID_DIR");
    if (!perf_dir) {
        file = nullptr;
        return;
    }

    const pid_t pid = getpid();
    const std::string filename = fmt::format("{:s}/perf-{:d}.map", perf_dir, pid);

    file = std::fopen(filename.c_str(), "w");
    if (!file) {
        return;
    }

    std::setvbuf(file, nullptr, _IONBF, 0);
}
}  // anonymous namespace

namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name) {
    if (start == end) {
        // Nothing to register
        return;
    }

    std::lock_guard guard{mutex};

    if (!file) {
        OpenFile();
        if (!file) {
            return;
        }
    }

    const std::string line = fmt::format("{:016x} {:016x} {:s}\n", reinterpret_cast<u64>(start), reinterpret_cast<u64>(end) - reinterpret_cast<u64>(start), friendly_name);
    std::fwrite(line.data(), sizeof *line.data(), line.size(), file);
}
}  // namespace detail

void PerfMapClear() {
    std::lock_guard guard{mutex};

    if (!file) {
        return;
    }

    std::fclose(file);
    file = nullptr;
    OpenFile();
}

}  // namespace Dynarmic::Backend::X64

#elif defined(_WIN32)

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>

#    include <atomic>
#    include <cstdio>
#    include <cstring>
#    include <mutex>

namespace Dynarmic::Backend::X64 {
namespace {

class WindowsPerfMap {
public:
    WindowsPerfMap() noexcept {
        wchar_t path[32768];
        const DWORD length = GetEnvironmentVariableW(L"EDEN_NX2_JIT_MAP", path, 32768);
        if (length == 0 || length >= 32768) {
            return;
        }

        LARGE_INTEGER frequency, counter;
        if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter)) {
            return;
        }
        FILETIME utc;
        GetSystemTimeAsFileTime(&utc);
        const u64 utc_filetime = (static_cast<u64>(utc.dwHighDateTime) << 32) | utc.dwLowDateTime;

        // A capture must use a fresh path; never overwrite or mix runs.
        file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return;
        }
        char header[512];
        const int size = std::snprintf(header, sizeof(header),
            "# eden-nx2-jit-map-v1 pid=%lu qpc_frequency=%lld qpc=%lld utc_filetime=%llu\n"
            "# R qpc host_address_hex size_hex name; C qpc invalidates_all; X qpc reason ends_coverage\n"
            "# E qpc normal-exit confirms complete coverage through that QPC.\n"
            "# Without E, coverage after the last R/C is unknown. C invalidates every prior mapping.\n",
            GetCurrentProcessId(), frequency.QuadPart, counter.QuadPart,
            static_cast<unsigned long long>(utc_filetime));
        if (size <= 0 || static_cast<size_t>(size) >= sizeof(header)) {
            Stop("header-too-large");
            return;
        }
        if (Append(header, static_cast<DWORD>(size))) {
            enabled.store(true, std::memory_order_relaxed);
        }
    }

    ~WindowsPerfMap() {
        enabled.store(false, std::memory_order_relaxed);
        std::lock_guard guard{mutex};
        if (file != INVALID_HANDLE_VALUE) {
            Stop("normal-exit", true);
        }
    }

    void Record(bool clear, const void* start = nullptr, const void* end = nullptr,
                std::string_view name = {}) {
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }
        std::lock_guard guard{mutex};
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }

        LARGE_INTEGER counter;
        if (!QueryPerformanceCounter(&counter)) {
            Stop("clock-error");
            return;
        }
        char line[1024];
        int size;
        if (clear) {
            // Clear is per cache, but this API has no cache identity. Invalidate conservatively.
            size = std::snprintf(line, sizeof(line), "C %lld\n", counter.QuadPart);
        } else {
            if (name.size() > 768 || name.find_first_of("\r\n") != std::string_view::npos) {
                Stop("invalid-name");
                return;
            }
            size = std::snprintf(line, sizeof(line), "R %lld %016llx %016llx %.*s\n",
                counter.QuadPart, static_cast<unsigned long long>(reinterpret_cast<u64>(start)),
                static_cast<unsigned long long>(reinterpret_cast<u64>(end) - reinterpret_cast<u64>(start)),
                static_cast<int>(name.size()), name.data());
        }
        if (size <= 0 || static_cast<size_t>(size) >= sizeof(line)) {
            Stop("record-too-large");
            return;
        }
        Append(line, static_cast<DWORD>(size));
    }

private:
    bool Append(const char* line, DWORD size) noexcept {
        constexpr u64 max_bytes = 256 * 1024 * 1024;
        if (bytes_written + size > max_bytes - 128) {
            Stop("size-limit");
            return false;
        }
        DWORD written = 0;
        if (!WriteFile(file, line, size, &written, nullptr) || written != size) {
            Stop("write-error");
            return false;
        }
        bytes_written += written;
        return true;
    }

    void Stop(const char* reason, bool complete = false) noexcept {
        enabled.store(false, std::memory_order_relaxed);
        LARGE_INTEGER counter{};
        if (!QueryPerformanceCounter(&counter)) {
            complete = false;
            reason = "clock-error";
        }
        char line[128];
        const int size = std::snprintf(line, sizeof(line), "\n%c %lld %s\n",
                                       complete ? 'E' : 'X', counter.QuadPart, reason);
        DWORD written;
        WriteFile(file, line, static_cast<DWORD>(size), &written, nullptr);
        CloseHandle(file);
        file = INVALID_HANDLE_VALUE;
    }

    std::mutex mutex;
    std::atomic<bool> enabled{false};
    HANDLE file = INVALID_HANDLE_VALUE;
    u64 bytes_written = 0;
};

WindowsPerfMap& GetWindowsPerfMap() {
    static WindowsPerfMap map;
    return map;
}

}  // anonymous namespace

namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name) {
    if (reinterpret_cast<u64>(end) <= reinterpret_cast<u64>(start)) {
        return;
    }
    GetWindowsPerfMap().Record(false, start, end, friendly_name);
}
}  // namespace detail

void PerfMapClear() {
    GetWindowsPerfMap().Record(true);
}

namespace {

// Binary records after a text header line: u64 qpc, u64 host address, u32 host size,
// u64 guest pc, u32 guest word count, the host bytes, then the guest words (little-endian).
class WindowsCodeDump {
public:
    WindowsCodeDump() noexcept {
        wchar_t path[32768];
        const DWORD length = GetEnvironmentVariableW(L"EDEN_NX2_JIT_CODE", path, 32768);
        if (length == 0 || length >= 32768) {
            return;
        }
        file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return;
        }
        static constexpr char header[] = "# eden-nx2-jit-code-v1\n";
        if (Append(header, sizeof(header) - 1)) {
            enabled.store(true, std::memory_order_relaxed);
        }
    }

    ~WindowsCodeDump() {
        enabled.store(false, std::memory_order_relaxed);
        std::lock_guard guard{mutex};
        Close();
    }

    bool Enabled() const noexcept {
        return enabled.load(std::memory_order_relaxed);
    }

    void Record(const void* host, size_t host_size, u64 guest_pc,
                std::span<const std::uint32_t> guest_words) noexcept {
        if (!Enabled()) {
            return;
        }
        std::lock_guard guard{mutex};
        if (!Enabled()) {
            return;
        }
        LARGE_INTEGER counter{};
        QueryPerformanceCounter(&counter);
        char head[32];
        const u64 qpc = static_cast<u64>(counter.QuadPart);
        const u64 address = reinterpret_cast<u64>(host);
        const auto size32 = static_cast<std::uint32_t>(host_size);
        const auto count32 = static_cast<std::uint32_t>(guest_words.size());
        std::memcpy(head, &qpc, 8);
        std::memcpy(head + 8, &address, 8);
        std::memcpy(head + 16, &size32, 4);
        std::memcpy(head + 20, &guest_pc, 8);
        std::memcpy(head + 28, &count32, 4);
        if (!Append(head, sizeof(head)) || !Append(host, size32)) {
            return;
        }
        Append(guest_words.data(), static_cast<DWORD>(guest_words.size_bytes()));
    }

private:
    bool Append(const void* data, DWORD size) noexcept {
        constexpr u64 max_bytes = 1024ULL * 1024 * 1024;
        DWORD written = 0;
        if (bytes_written + size > max_bytes ||
            !WriteFile(file, data, size, &written, nullptr) || written != size) {
            enabled.store(false, std::memory_order_relaxed);
            Close();
            return false;
        }
        bytes_written += written;
        return true;
    }

    void Close() noexcept {
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
            file = INVALID_HANDLE_VALUE;
        }
    }

    std::mutex mutex;
    std::atomic<bool> enabled{false};
    HANDLE file = INVALID_HANDLE_VALUE;
    u64 bytes_written = 0;
};

WindowsCodeDump& GetWindowsCodeDump() {
    static WindowsCodeDump dump;
    return dump;
}

}  // anonymous namespace

bool PerfMapCodeEnabled() noexcept {
    return GetWindowsCodeDump().Enabled();
}

void PerfMapRecordCode(const void* host, std::size_t host_size, std::uint64_t guest_pc,
                       std::span<const std::uint32_t> guest_words) noexcept {
    GetWindowsCodeDump().Record(host, host_size, guest_pc, guest_words);
}

}  // namespace Dynarmic::Backend::X64

#endif
