// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_vector.h"

#include "core/hle/service/filesystem/fsp/fs_read_trace.h"

using Service::FileSystem::ReadReuse;

TEST_CASE("Read reuse counts only bytes in chunks read before", "[fs_read_trace]") {
    constexpr u64 Chunk = ReadReuse::ChunkSize;
    ReadReuse reuse;
    CHECK(reuse.Add(0, Chunk * 2) == 0);
    CHECK(reuse.DistinctChunks() == 2);
    // Half old, half new: the overlap is the second chunk only.
    CHECK(reuse.Add(Chunk, Chunk * 2) == Chunk);
    CHECK(reuse.DistinctChunks() == 3);
    // A small read inside a chunk read before counts in full.
    CHECK(reuse.Add(Chunk + 100, 50) == 50);
    // A read that starts mid-chunk and crosses into a new one.
    CHECK(reuse.Add(Chunk * 3 - 10, 20) == 10);
    CHECK(reuse.DistinctChunks() == 4);
    CHECK(reuse.Add(Chunk * 100, 0) == 0);
    CHECK(reuse.DistinctChunks() == 4);
}

TEST_CASE("RomFS index finds every file's path and storage offset", "[fs_read_trace]") {
    using namespace FileSys;
    const auto file = [](std::string name, std::string text) {
        return std::make_shared<VectorVfsFile>(std::vector<u8>(text.begin(), text.end()),
                                               std::move(name));
    };
    auto deep = std::make_shared<VectorVfsDirectory>(
        std::vector<VirtualFile>{file("y.dat", "deep file")}, std::vector<VirtualDir>{}, "deep");
    auto data = std::make_shared<VectorVfsDirectory>(
        std::vector<VirtualFile>{file("x.dat", "data file"), file("w.dat", "second data file")},
        std::vector<VirtualDir>{deep}, "data");
    auto other = std::make_shared<VectorVfsDirectory>(
        std::vector<VirtualFile>{file("z.bin", "other file")}, std::vector<VirtualDir>{}, "other");
    auto root = std::make_shared<VectorVfsDirectory>(
        std::vector<VirtualFile>{file("a.bin", "root file")}, std::vector<VirtualDir>{data, other});
    const VirtualFile romfs = CreateRomFS(root);
    REQUIRE(romfs != nullptr);

    auto entries = Service::FileSystem::IndexRomFs(romfs);
    std::map<std::string, std::string> contents;
    for (const auto& e : entries) {
        const auto bytes = romfs->ReadBytes(e.size, e.offset);
        contents[e.path] = std::string(bytes.begin(), bytes.end());
    }
    const std::map<std::string, std::string> expected{
        {"/a.bin", "root file"},          {"/data/x.dat", "data file"},
        {"/data/w.dat", "second data file"}, {"/data/deep/y.dat", "deep file"},
        {"/other/z.bin", "other file"}};
    CHECK(contents == expected);

    const VirtualFile not_romfs = file("junk", std::string(0x60, 'x'));
    CHECK(Service::FileSystem::IndexRomFs(not_romfs).empty());
}
