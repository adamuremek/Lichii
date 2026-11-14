// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstdlib>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "common/fs/file.h"
#include "common/fs/path_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "input_common/drivers/tas_input.h"
#include "input_common/input_poller.h"

namespace {

void SetRouteEnvironment(const char* name, const std::filesystem::path& path) {
#ifdef _WIN32
    REQUIRE(_wputenv_s(Common::UTF8ToUTF16W(name).c_str(), path.c_str()) == 0);
#else
    REQUIRE((path.empty() ? unsetenv(name) : setenv(name, path.c_str(), 1)) == 0);
#endif
}

} // namespace

TEST_CASE("A recorded TAS route replays once with the normal TAS setting disabled",
          "[input][tas]") {
    using InputCommon::TasInput::Tas;
    using InputCommon::TasInput::TasState;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory =
        std::filesystem::temp_directory_path() / ("tas-route-" + std::to_string(stamp));
    REQUIRE(std::filesystem::create_directory(directory));
    const auto previous_directory = Common::FS::GetEdenPath(Common::FS::EdenPath::TASDir);
    const bool previous_enabled = Settings::values.tas_enable.GetValue();
    const bool previous_loop = Settings::values.tas_loop.GetValue();
    SCOPE_EXIT {
        Settings::values.tas_enable = previous_enabled;
        Settings::values.tas_loop = previous_loop;
        Common::FS::SetEdenPath(Common::FS::EdenPath::TASDir, previous_directory);
        SetRouteEnvironment("EDEN_NX2_TAS_RECORD_ROUTE", {});
        SetRouteEnvironment("EDEN_NX2_TAS_ROUTE", {});
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    };
    Common::FS::SetEdenPath(Common::FS::EdenPath::TASDir, directory);
    Settings::values.tas_enable = false;
    Settings::values.tas_loop = true;
    const auto path = directory / u8"route café.txt";
    SetRouteEnvironment("EDEN_NX2_TAS_RECORD_ROUTE", path);
    {
        Tas recorder{"tas"};
        REQUIRE(recorder.IsEnabled());
        REQUIRE(recorder.HasRouteRecordingFile());
        REQUIRE(recorder.Record());
        recorder.RecordInput(1, {1.0f, 0.0f}, {});
        recorder.UpdateThread();
        recorder.UpdateThread();
        recorder.RecordInput(0, {}, {});
        recorder.UpdateThread();
        REQUIRE_FALSE(recorder.Record());
        recorder.SaveRecording(true);
    }
    CHECK_FALSE(std::filesystem::exists(directory / "record.txt"));
    CHECK_FALSE(std::filesystem::exists(directory / "script0-1.txt"));
    REQUIRE(std::filesystem::is_regular_file(path));

    SetRouteEnvironment("EDEN_NX2_TAS_RECORD_ROUTE", {});
    SetRouteEnvironment("EDEN_NX2_TAS_ROUTE", path);
    Tas replay{"tas"};
    REQUIRE(replay.IsEnabled());
    CHECK_FALSE(replay.HasRouteRecordingFile());
    const PadIdentifier pad{};
    size_t button_changes = 0;
    replay.SetCallback({pad, EngineInputType::Button, 0, {[&] { ++button_changes; }}});
    replay.StartStop();
    replay.UpdateThread();
    CHECK(replay.GetButton(pad, 0));
    CHECK(replay.GetAxis(pad, 0) == 1.0f);
    CHECK(button_changes == 1);
    replay.UpdateThread();
    CHECK(button_changes == 1);
    replay.UpdateThread();
    CHECK_FALSE(replay.GetButton(pad, 0));
    CHECK(replay.GetAxis(pad, 0) == 0.0f);
    CHECK(button_changes == 2);
    replay.UpdateThread();
    CHECK(std::get<0>(replay.GetStatus()) == TasState::Stopped);
    CHECK_FALSE(Settings::values.tas_enable.GetValue());
}
