// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <catch2/catch_test_macros.hpp>
#include "common/game_presets.h"
#include "common/settings.h"

TEST_CASE("Display filter defaults use ScaleForce and FXAA", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    CHECK(settings->scaling_filter.GetDefault() == Settings::ScalingFilter::ScaleForce);
    CHECK(settings->anti_aliasing.GetDefault() == Settings::AntiAliasing::Fxaa);
}

TEST_CASE("Ocarina uses Bounded draws while the setting is at its Wait default", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    REQUIRE(settings->native_async_draws.GetValue() == Settings::NativeAsyncDraws::Wait);
    CHECK(Settings::ApplyOcarinaOfTimeAsyncDraws(*settings, Settings::OcarinaOfTimeProgramId));
    CHECK(settings->native_async_draws.GetValue() == Settings::NativeAsyncDraws::Bounded);
    // The session value is per-game: the global preference stays Wait.
    CHECK_FALSE(settings->native_async_draws.UsingGlobal());
    CHECK(settings->native_async_draws.GetValue(true) == Settings::NativeAsyncDraws::Wait);
}

TEST_CASE("Ocarina keeps a chosen draw mode and other games keep Wait", "[game_presets]") {
    for (const auto chosen : {Settings::NativeAsyncDraws::Skip, Settings::NativeAsyncDraws::Bounded}) {
        auto settings = std::make_unique<Settings::Values>();
        settings->native_async_draws.SetValue(chosen);
        CHECK_FALSE(Settings::ApplyOcarinaOfTimeAsyncDraws(*settings, Settings::OcarinaOfTimeProgramId));
        CHECK(settings->native_async_draws.GetValue() == chosen);
        CHECK(settings->native_async_draws.UsingGlobal());
    }
    auto settings = std::make_unique<Settings::Values>();
    CHECK_FALSE(Settings::ApplyOcarinaOfTimeAsyncDraws(*settings, Settings::MarioKartWorldProgramId));
    CHECK_FALSE(Settings::ApplyOcarinaOfTimeAsyncDraws(*settings, 0));
    CHECK(settings->native_async_draws.GetValue() == Settings::NativeAsyncDraws::Wait);
    CHECK(settings->native_async_draws.UsingGlobal());
}
