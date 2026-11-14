// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fmt/ranges.h>

#include "common/cityhash.h"
#include "common/dkb_frame_trace.h"
#include "common/fs/file.h"
#include "common/fs/fs_types.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "input_common/drivers/tas_input.h"

namespace InputCommon::TasInput {
namespace {

std::filesystem::path RouteFileFromEnvironment(const char* name) {
#ifdef _WIN32
    const auto* value = _wgetenv(Common::UTF8ToUTF16W(name).c_str());
#else
    const auto* value = std::getenv(name);
#endif
    return value ? std::filesystem::path{value} : std::filesystem::path{};
}

std::string_view BenchmarkName(GpuBenchmark::Experiment experiment) {
    using Experiment = GpuBenchmark::Experiment;
    switch (experiment) {
    case Experiment::Renderer:
        return "renderer";
    case Experiment::CpuPlacement:
        return "cpu-placement";
    case Experiment::TextureReads:
        return "texture-reads";
    case Experiment::BufferReads:
        return "buffer-reads";
    case Experiment::AcquireWaits:
        return "acquire-waits";
    default:
        return "adaptive-submit";
    }
}

std::string_view BenchmarkMode(GpuBenchmark::Experiment experiment, GpuBenchmark::Mode mode) {
    using Mode = GpuBenchmark::Mode;
    switch (mode) {
    case Mode::AdaptiveSubmit:
        return experiment == GpuBenchmark::Experiment::Renderer ? "adaptive-submit" : "candidate";
    case Mode::RasterStateReuse:
        return "raster-state";
    case Mode::Both:
        return "both";
    case Mode::CacheLocal:
        return "cache-local";
    case Mode::TextureReadFilter:
    case Mode::BufferReadFilter:
    case Mode::PreciseAcquireWait:
        return "candidate";
    default:
        return "baseline";
    }
}

} // Anonymous namespace

enum class Tas::TasAxis : u8 {
    StickX,
    StickY,
    SubstickX,
    SubstickY,
    Undefined,
};

// Supported keywords and buttons from a TAS file
constexpr std::array<std::pair<std::string_view, TasButton>, 18> text_to_tas_button = {
    std::pair{"KEY_A", TasButton::BUTTON_A},
    {"KEY_B", TasButton::BUTTON_B},
    {"KEY_X", TasButton::BUTTON_X},
    {"KEY_Y", TasButton::BUTTON_Y},
    {"KEY_LSTICK", TasButton::STICK_L},
    {"KEY_RSTICK", TasButton::STICK_R},
    {"KEY_L", TasButton::TRIGGER_L},
    {"KEY_R", TasButton::TRIGGER_R},
    {"KEY_PLUS", TasButton::BUTTON_PLUS},
    {"KEY_MINUS", TasButton::BUTTON_MINUS},
    {"KEY_DLEFT", TasButton::BUTTON_LEFT},
    {"KEY_DUP", TasButton::BUTTON_UP},
    {"KEY_DRIGHT", TasButton::BUTTON_RIGHT},
    {"KEY_DDOWN", TasButton::BUTTON_DOWN},
    {"KEY_SL", TasButton::BUTTON_SL},
    {"KEY_SR", TasButton::BUTTON_SR},
    // These buttons are disabled to avoid TAS input from activating hotkeys
    // {"KEY_CAPTURE", TasButton::BUTTON_CAPTURE},
    // {"KEY_HOME", TasButton::BUTTON_HOME},
    {"KEY_ZL", TasButton::TRIGGER_ZL},
    {"KEY_ZR", TasButton::TRIGGER_ZR},
};

Tas::Tas(std::string input_engine_)
    : InputEngine(std::move(input_engine_)), idle_benchmark_frames{[] {
          const char* const value = std::getenv("EDEN_NX2_TAS_IDLE_FRAMES");
          if (value == nullptr) {
              return size_t{0};
          }
          const std::string_view text{value};
          const auto* const end = text.data() + text.size();
          size_t frames{};
          const auto parsed = std::from_chars(text.data(), end, frames);
          return parsed.ec == std::errc{} && parsed.ptr == end && frames <= 100'000 ? frames
                                                                                 : size_t{0};
      }()},
      benchmark_experiment{Common::GpuBenchmarkTrial::ConfiguredExperiment()},
      route_file{RouteFileFromEnvironment("EDEN_NX2_TAS_ROUTE")},
      recording_file{RouteFileFromEnvironment("EDEN_NX2_TAS_RECORD_ROUTE")} {
    for (size_t player_index = 0; player_index < PLAYER_NUMBER; player_index++) {
        PadIdentifier identifier{
            .guid = Common::UUID{},
            .port = player_index,
            .pad = 0,
        };
        PreSetController(identifier);
        for (int button = 0; button < 64; ++button) {
            PreSetButton(identifier, button);
        }
        for (int axis = 0; axis < static_cast<int>(TasAxis::Undefined); ++axis) {
            PreSetAxis(identifier, axis);
        }
    }
    ClearInput();
    if (!IsEnabled()) {
        needs_reset = true;
        return;
    }
    LoadTasFiles();
}

Tas::~Tas() {
    Stop();
    gpu_benchmark.Reset();
}

bool Tas::IsEnabled() const {
    return idle_benchmark_frames != 0 || !route_file.empty() || !recording_file.empty() ||
           Settings::values.tas_enable.GetValue();
}

bool Tas::HasRouteRecordingFile() const {
    return !recording_file.empty();
}

void Tas::LoadTasFiles() {
    if (idle_benchmark_frames != 0) {
        for (auto& player : commands) {
            player.clear();
        }
        script_length = idle_benchmark_frames;
        const auto identity = fmt::format("idle:{}", idle_benchmark_frames);
        route_hash = Common::CityHash64(identity.data(), identity.size());
        return;
    }
    script_length = 0;
    std::array<u64, PLAYER_NUMBER> hashes{};
    if (HasRouteRecordingFile()) {
        return;
    }
    if (!route_file.empty()) {
        hashes[0] = LoadTasFile(0, route_file);
        script_length = commands[0].size();
        if (script_length == 0) {
            LOG_ERROR(Input, "TAS replay route has no frames: {}",
                      Common::FS::PathToUTF8String(route_file));
        }
    } else {
        const auto& directory = Common::FS::GetEdenPath(Common::FS::EdenPath::TASDir);
        for (size_t i = 0; i < commands.size(); i++) {
            hashes[i] = LoadTasFile(i, directory / fmt::format("script0-{}.txt", i + 1));
            if (commands[i].size() > script_length) {
                script_length = commands[i].size();
            }
        }
    }
    route_hash = Common::CityHash64(reinterpret_cast<const char*>(hashes.data()), sizeof(hashes));
}

u64 Tas::LoadTasFile(size_t player_index, const std::filesystem::path& path) {
    commands[player_index].clear();

    std::string file = Common::FS::ReadStringFromFile(path, Common::FS::FileType::BinaryFile);
    std::istringstream command_line(file);
    std::string line;
    int frame_no = 0;
    while (std::getline(command_line, line, '\n')) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> seg_list;
        {
            std::istringstream line_stream(line);
            std::string segment;
            while (std::getline(line_stream, segment, ' ')) {
                seg_list.push_back(std::move(segment));
            }
        }

        if (seg_list.size() < 4) {
            continue;
        }

        try {
            const auto num_frames = std::stoi(seg_list[0]);
            while (frame_no < num_frames) {
                commands[player_index].emplace_back();
                frame_no++;
            }
        } catch (const std::invalid_argument&) {
            LOG_ERROR(Input, "Invalid argument: '{}' at command {}", seg_list[0], frame_no);
        } catch (const std::out_of_range&) {
            LOG_ERROR(Input, "Out of range: '{}' at command {}", seg_list[0], frame_no);
        }

        TASCommand command = {
            .buttons = ReadCommandButtons(seg_list[1]),
            .l_axis = ReadCommandAxis(seg_list[2]),
            .r_axis = ReadCommandAxis(seg_list[3]),
        };
        commands[player_index].push_back(command);
        frame_no++;
    }
    LOG_INFO(Input, "TAS file loaded! {} frames", frame_no);
    return Common::CityHash64(file.data(), file.size());
}

bool Tas::WriteTasFile(const std::filesystem::path& path) {
    std::string output_text;
    for (size_t frame = 0; frame < record_commands.size(); frame++) {
        const TASCommand& line = record_commands[frame];
        output_text += fmt::format("{} {} {} {}\n", frame, WriteCommandButtons(line.buttons),
                                   WriteCommandAxis(line.l_axis), WriteCommandAxis(line.r_axis));
    }

    const auto bytes_written =
        Common::FS::WriteStringToFile(path, Common::FS::FileType::BinaryFile, output_text);
    if (bytes_written == output_text.size()) {
        LOG_INFO(Input, "TAS saved {} frames to {}", record_commands.size(),
                 Common::FS::PathToUTF8String(path));
        return true;
    }
    LOG_ERROR(Input, "TAS write to {} failed at {} / {} bytes",
              Common::FS::PathToUTF8String(path), bytes_written, output_text.size());
    return false;
}

void Tas::RecordInput(u64 buttons, TasAnalog left_axis, TasAnalog right_axis) {
    const std::scoped_lock lock{recording_mutex};
    last_input = {
        .buttons = buttons,
        .l_axis = left_axis,
        .r_axis = right_axis,
    };
}

std::tuple<TasState, size_t, std::array<size_t, PLAYER_NUMBER>> Tas::GetStatus() const {
    TasState state;
    std::array<size_t, PLAYER_NUMBER> lengths{0};
    if (is_recording.load(std::memory_order_relaxed)) {
        const std::scoped_lock lock{recording_mutex};
        if (is_recording.load(std::memory_order_relaxed)) {
            lengths[0] = record_commands.size();
            return {TasState::Recording, lengths[0], lengths};
        }
    }

    if (is_running) {
        state = TasState::Running;
    } else {
        state = TasState::Stopped;
    }

    for (size_t i = 0; i < PLAYER_NUMBER; i++) {
        lengths[i] = commands[i].size();
    }
    if (idle_benchmark_frames != 0) {
        lengths[0] = idle_benchmark_frames;
    }

    return {state, current_command, lengths};
}

void Tas::UpdateThread() {
    if (!IsEnabled()) {
        if (is_running) {
            Stop();
        }
        playback_timing.Reset();
        gpu_benchmark.Reset();
        return;
    }

    if (is_recording.load(std::memory_order_relaxed)) {
        const std::scoped_lock lock{recording_mutex};
        if (is_recording.load(std::memory_order_relaxed)) {
            record_commands.push_back(last_input);
        }
    }
    if (needs_reset) {
        if (is_running && current_command != 0) {
            LOG_INFO(Input, "TAS playback interrupted");
        }
        current_command = 0;
        playback_timing.Reset();
        gpu_benchmark.Reset();
        needs_reset = false;
        LoadTasFiles();
        LOG_DEBUG(Input, "tas_reset done");
    }

    if (!is_running) {
        playback_timing.Reset();
        gpu_benchmark.Reset();
        ClearInput();
        return;
    }
    if (current_command < script_length) {
        if (current_command == 0) {
            LOG_INFO(Input, "TAS playback started frames={} route={:016x}", script_length, route_hash);
            if (Common::DkbFrameTrace::Enabled()) {
                playback_timing.Start(PlaybackTiming::Clock::now(), script_length);
            }
            if (benchmark_experiment != GpuBenchmark::Experiment::None &&
                idle_benchmark_frames != 0) {
                last_benchmark_frame = Common::GpuBenchmarkTrial::RenderedFrames();
                if (gpu_benchmark.Start(script_length, benchmark_experiment)) {
                    LOG_INFO(Input, "TAS A/B started experiment={} frames={} warmup={}",
                             BenchmarkName(benchmark_experiment), script_length,
                             GpuBenchmark::WarmupFrames);
                } else {
                    LOG_WARNING(Input, "TAS A/B idle playback is too short or has unequal phases");
                }
            }
        } else {
            RecordPlaybackFrame();
        }
        const size_t frame = current_command++;
        if (idle_benchmark_frames != 0 && frame != 0) {
            return; // The first idle frame already applied neutral inputs.
        }
        for (size_t player_index = 0; player_index < commands.size(); player_index++) {
            TASCommand command{};
            if (frame < commands[player_index].size()) {
                command = commands[player_index][frame];
            }

            ApplyCommand(player_index, command);
        }
    } else {
        if (script_length != 0) {
            RecordPlaybackFrame();
            if (const auto timing = playback_timing.Finish(script_length)) {
                LOG_INFO(Input,
                         "TAS playback completed frames={} route={:016x} elapsed_ns={} p50_ns={} "
                         "p95_ns={} p99_ns={} max_ns={} near_120={} over_50_ms={} over_100_ms={}",
                         script_length, route_hash, timing->elapsed_ns, timing->p50_ns,
                         timing->p95_ns, timing->p99_ns, timing->max_ns, timing->near_120,
                         timing->over_50_ms, timing->over_100_ms);
            } else {
                LOG_INFO(Input, "TAS playback completed frames={} route={:016x}", script_length,
                         route_hash);
            }
        }
        is_running = idle_benchmark_frames == 0 && route_file.empty() &&
                     Settings::values.tas_loop.GetValue();
        LoadTasFiles();
        current_command = 0;
        ClearInput();
    }
}

void Tas::RecordPlaybackFrame() {
    const bool trace = Common::DkbFrameTrace::Enabled();
    if (!trace && benchmark_experiment == GpuBenchmark::Experiment::None) {
        return;
    }
    const auto now = PlaybackTiming::Clock::now();
    if (trace) {
        playback_timing.Frame(now);
    }
    if (!gpu_benchmark.Active()) {
        return;
    }
    const auto rendered = Common::GpuBenchmarkTrial::RenderedFrames();
    if (rendered != last_benchmark_frame + 1) {
        LOG_WARNING(Input, "TAS A/B aborted experiment={} because a frame was not rendered",
                    BenchmarkName(benchmark_experiment));
        gpu_benchmark.Reset();
        return;
    }
    last_benchmark_frame = rendered;
    if (const auto phase = gpu_benchmark.Frame(now)) {
        if (!phase->timing) {
            const char* reason = "the native Vulkan back thread did not acknowledge the phase";
            if (benchmark_experiment == GpuBenchmark::Experiment::CpuPlacement) {
                reason = "CPU placement could not be applied to all seven threads";
            } else if (benchmark_experiment == GpuBenchmark::Experiment::TextureReads ||
                       benchmark_experiment == GpuBenchmark::Experiment::BufferReads ||
                       benchmark_experiment == GpuBenchmark::Experiment::AcquireWaits) {
                reason = "the Vulkan renderer did not acknowledge the phase";
            }
            LOG_WARNING(Input, "TAS A/B aborted experiment={} phase={} because {}",
                        BenchmarkName(benchmark_experiment), phase->index + 1, reason);
            return;
        }
        const auto& timing = *phase->timing;
        LOG_INFO(Input,
                 "TAS A/B phase experiment={} index={} mode={} frames={} "
                 "elapsed_ns={} p50_ns={} p95_ns={} p99_ns={} max_ns={} near_120={} "
                 "over_50_ms={} over_100_ms={} texture_checks_skipped={} buffer_checks_skipped={} "
                 "acquire_waits_shortened={}",
                 BenchmarkName(benchmark_experiment), phase->index + 1,
                 BenchmarkMode(benchmark_experiment, phase->mode), phase->frames,
                 timing.elapsed_ns, timing.p50_ns, timing.p95_ns, timing.p99_ns, timing.max_ns,
                 timing.near_120, timing.over_50_ms, timing.over_100_ms,
                 benchmark_experiment == GpuBenchmark::Experiment::TextureReads
                     ? phase->read_checks_skipped : 0,
                 benchmark_experiment == GpuBenchmark::Experiment::BufferReads
                     ? phase->read_checks_skipped : 0,
                 phase->acquire_waits_shortened);
        if (!gpu_benchmark.Active()) {
            LOG_INFO(Input, "TAS A/B completed experiment={} frames={}",
                     BenchmarkName(benchmark_experiment), script_length);
        }
    }
}

void Tas::ClearInput() {
    for (size_t player_index = 0; player_index < commands.size(); ++player_index) {
        ApplyCommand(player_index, {});
    }
}

void Tas::ApplyCommand(size_t player_index, const TASCommand& command) {
    const PadIdentifier identifier{
        .guid = Common::UUID{},
        .port = player_index,
        .pad = 0,
    };
    for (std::size_t i = 0; i < sizeof(command.buttons) * 8; ++i) {
        const bool button_status = (command.buttons & (1LLU << i)) != 0;
        const int button = static_cast<int>(i);
        if (GetButton(identifier, button) != button_status) {
            SetButton(identifier, button, button_status);
        }
    }
    SetTasAxis(identifier, TasAxis::StickX, command.l_axis.x);
    SetTasAxis(identifier, TasAxis::StickY, command.l_axis.y);
    SetTasAxis(identifier, TasAxis::SubstickX, command.r_axis.x);
    SetTasAxis(identifier, TasAxis::SubstickY, command.r_axis.y);
}

TasAnalog Tas::ReadCommandAxis(const std::string& line) const {
    std::vector<std::string> seg_list;
    {
        std::istringstream line_stream(line);
        std::string segment;
        while (std::getline(line_stream, segment, ';')) {
            seg_list.push_back(std::move(segment));
        }
    }

    if (seg_list.size() < 2) {
        LOG_ERROR(Input, "Invalid axis data: '{}'", line);
        return {};
    }

    try {
        const float x = std::stof(seg_list.at(0)) / 32767.0f;
        const float y = std::stof(seg_list.at(1)) / 32767.0f;
        return {x, y};
    } catch (const std::invalid_argument&) {
        LOG_ERROR(Input, "Invalid argument: '{}'", line);
    } catch (const std::out_of_range&) {
        LOG_ERROR(Input, "Out of range: '{}'", line);
    }
    return {};
}

u64 Tas::ReadCommandButtons(const std::string& line) const {
    std::istringstream button_text(line);
    std::string button_line;
    u64 buttons = 0;
    while (std::getline(button_text, button_line, ';')) {
        for (const auto& [text, tas_button] : text_to_tas_button) {
            if (text == button_line) {
                buttons |= static_cast<u64>(tas_button);
                break;
            }
        }
    }
    return buttons;
}

std::string Tas::WriteCommandButtons(u64 buttons) const {
    std::string returns;
    for (const auto& [text_button, tas_button] : text_to_tas_button) {
        if ((buttons & static_cast<u64>(tas_button)) != 0) {
            returns += fmt::format("{};", text_button);
        }
    }
    return returns.empty() ? "NONE" : returns;
}

std::string Tas::WriteCommandAxis(TasAnalog analog) const {
    return fmt::format("{};{}", analog.x * 32767, analog.y * 32767);
}

void Tas::SetTasAxis(const PadIdentifier& identifier, TasAxis axis, f32 value) {
    if (GetAxis(identifier, static_cast<int>(axis)) != value) {
        SetAxis(identifier, static_cast<int>(axis), value);
    }
}

void Tas::StartStop() {
    if (!IsEnabled() || HasRouteRecordingFile()) {
        return;
    }
    if (is_running) {
        Stop();
    } else {
        if (idle_benchmark_frames != 0 || !route_file.empty()) {
            needs_reset = true;
        }
        is_running = true;
    }
}

void Tas::Stop() {
    if (is_running && current_command != 0) {
        LOG_INFO(Input, "TAS playback interrupted");
    }
    is_running = false;
}

void Tas::Reset() {
    if (!IsEnabled()) {
        return;
    }
    needs_reset = true;
}

bool Tas::Record() {
    if (!IsEnabled() || !route_file.empty() || idle_benchmark_frames != 0) {
        return true;
    }
    const std::scoped_lock lock{recording_mutex};
    const bool recording = !is_recording.load(std::memory_order_relaxed);
    is_recording.store(recording, std::memory_order_relaxed);
    return recording;
}

void Tas::SaveRecording(bool overwrite_file) {
    const std::scoped_lock lock{recording_mutex};
    if (is_recording.load(std::memory_order_relaxed) || record_commands.empty()) {
        return;
    }
    if (HasRouteRecordingFile()) {
        if (WriteTasFile(recording_file)) {
            record_commands.clear();
        }
        return;
    }
    const auto& directory = Common::FS::GetEdenPath(Common::FS::EdenPath::TASDir);
    WriteTasFile(directory / u8"record.txt");
    if (overwrite_file) {
        WriteTasFile(directory / u8"script0-1.txt");
    }
    needs_reset = true;
    record_commands.clear();
}

} // namespace InputCommon::TasInput
