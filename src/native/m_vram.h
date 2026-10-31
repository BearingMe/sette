// Optional D3D12 memory and host-phase report (M14).  Compiled only into the
// `*_timed` executables, so the accepted hosts and their outputs are unchanged.
//
// The report keeps three different numbers apart:
//   * declared/committed allocation sizes tracked by the host;
//   * the process-local DXGI budget and usage sampled at defined points;
//   * host wall phases (startup, load/alloc, recording, submit/wait, readback).
// DXGI `CurrentUsage` is documented as process usage, but it also includes
// driver-side allocations, so it is not expected to equal the tracked sum.
#pragma once
#ifdef M_TIMING

#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vram {

using Microsoft::WRL::ComPtr;

struct Entry {
    std::string name;
    std::string heap;   // default | upload | readback | mixed
    uint64_t bytes = 0;
};

struct MemorySample {
    std::string label;
    bool ok = false;
    uint64_t local_usage = 0, local_budget = 0, local_available = 0, local_reserved = 0;
    uint64_t nonlocal_usage = 0, nonlocal_budget = 0;
};

struct HostPhases {
    double startup_s = 0.0;
    double load_alloc_s = 0.0;
    double record_s = 0.0;
    double execute_wait_s = 0.0;
    double readback_write_s = 0.0;
    double total_s = 0.0;
};

inline std::string Utf8(const wchar_t* text) {
    if (!text) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string out(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), n, nullptr, nullptr);
    return out;
}

inline std::string Escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out.push_back(c);
    }
    return out;
}

class Tracker {
public:
    explicit Tracker(IDXGIAdapter1* adapter) : adapter_(adapter) {
        adapter->GetDesc1(&desc_);
        adapter->QueryInterface(IID_PPV_ARGS(&adapter3_));
        Sample("start");
    }

    bool ok() const { return adapter3_ != nullptr; }

    void Add(const std::string& name, uint64_t bytes, const char* heap) {
        if (!bytes) return;
        entries_.push_back({name, heap ? heap : "unknown", bytes});
        total_bytes_ += bytes;
        if (std::string(heap) == "default") default_bytes_ += bytes;
        else if (std::string(heap) == "upload") upload_bytes_ += bytes;
        else if (std::string(heap) == "readback") readback_bytes_ += bytes;
    }

    // A staging buffer that exists only while one resource is copied in.
    void Transient(uint64_t bytes) {
        transient_peak_ = std::max(transient_peak_, bytes);
    }

    void Sample(const std::string& label) {
        MemorySample sample;
        sample.label = label;
        if (adapter3_) {
            DXGI_QUERY_VIDEO_MEMORY_INFO local{}, nonlocal{};
            if (SUCCEEDED(adapter3_->QueryVideoMemoryInfo(
                    0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
                SUCCEEDED(adapter3_->QueryVideoMemoryInfo(
                    0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal))) {
                sample.ok = true;
                sample.local_usage = local.CurrentUsage;
                sample.local_budget = local.Budget;
                sample.local_available = local.AvailableForReservation;
                sample.local_reserved = local.CurrentReservation;
                sample.nonlocal_usage = nonlocal.CurrentUsage;
                sample.nonlocal_budget = nonlocal.Budget;
            }
        }
        samples_.push_back(std::move(sample));
    }

    void Host(const HostPhases& phases) { host_ = phases; }

    void Write(const std::string& path) {
        std::stable_sort(entries_.begin(), entries_.end(),
                         [](const Entry& a, const Entry& b) { return a.bytes > b.bytes; });
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cannot write " + path);
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"adapter\": {\"name\": \"" << Escape(Utf8(desc_.Description))
            << "\", \"vendor_id\": " << desc_.VendorId
            << ", \"device_id\": " << desc_.DeviceId
            << ", \"dedicated_video_memory\": " << desc_.DedicatedVideoMemory
            << ", \"dedicated_system_memory\": " << desc_.DedicatedSystemMemory
            << ", \"shared_system_memory\": " << desc_.SharedSystemMemory << "},\n";
        out << "  \"tracked\": {\"allocations\": " << entries_.size()
            << ", \"total_bytes\": " << total_bytes_
            << ", \"default_bytes\": " << default_bytes_
            << ", \"upload_bytes\": " << upload_bytes_
            << ", \"readback_bytes\": " << readback_bytes_
            << ", \"transient_peak_bytes\": " << transient_peak_
            << ", \"peak_bound_bytes\": " << (total_bytes_ + transient_peak_)
            << ", \"largest\": [";
        const size_t count = std::min<size_t>(20, entries_.size());
        for (size_t i = 0; i < count; ++i) {
            if (i) out << ", ";
            out << "{\"name\": \"" << Escape(entries_[i].name)
                << "\", \"heap\": \"" << Escape(entries_[i].heap)
                << "\", \"bytes\": " << entries_[i].bytes << "}";
        }
        out << "]},\n";
        out << "  \"dxgi\": [";
        for (size_t i = 0; i < samples_.size(); ++i) {
            const MemorySample& s = samples_[i];
            if (i) out << ", ";
            out << "{\"label\": \"" << Escape(s.label) << "\", \"ok\": "
                << (s.ok ? "true" : "false")
                << ", \"local\": {\"usage\": " << s.local_usage
                << ", \"budget\": " << s.local_budget
                << ", \"available_reservation\": " << s.local_available
                << ", \"currently_reserved\": " << s.local_reserved << "}"
                << ", \"non_local\": {\"usage\": " << s.nonlocal_usage
                << ", \"budget\": " << s.nonlocal_budget << "}}";
        }
        out << "],\n";
        out << "  \"host\": {\"startup_s\": " << host_.startup_s
            << ", \"load_alloc_s\": " << host_.load_alloc_s
            << ", \"record_s\": " << host_.record_s
            << ", \"execute_wait_s\": " << host_.execute_wait_s
            << ", \"readback_write_s\": " << host_.readback_write_s
            << ", \"total_s\": " << host_.total_s << "}\n";
        out << "}\n";
    }

    static double Seconds(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    }

private:
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<IDXGIAdapter3> adapter3_;
    DXGI_ADAPTER_DESC1 desc_{};
    std::vector<Entry> entries_;
    std::vector<MemorySample> samples_;
    HostPhases host_;
    uint64_t total_bytes_ = 0, default_bytes_ = 0, upload_bytes_ = 0, readback_bytes_ = 0;
    uint64_t transient_peak_ = 0;
};

}  // namespace vram

#endif  // M_TIMING
