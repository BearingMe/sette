// M05 B5: one native D3D12 graph of compute dispatches.
//
// `m_image_stage` runs one captured launch, one dispatch, from a text fixture.
// This program runs a *graph*: named GPU-resident buffers, a fixed sequence of
// dispatches that read and write them, and a final readback of the buffers the
// caller asks for.  Intermediate tensors never leave the GPU, so the ten-layer
// chain is the same code path as a single layer.
//
// Manifest (text):
//
//   version 1
//   resource <name> <bytes> [<init-file>]
//   stage <name>
//     shader  <file.cso>
//     threads <x> <y> <z>
//     groups  <x> <y> <z>
//     params  <file> <bytes>
//     srv <reg> <resource>
//     uav <reg> <resource>
//   dump <resource> <offset> <bytes> <file>
//
// Every resource is a DEFAULT-heap buffer viewed as R32_UINT, so f16/f32/e4m3
// payloads are addressed by word and never reinterpreted by the host.  A
// resource without an init file is filled with a position-dependent sentinel:
// two runs with different seeds show exactly which bytes the graph wrote.
//
// usage: m_network_stage --graph graph.txt --out DIR [--shaders DIR] [--seed N]
//        [--timing FILE]        (`m_network_stage_timed` only)

#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

#ifdef M_TIMING
#include "m_timing.h"
#include "m_vram.h"
#endif

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

namespace {

constexpr uint32_t kAlignment = 256;

void Check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char text[256];
        std::snprintf(text, sizeof(text), "%s failed (hr=0x%08lx)", what,
                      static_cast<unsigned long>(hr));
        throw std::runtime_error(text);
    }
}

std::vector<uint8_t> ReadFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

void WriteFile(const fs::path& path, const void* data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.flush();
    if (!out) throw std::runtime_error("incomplete write (check free disk space): " + path.string());
}

uint32_t Align(uint64_t value) {
    return static_cast<uint32_t>((value + kAlignment - 1) & ~uint64_t(kAlignment - 1));
}

// D3D12 placed-resource offsets must be a multiple of 64 KiB.
constexpr uint64_t kPlacement = 65536;
uint64_t AlignPlacement(uint64_t value) {
    return (value + kPlacement - 1) & ~(kPlacement - 1);
}

enum class Lifetime { Persistent, Transient, Dedicated, Dead };

struct Resource {
    std::string name;
    uint64_t bytes = 0;
    std::string init;                 // empty = sentinel fill
    ComPtr<ID3D12Resource> gpu;
    // M15-A lifetime inventory (filled by Analyze).
    int first = -1, last = -1;
    bool bound = false, dumped = false, srv = false, uav = false;
    bool both_states_one_stage = false, reads_before_write = false;
    Lifetime lifetime = Lifetime::Transient;
    uint64_t arena_offset = 0, arena_size = 0;
};

struct Binding {
    bool input = true;
    uint32_t reg = 0;
    std::string resource;
};

struct Stage {
    std::string name;
    fs::path shader;
    uint32_t threads[3] = {1, 1, 1};
    uint32_t groups[3] = {1, 1, 1};
    std::vector<uint8_t> params;
    std::vector<Binding> bindings;
    ComPtr<ID3D12RootSignature> signature;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> paramUpload;
};

struct Dump {
    std::string resource;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    fs::path file;
};

struct Graph {
    fs::path directory;
    std::vector<Resource> resources;
    std::vector<Stage> stages;
    std::vector<Dump> dumps;
    size_t dropped_dumps = 0;

    Resource& Find(const std::string& name) {
        for (Resource& r : resources)
            if (r.name == name) return r;
        throw std::runtime_error("unknown resource " + name);
    }
};

struct ArenaPlan {
    bool enabled = false;
    uint64_t heap_bytes = 0;
    uint64_t persistent_bytes = 0, dedicated_bytes = 0;
    uint64_t peak_live_bytes = 0, peak_transient_bytes = 0;
    uint64_t largest_transient_bytes = 0;
    std::string largest_transient_name;
    size_t placed = 0, persistent = 0, dedicated = 0, dead = 0, slots = 0;
};

struct BarrierCounts {
    uint64_t transition = 0, uav = 0, aliasing = 0;
};

Graph Parse(const fs::path& manifest) {
    Graph g;
    g.directory = manifest.parent_path();
    std::ifstream in(manifest);
    if (!in) throw std::runtime_error("cannot open " + manifest.string());
    std::string line;
    int version = 0;
    Stage* stage = nullptr;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::string key;
        if (!(words >> key) || key.empty() || key[0] == '#') continue;
        if (key == "version") {
            words >> version;
        } else if (key == "resource") {
            Resource r;
            words >> r.name >> r.bytes;
            if (r.bytes == 0 || (r.bytes % 4)) throw std::runtime_error("resource size");
            words >> r.init;
            if (!r.init.empty()) r.init = (g.directory / r.init).string();
            g.resources.push_back(std::move(r));
            stage = nullptr;
        } else if (key == "stage") {
            Stage s;
            words >> s.name;
            g.stages.push_back(std::move(s));
            stage = &g.stages.back();
        } else if (key == "dump") {
            Dump d;
            words >> d.resource >> d.offset >> d.bytes >> d.file;
            g.dumps.push_back(std::move(d));
        } else if (!stage) {
            throw std::runtime_error("key outside a stage: " + key);
        } else if (key == "shader") {
            std::string name;
            words >> name;
            stage->shader = fs::path(name).is_absolute() ? fs::path(name)
                                                         : g.directory / name;
        } else if (key == "threads") {
            words >> stage->threads[0] >> stage->threads[1] >> stage->threads[2];
        } else if (key == "groups") {
            words >> stage->groups[0] >> stage->groups[1] >> stage->groups[2];
        } else if (key == "params") {
            std::string name;
            uint64_t bytes = 0;
            words >> name >> bytes;
            stage->params = ReadFile(g.directory / name);
            if (stage->params.size() != bytes)
                throw std::runtime_error("params size mismatch in " + manifest.string());
        } else if (key == "srv" || key == "uav") {
            Binding b;
            b.input = key == "srv";
            words >> b.reg >> b.resource;
            stage->bindings.push_back(std::move(b));
        } else {
            throw std::runtime_error("unknown key " + key);
        }
    }
    if (version != 1) throw std::runtime_error("unsupported graph version");
    if (g.stages.empty()) throw std::runtime_error("graph has no stages");
    return g;
}

// Production keeps only the final tensor: dumps of the last stage's UAVs, one
// per resource, preferring the last declared file (`out-dec1.bin` in the
// accepted chain). Diagnostic keeps every dump.
void SelectDumps(Graph& g, const std::string& mode) {
    if (mode != "production") return;
    std::set<std::string> produced;
    for (const Binding& b : g.stages.back().bindings)
        if (!b.input) produced.insert(b.resource);
    std::vector<Dump> kept;
    std::set<std::string> seen;
    for (auto it = g.dumps.rbegin(); it != g.dumps.rend(); ++it)
        if (produced.count(it->resource) && seen.insert(it->resource).second)
            kept.push_back(*it);
    if (kept.empty() && !g.dumps.empty()) kept.push_back(g.dumps.back());
    std::reverse(kept.begin(), kept.end());
    g.dropped_dumps = g.dumps.size() - kept.size();
    g.dumps = std::move(kept);
}

// First/last use, initialization, access flags and diagnostic classification.
// A dump is a use after the last dispatch, so a dumped tensor is live to the end
// of the frame; that is what makes aliasing conservative.
void Analyze(Graph& g) {
    const int stages = static_cast<int>(g.stages.size());
    std::map<std::string, size_t> index;
    for (size_t i = 0; i < g.resources.size(); ++i) index[g.resources[i].name] = i;
    std::vector<std::vector<std::pair<std::string, int>>> states(stages);
    for (int s = 0; s < stages; ++s) {
        for (const Binding& b : g.stages[s].bindings) {
            auto it = index.find(b.resource);
            if (it == index.end())
                throw std::runtime_error("unknown resource " + b.resource);
            Resource& r = g.resources[it->second];
            r.bound = true;
            r.srv |= b.input;
            r.uav |= !b.input;
            r.first = r.first < 0 ? s : std::min(r.first, s);
            r.last = std::max(r.last, s);
            states[s].push_back({b.resource, b.input ? 0 : 1});
        }
    }
    for (int s = 0; s < stages; ++s) {
        std::map<std::string, int> mask;
        for (const auto& entry : states[s]) mask[entry.first] |= 1 << entry.second;
        for (const auto& entry : mask)
            if (entry.second == 3) g.resources[index[entry.first]].both_states_one_stage = true;
    }
    for (const Dump& d : g.dumps) {
        auto it = index.find(d.resource);
        if (it == index.end()) throw std::runtime_error("dump of unknown resource " + d.resource);
        Resource& r = g.resources[it->second];
        r.dumped = true;
        r.last = std::max(r.last, stages);
    }
    for (Resource& r : g.resources) {
        if (!r.bound && !r.dumped) {
            // Unbound inputs are placeholders left by the per-layer fixtures;
            // the chain rewires them to the producer. Do not upload them.
            r.lifetime = Lifetime::Dead;
        } else if (!r.init.empty()) {
            r.lifetime = Lifetime::Persistent;
            r.first = 0;
            if (r.last < 0) r.last = 0;
        } else if (!r.bound) {
            throw std::runtime_error("dumped resource is never bound or initialized: " + r.name);
        }
        if (r.lifetime == Lifetime::Transient && r.first >= 0) {
            // First use is a write iff the first stage binds it as a UAV.
            // Binding it as both states in one stage is rejected separately.
            bool writes = false;
            for (const Binding& b : g.stages[r.first].bindings)
                if (b.resource == r.name && !b.input) writes = true;
            r.reads_before_write = !writes;
        }
    }
}

struct Range {
    uint64_t offset = 0, size = 0;
};

void AddFree(std::vector<Range>& free, uint64_t offset, uint64_t size) {
    if (!size) return;
    free.push_back({offset, size});
    std::sort(free.begin(), free.end(),
              [](const Range& a, const Range& b) { return a.offset < b.offset; });
    std::vector<Range> merged;
    for (const Range& r : free) {
        if (!merged.empty() && r.offset <= merged.back().offset + merged.back().size)
            merged.back().size = std::max(merged.back().offset + merged.back().size,
                                          r.offset + r.size) - merged.back().offset;
        else
            merged.push_back(r);
    }
    free = std::move(merged);
}

// Static best-fit interval allocation over one DEFAULT heap. Resources are
// allocated at their first use and freed after their last use; nested lifetimes
// never share memory. This is deliberately not a general-purpose allocator.
void PlanArena(Graph& g, ArenaPlan& plan) {
    const int stages = static_cast<int>(g.stages.size());
    std::vector<uint64_t> live(stages + 1, 0), transient(stages + 1, 0);
    for (const Resource& r : g.resources) {
        if (r.lifetime == Lifetime::Dead || r.first < 0) continue;
        for (int i = r.first; i <= r.last && i <= stages; ++i) {
            live[i] += r.bytes;
            if (r.lifetime != Lifetime::Persistent) transient[i] += r.bytes;
        }
    }
    for (uint64_t v : live) plan.peak_live_bytes = std::max(plan.peak_live_bytes, v);
    for (uint64_t v : transient) plan.peak_transient_bytes = std::max(plan.peak_transient_bytes, v);
    if (!plan.enabled) return;

    std::vector<std::vector<Resource*>> starting(stages + 1), ending(stages + 1);
    for (Resource& r : g.resources) {
        if (r.lifetime == Lifetime::Persistent) {
            r.arena_size = AlignPlacement(r.bytes);
            plan.persistent_bytes += r.arena_size;
            ++plan.persistent;
        } else if (r.lifetime == Lifetime::Dead) {
            ++plan.dead;
        } else if (!(r.uav && !r.reads_before_write && !r.both_states_one_stage)) {
            r.lifetime = Lifetime::Dedicated;
            r.arena_size = AlignPlacement(r.bytes);
            plan.dedicated_bytes += r.arena_size;
            ++plan.dedicated;
        } else {
            r.arena_size = AlignPlacement(r.bytes);
            if (r.bytes > plan.largest_transient_bytes) {
                plan.largest_transient_bytes = r.bytes;
                plan.largest_transient_name = r.name;
            }
            starting[r.first].push_back(&r);
            ending[r.last].push_back(&r);
        }
    }
    std::vector<Range> free;
    uint64_t high = 0;
    for (int i = 0; i <= stages; ++i) {
        if (i > 0)
            for (Resource* r : ending[i - 1]) AddFree(free, r->arena_offset, r->arena_size);
        auto& batch = starting[i];
        std::sort(batch.begin(), batch.end(), [](const Resource* a, const Resource* b) {
            if (a->arena_size != b->arena_size) return a->arena_size > b->arena_size;
            return a->name < b->name;
        });
        for (Resource* r : batch) {
            int best = -1;
            for (size_t k = 0; k < free.size(); ++k)
                if (free[k].size >= r->arena_size &&
                    (best < 0 || free[k].size < free[best].size ||
                     (free[k].size == free[best].size &&
                      free[k].offset < free[best].offset)))
                    best = static_cast<int>(k);
            if (best >= 0) {
                r->arena_offset = free[best].offset;
                free[best].offset += r->arena_size;
                free[best].size -= r->arena_size;
                if (!free[best].size) free.erase(free.begin() + best);
            } else {
                r->arena_offset = high;
                high += r->arena_size;
                ++plan.slots;
            }
            ++plan.placed;
        }
    }
    plan.heap_bytes = AlignPlacement(high);
}

const char* LifetimeName(Lifetime lifetime) {
    switch (lifetime) {
        case Lifetime::Persistent: return "persistent";
        case Lifetime::Transient: return "transient";
        case Lifetime::Dedicated: return "dedicated";
        case Lifetime::Dead: return "dead";
    }
    return "unknown";
}

void WriteArenaReport(const std::string& path, const Graph& g, const ArenaPlan& plan,
                      const std::string& mode, int frames, const BarrierCounts& barriers) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    uint64_t declared = 0;
    for (const Resource& r : g.resources) declared += r.bytes;
    out << "{\n";
    out << "  \"mode\": \"" << mode << "\", \"arena_enabled\": "
        << (plan.enabled ? "true" : "false") << ", \"frames\": " << frames << ",\n";
    out << "  \"stage_count\": " << g.stages.size() << ", \"resource_count\": "
        << g.resources.size() << ", \"dumps\": " << g.dumps.size() << ", \"dropped_dumps\": " << g.dropped_dumps
        << ",\n";
    out << "  \"declared_bytes\": " << declared << ",\n";
    out << "  \"arena\": {\"heap_bytes\": " << plan.heap_bytes
        << ", \"persistent_bytes\": " << plan.persistent_bytes
        << ", \"dedicated_bytes\": " << plan.dedicated_bytes
        << ", \"placed_resources\": " << plan.placed
        << ", \"persistent_resources\": " << plan.persistent
        << ", \"dedicated_resources\": " << plan.dedicated
        << ", \"dead_resources\": " << plan.dead
        << ", \"bump_regions\": " << plan.slots
        << ", \"peak_live_bytes\": " << plan.peak_live_bytes
        << ", \"peak_transient_bytes\": " << plan.peak_transient_bytes
        << ", \"largest_transient_bytes\": " << plan.largest_transient_bytes
        << ", \"largest_transient_name\": \"" << plan.largest_transient_name << "\""
        << ", \"utilization\": "
        << (plan.heap_bytes ? double(plan.peak_transient_bytes) / double(plan.heap_bytes) : 0.0)
        << "},\n";
    out << "  \"barriers\": {\"transition\": " << barriers.transition
        << ", \"uav\": " << barriers.uav
        << ", \"aliasing\": " << barriers.aliasing << "},\n";
    out << "  \"resources\": [\n";
    std::vector<const Resource*> sorted;
    for (const Resource& r : g.resources) sorted.push_back(&r);
    std::sort(sorted.begin(), sorted.end(),
              [](const Resource* a, const Resource* b) { return a->name < b->name; });
    for (size_t i = 0; i < sorted.size(); ++i) {
        const Resource& r = *sorted[i];
        out << "    {\"name\": \"" << r.name << "\", \"bytes\": " << r.bytes
            << ", \"first\": " << r.first << ", \"last\": " << r.last
            << ", \"lifetime\": \"" << LifetimeName(r.lifetime) << "\""
            << ", \"offset\": " << r.arena_offset
            << ", \"aligned_bytes\": " << r.arena_size
            << ", \"init\": " << (r.init.empty() ? "false" : "true")
            << ", \"dumped\": " << (r.dumped ? "true" : "false")
            << ", \"srv\": " << (r.srv ? "true" : "false")
            << ", \"uav\": " << (r.uav ? "true" : "false")
            << ", \"both_states_one_stage\": "
            << (r.both_states_one_stage ? "true" : "false")
            << ", \"reads_before_write\": " << (r.reads_before_write ? "true" : "false")
            << "}" << (i + 1 == sorted.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

class Runner {
public:
    Runner() {
        UINT flags = 0;
        const bool wantDebug = GetEnvironmentVariableW(L"M_STAGE_DEBUG", nullptr, 0) > 0;
        ComPtr<ID3D12Debug> debug;
        if (wantDebug) {
            Check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)), "D3D12 debug layer required");
            debug->EnableDebugLayer();
            flags |= DXGI_CREATE_FACTORY_DEBUG;
        }
        ComPtr<IDXGIFactory6> factory;
        Check(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0;
             factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                 IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
             ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                            IID_PPV_ARGS(&device_)))) {
                adapter_ = adapter;
                std::wprintf(L"device: %s (vendor 0x%04x)\n", desc.Description, desc.VendorId);
                break;
            }
        }
        if (!device_) throw std::runtime_error("no D3D12 device at feature level 12_0");
        if (wantDebug) Check(device_.As(&infoQueue_), "ID3D12InfoQueue");

        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        Check(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options,
                                           sizeof(options)),
              "CheckFeatureSupport");
        if (!options.DoublePrecisionFloatShaderOps)
            throw std::runtime_error("device lacks DoublePrecisionFloatShaderOps");

        D3D12_COMMAND_QUEUE_DESC queue{};
        queue.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        Check(device_->CreateCommandQueue(&queue, IID_PPV_ARGS(&queue_)), "CreateCommandQueue");
        Check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                              IID_PPV_ARGS(&allocator_)),
              "CreateCommandAllocator");
        Check(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator_.Get(),
                                         nullptr, IID_PPV_ARGS(&list_)),
              "CreateCommandList");
        Check(list_->Close(), "Close");
        Check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence");
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_) throw std::runtime_error("CreateEventW failed");
    }

    ~Runner() {
        if (event_) CloseHandle(event_);
    }

    struct Options {
        fs::path outputDir;
        uint32_t seed = 0;
        std::string timingPath, vramPath, arenaPath;
        std::string mode = "diagnostic";
        int frames = 1;
        bool arena = false;
        std::chrono::steady_clock::time_point processStart{}, parseDone{};
    };

    struct FramePhases {
        double record = 0.0, wait = 0.0, read = 0.0;
    };

    void Run(Graph& g, const Options& opt) {
        const auto t0 = std::chrono::steady_clock::now();
#ifdef M_TIMING
        if (!opt.vramPath.empty())
            vram_ = std::make_unique<vram::Tracker>(adapter_.Get());
#endif
        ArenaPlan plan;
        plan.enabled = opt.arena;
        PlanArena(g, plan);
        Initialize(g, opt.seed, plan);
        for (Stage& s : g.stages) Prepare(s, g);
        const auto t1 = std::chrono::steady_clock::now();
#ifdef M_TIMING
        if (vram_) vram_->Sample("after_load");
#endif

        FramePhases phases;
        for (int frame = 0; frame < opt.frames; ++frame) {
            phases = ExecuteFrame(g, opt, plan, frame);
            std::printf("frame %d/%d complete\n", frame + 1, opt.frames);
        }

        // Validation messages may arrive asynchronously; check once the whole
        // run has drained, so a multi-frame run is covered as well.
        if (infoQueue_) {
            const UINT64 count = infoQueue_->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; ++i) {
                SIZE_T size = 0;
                Check(infoQueue_->GetMessage(i, nullptr, &size), "InfoQueue size");
                std::vector<uint8_t> bytes(size);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
                Check(infoQueue_->GetMessage(i, message, &size), "InfoQueue message");
                if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
                    message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)
                    throw std::runtime_error(std::string("D3D12 validation: ") + message->pDescription);
                if (message->Severity == D3D12_MESSAGE_SEVERITY_WARNING)
                    std::printf("D3D12 warning: %s\n", message->pDescription);
            }
            std::printf("D3D12 debug: zero error/corruption messages (%llu total)\n", count);
        }
        std::printf("graph: %zu stage(s), %zu resource(s), %s mode%s\n", g.stages.size(),
                    g.resources.size(), opt.mode.c_str(),
                    plan.enabled ? ", arena" : ", committed");
#ifdef M_TIMING
        if (vram_) {
            const auto t4 = std::chrono::steady_clock::now();
            vram_->Sample("after_readback");
            vram::HostPhases report;
            report.startup_s = Seconds(opt.processStart, opt.parseDone);
            report.load_alloc_s = Seconds(t0, t1);
            report.record_s = phases.record;
            report.execute_wait_s = phases.wait;
            report.readback_write_s = phases.read;
            report.total_s = Seconds(opt.processStart, t4);
            vram_->Host(report);
            vram_->Write(opt.vramPath);
        }
#endif
        WriteArenaReport(opt.arenaPath, g, plan, opt.mode, opt.frames, barriers_);
    }

private:
    FramePhases ExecuteFrame(Graph& g, const Options& opt, const ArenaPlan& plan, int frame) {
        FramePhases phases;
        barriers_ = {};
        const auto f0 = std::chrono::steady_clock::now();
        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), nullptr), "Reset(list)");

#ifdef M_TIMING
        std::unique_ptr<timing::GpuTimer> timer;
        if (!opt.timingPath.empty())
            timer = std::make_unique<timing::GpuTimer>(device_.Get(), queue_.Get(),
                                                       g.stages.size() + 2);
        if (timer) timer->BeginGlobal(list_.Get());
#endif
        for (Stage& s : g.stages) {
#ifdef M_TIMING
            if (timer) timer->Begin(list_.Get());
#endif
            // Reusing arena memory for a different logical resource is a D3D12
            // aliasing transition; a global aliasing barrier is valid for every
            // switch and keeps the conservative schedule honest.
            if (plan.enabled) {
                D3D12_RESOURCE_BARRIER alias{};
                alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                alias.Aliasing.pResourceBefore = nullptr;
                alias.Aliasing.pResourceAfter = nullptr;
                list_->ResourceBarrier(1, &alias);
                ++barriers_.aliasing;
            }
            ID3D12DescriptorHeap* heaps[] = {s.heap.Get()};
            list_->SetDescriptorHeaps(1, heaps);
            list_->SetPipelineState(s.pipeline.Get());
            list_->SetComputeRootSignature(s.signature.Get());
            list_->SetComputeRootConstantBufferView(0, s.paramUpload->GetGPUVirtualAddress());
            // A resource bound as a UAV is never also read through an SRV in
            // this graph, so the state of each resource is unambiguous.
            for (const Binding& b : s.bindings) {
                Resource& r = g.Find(b.resource);
                Transition(r, b.input ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                      : D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            const uint32_t srvCount = Count(s, true);
            const UINT stride = device_->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_GPU_DESCRIPTOR_HANDLE base = s.heap->GetGPUDescriptorHandleForHeapStart();
            list_->SetComputeRootDescriptorTable(1, base);
            D3D12_GPU_DESCRIPTOR_HANDLE uav = base;
            uav.ptr += SIZE_T(srvCount) * stride;
            list_->SetComputeRootDescriptorTable(2, uav);
            // every stage reads what an earlier stage wrote
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = nullptr;
            list_->ResourceBarrier(1, &barrier);
            ++barriers_.uav;
#ifdef M_TIMING
            if (timer) timer->Begin(list_.Get());
#endif
            list_->Dispatch(s.groups[0], s.groups[1], s.groups[2]);
#ifdef M_TIMING
            if (timer) timer->End(list_.Get(), s.name);
#endif
        }

#ifdef M_TIMING
        if (timer) timer->EndGlobal(list_.Get());
#endif
        // readback of the requested ranges
        std::vector<ComPtr<ID3D12Resource>> readbacks;
        for (Dump& d : g.dumps) {
            Resource& r = g.Find(d.resource);
            if (d.offset + d.bytes > r.bytes)
                throw std::runtime_error("dump exceeds " + d.resource);
            Transition(r, D3D12_RESOURCE_STATE_COPY_SOURCE);
            ComPtr<ID3D12Resource> readback = Readback(Align(d.bytes));
            list_->CopyBufferRegion(readback.Get(), 0, r.gpu.Get(), d.offset, d.bytes);
            readbacks.push_back(readback);
#ifdef M_TIMING
            if (vram_ && frame == 0)
                vram_->Add("readback:" + d.file.string(), Align(d.bytes), "readback");
#endif
        }
#ifdef M_TIMING
        if (timer) timer->Resolve(list_.Get());
        if (vram_) vram_->Sample("after_alloc");
#endif
        Check(list_->Close(), "Close(list)");
        const auto t2 = std::chrono::steady_clock::now();
        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion");
        WaitForSingleObject(event_, INFINITE);
        const auto t3 = std::chrono::steady_clock::now();

#ifdef M_TIMING
        if (timer && frame + 1 == opt.frames) timer->Write(opt.timingPath);
        if (vram_) vram_->Sample("after_execute");
#endif

        const fs::path directory =
            opt.frames > 1 ? opt.outputDir / ("frame-" + std::to_string(frame)) : opt.outputDir;
        fs::create_directories(directory);
        for (size_t i = 0; i < g.dumps.size(); ++i) {
            void* mapped = nullptr;
            Check(readbacks[i]->Map(0, nullptr, &mapped), "Map(readback)");
            WriteFile(directory / g.dumps[i].file, mapped, g.dumps[i].bytes);
            readbacks[i]->Unmap(0, nullptr);
            std::printf("wrote %s (%llu bytes)\n", g.dumps[i].file.string().c_str(),
                        static_cast<unsigned long long>(g.dumps[i].bytes));
        }
        const auto t4 = std::chrono::steady_clock::now();
        phases.record = Seconds(f0, t2);
        phases.wait = Seconds(t2, t3);
        phases.read = Seconds(t3, t4);
        return phases;
    }

private:
    static uint32_t Count(const Stage& s, bool input) {
        uint32_t n = 0;
        for (const Binding& b : s.bindings)
            if (b.input == input) ++n;
        return n;
    }

    void TransitionRaw(ID3D12Resource* key, D3D12_RESOURCE_STATES to) {
        auto it = states_.find(key);
        D3D12_RESOURCE_STATES from =
            it == states_.end() ? D3D12_RESOURCE_STATE_COMMON : it->second;
        if (from == to) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = key;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = from;
        barrier.Transition.StateAfter = to;
        list_->ResourceBarrier(1, &barrier);
        ++barriers_.transition;
        states_[key] = to;
    }

    void Transition(Resource& r, D3D12_RESOURCE_STATES to) {
        if (!r.gpu) throw std::runtime_error("transition of unallocated resource " + r.name);
        TransitionRaw(r.gpu.Get(), to);
    }

    static D3D12_RESOURCE_DESC BufferDesc(uint64_t bytes) {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        return desc;
    }

    static double Seconds(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    }

    // Uploads are batched: one submit per 64 MiB instead of one per resource.
    static constexpr uint64_t kUploadBatch = 64ull << 20;

    void CopyIn(ID3D12Resource* dest, uint64_t offset, const void* data, uint64_t bytes) {
        if (!bytes) return;
        D3D12_HEAP_PROPERTIES up{};
        up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC desc = BufferDesc(Align(bytes));
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> upload;
        Check(device_->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&upload)),
              "CreateCommittedResource(upload)");
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "Map(upload)");
        std::memcpy(mapped, data, static_cast<size_t>(bytes));
        upload->Unmap(0, nullptr);
        TransitionRaw(dest, D3D12_RESOURCE_STATE_COPY_DEST);
        list_->CopyBufferRegion(dest, offset, upload.Get(), 0, bytes);
        uploads_.push_back(upload);
        pending_upload_ += Align(bytes);
#ifdef M_TIMING
        if (vram_) vram_->Transient(pending_upload_);
#endif
        if (pending_upload_ >= kUploadBatch) FlushUploads();
    }

    void FlushUploads() {
        if (uploads_.empty()) return;
        Check(list_->Close(), "Close(init batch)");
        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion");
        WaitForSingleObject(event_, INFINITE);
        uploads_.clear();
        pending_upload_ = 0;
        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), nullptr), "Reset(list)");
    }

    void Initialize(Graph& g, uint32_t seed, const ArenaPlan& plan) {
        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), nullptr), "Reset(list)");
        if (plan.enabled && plan.heap_bytes) {
            D3D12_HEAP_DESC hd{};
            hd.SizeInBytes = plan.heap_bytes;
            hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            hd.Properties.CreationNodeMask = 1;
            hd.Properties.VisibleNodeMask = 1;
            hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
            Check(device_->CreateHeap(&hd, IID_PPV_ARGS(&arena_heap_)), "CreateHeap");
#ifdef M_TIMING
            if (vram_) vram_->Add("arena-heap", plan.heap_bytes, "default");
#endif
            // One resource spans the heap so a single seed-dependent sentinel
            // fill covers every transient slot. An unwritten read is visible.
            D3D12_RESOURCE_DESC canary = BufferDesc(plan.heap_bytes);
            Check(device_->CreatePlacedResource(arena_heap_.Get(), 0, &canary,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&arena_canary_)),
                  "CreatePlacedResource(canary)");
            std::vector<uint8_t> chunk(8u << 20);
            for (uint64_t offset = 0; offset < plan.heap_bytes; offset += chunk.size()) {
                const uint64_t n = std::min<uint64_t>(chunk.size(), plan.heap_bytes - offset);
                for (uint64_t i = 0; i < n; ++i)
                    chunk[i] = static_cast<uint8_t>(
                        ((offset + i) * 37u + 11u + seed * 97u) & 0xFFu);
                CopyIn(arena_canary_.Get(), offset, chunk.data(), n);
            }
            TransitionRaw(arena_canary_.Get(), D3D12_RESOURCE_STATE_COMMON);
        }
        for (Resource& r : g.resources) {
            if (r.lifetime == Lifetime::Dead) {
                r.gpu.Reset();
                continue;
            }
            if (plan.enabled && r.lifetime == Lifetime::Transient) {
                D3D12_RESOURCE_DESC desc = BufferDesc(r.arena_size);
                Check(device_->CreatePlacedResource(arena_heap_.Get(), r.arena_offset, &desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&r.gpu)),
                      "CreatePlacedResource");
                continue;
            }
            // Persistent and dedicated resources, and every resource when the
            // arena is off, are committed: contents survive across frames.
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC desc = BufferDesc(r.bytes);
            Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&r.gpu)),
                  "CreateCommittedResource");
#ifdef M_TIMING
            if (vram_) {
                const char* heap_name = r.lifetime == Lifetime::Persistent ? "persistent"
                                       : r.lifetime == Lifetime::Dedicated ? "dedicated"
                                                                           : "default";
                vram_->Add(r.name, r.bytes, heap_name);
            }
#endif
            std::vector<uint8_t> init(static_cast<size_t>(r.bytes));
            if (!r.init.empty()) {
                init = ReadFile(fs::path(r.init));
                if (init.size() != r.bytes) throw std::runtime_error("init size for " + r.name);
            } else {
                for (size_t i = 0; i < init.size(); ++i)
                    init[i] = static_cast<uint8_t>((i * 37u + 11u + seed * 97u) & 0xFFu);
            }
            CopyIn(r.gpu.Get(), 0, init.data(), r.bytes);
            Transition(r, D3D12_RESOURCE_STATE_COMMON);
        }
        FlushUploads();
        // Execute any trailing transition even if the last CopyIn triggered a
        // flush; otherwise a persistent resource would be left in COPY_DEST
        // while the host believes it is COMMON.
        Check(list_->Close(), "Close(init)");
        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion");
        WaitForSingleObject(event_, INFINITE);
    }

    void Prepare(Stage& s, Graph& g) {
        uint32_t srv = 0, uav = 0;
        for (const Binding& b : s.bindings) {
            const uint32_t expected = b.input ? srv++ : uav++;
            if (b.reg != expected)
                throw std::runtime_error("bindings must be declared in register order: " + s.name);
        }
        auto code = ReadFile(s.shader);
        const uint32_t srvCount = Count(s, true), uavCount = Count(s, false);
        s.signature = MakeRootSignature(srvCount, uavCount);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = s.signature.Get();
        pso.CS.pShaderBytecode = code.data();
        pso.CS.BytecodeLength = code.size();
        Check(device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&s.pipeline)),
              "CreateComputePipelineState");

        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = srvCount + uavCount;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s.heap)),
              "CreateDescriptorHeap");
        const UINT stride = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = s.heap->GetCPUDescriptorHandleForHeapStart();
        srv = 0; uav = 0;
        for (const Binding& b : s.bindings) {
            Resource& r = g.Find(b.resource);
            const uint32_t at = b.input ? srv++ : srvCount + uav++;
            D3D12_CPU_DESCRIPTOR_HANDLE handle{cpu.ptr + SIZE_T(at) * stride};
            if (b.input) {
                D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                view.Format = DXGI_FORMAT_R32_UINT;
                view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                view.Buffer.NumElements = static_cast<UINT>(r.bytes / 4);
                device_->CreateShaderResourceView(r.gpu.Get(), &view, handle);
            } else {
                D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                view.Format = DXGI_FORMAT_R32_UINT;
                view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                view.Buffer.NumElements = static_cast<UINT>(r.bytes / 4);
                device_->CreateUnorderedAccessView(r.gpu.Get(), nullptr, &view, handle);
            }
        }

        D3D12_HEAP_PROPERTIES up{};
        up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = Align(s.params.size());
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Check(device_->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&s.paramUpload)),
              "CreateCommittedResource(params)");
        void* mapped = nullptr;
        Check(s.paramUpload->Map(0, nullptr, &mapped), "Map(params)");
        std::memset(mapped, 0, Align(s.params.size()));
        std::memcpy(mapped, s.params.data(), s.params.size());
        s.paramUpload->Unmap(0, nullptr);
    }

    ComPtr<ID3D12RootSignature> MakeRootSignature(uint32_t srvCount, uint32_t uavCount) {
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = srvCount;
        ranges[0].BaseShaderRegister = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = uavCount;
        ranges[1].BaseShaderRegister = 0;
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor.ShaderRegister = 0;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = srvCount ? 1u : 0u;
        params[1].DescriptorTable.pDescriptorRanges = &ranges[0];
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable.NumDescriptorRanges = uavCount ? 1u : 0u;
        params[2].DescriptorTable.pDescriptorRanges = &ranges[1];
        D3D12_ROOT_SIGNATURE_DESC desc{};
        desc.NumParameters = 3;
        desc.pParameters = params;
        ComPtr<ID3DBlob> signature, error;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature,
                                                 &error);
        if (FAILED(hr) && error)
            std::fprintf(stderr, "%s\n", static_cast<const char*>(error->GetBufferPointer()));
        Check(hr, "D3D12SerializeRootSignature");
        ComPtr<ID3D12RootSignature> result;
        Check(device_->CreateRootSignature(0, signature->GetBufferPointer(),
                                           signature->GetBufferSize(), IID_PPV_ARGS(&result)),
              "CreateRootSignature");
        return result;
    }

    ComPtr<ID3D12Resource> Readback(uint64_t size) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> result;
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&result)),
              "CreateCommittedResource(readback)");
        return result;
    }

    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12InfoQueue> infoQueue_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    uint64_t fenceValue_ = 0;
    std::map<ID3D12Resource*, D3D12_RESOURCE_STATES> states_;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Heap> arena_heap_;
    ComPtr<ID3D12Resource> arena_canary_;
    BarrierCounts barriers_;
    std::vector<ComPtr<ID3D12Resource>> uploads_;
    uint64_t pending_upload_ = 0;
#ifdef M_TIMING
    std::unique_ptr<vram::Tracker> vram_;
#endif
};

int Run(int argc, char** argv) {
    const auto processStart = std::chrono::steady_clock::now();
    fs::path manifest, outputDir = ".", shaderDir;
    std::string timingPath, vramPath, arenaPath, mode = "diagnostic";
    uint32_t seed = 0;
    int frames = 1;
    bool arena = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--graph" && i + 1 < argc) manifest = argv[++i];
        else if (arg == "--out" && i + 1 < argc) outputDir = argv[++i];
        else if (arg == "--shaders" && i + 1 < argc) shaderDir = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--frames" && i + 1 < argc) frames = std::stoi(argv[++i]);
        else if (arg == "--arena") arena = true;
        else if (arg == "--arena-report" && i + 1 < argc) arenaPath = argv[++i];
#ifdef M_TIMING
        else if (arg == "--timing" && i + 1 < argc) timingPath = argv[++i];
        else if (arg == "--vram" && i + 1 < argc) vramPath = argv[++i];
#endif
        else throw std::runtime_error("unknown option " + arg);
    }
    if (mode != "production" && mode != "diagnostic")
        throw std::runtime_error("--mode must be production or diagnostic");
    if (frames < 1) throw std::runtime_error("--frames must be at least 1");
    if (manifest.empty()) {
        std::fprintf(stderr, "usage: m_network_stage --graph graph.txt [--out DIR] "
                             "[--shaders DIR] [--seed N] [--mode production|diagnostic] "
                             "[--arena] [--frames N] [--arena-report FILE]\n");
        return 2;
    }
    Graph graph = Parse(manifest);
    // shader paths that are just a file name resolve against --shaders
    for (Stage& s : graph.stages)
        if (!fs::exists(s.shader) && !shaderDir.empty()) s.shader = shaderDir / s.shader.filename();
    SelectDumps(graph, mode);
    Analyze(graph);
    const auto parseDone = std::chrono::steady_clock::now();
    Runner runner;
    Runner::Options options;
    options.outputDir = outputDir;
    options.seed = seed;
    options.timingPath = timingPath;
    options.vramPath = vramPath;
    options.arenaPath = arenaPath;
    options.mode = mode;
    options.frames = frames;
    options.arena = arena;
    options.processStart = processStart;
    options.parseDone = parseDone;
    runner.Run(graph, options);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
