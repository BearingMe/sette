// M15-C: persistent-engine harness.
//
// Initializes one engine from a composed pipeline manifest, uploads the frame
// colour inputs once, creates one caller-owned output texture, then executes a
// sequence of frames in one process.  Only descriptor views and external
// resource states change between frames; no graph, pipeline, weight, descriptor
// heap or arena is rebuilt.
//
// Validation mode adds the readbacks needed to compare each frame with the
// accepted standalone runner.  Production mode performs none, so its steady
// state is the inference path itself.
//
// usage:
//   m_inference_stage --manifest engine.txt --shaders DIR \
//     --input enc0_in_colour=frame-00.bin [--input ...] \
//     --output frame_out [--frames N] [--mode production|validation] \
//     [--dump-dir DIR] [--dump-frame 0] [--output-dump frame_out] \
//     [--recover] [--report FILE] [--debug] [--seed N]
#include "m_inference_engine.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using sette::PersistentEngine;

namespace {

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

double Seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct InputSpec {
    std::string resource;
    fs::path file;
};

struct UploadedTexture {
    ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct Options {
    fs::path manifest, shaderDir, report, dumpDir;
    std::vector<InputSpec> inputs;
    std::string output;
    std::string outputDump = "frame_out";
    fs::path callerReadback;
    int frames = 0;
    int dumpFrame = 0;
    bool validation = false;
    bool recover = false;
    bool debug = false;
    bool callerList = false;
    uint32_t seed = 0;
};

struct GpuMemory {
    uint64_t usage = 0, budget = 0;
};

GpuMemory SampleMemory(ID3D12Device* device) {
    GpuMemory out;
    if (!device) return out;
    ComPtr<IDXGIDevice> dxgiDevice;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice)))) return out;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgiDevice->GetAdapter(&adapter))) return out;
    ComPtr<IDXGIAdapter3> adapter3;
    if (FAILED(adapter.As(&adapter3))) return out;
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        out.usage = info.CurrentUsage;
        out.budget = info.Budget;
    }
    return out;
}

GpuMemory SampleAdapter(IDXGIAdapter1* adapter) {
    GpuMemory out;
    ComPtr<IDXGIAdapter3> adapter3;
    if (!adapter || FAILED(adapter->QueryInterface(IID_PPV_ARGS(&adapter3)))) return out;
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        out.usage = info.CurrentUsage;
        out.budget = info.Budget;
    }
    return out;
}

ComPtr<ID3D12Resource> MakeTexture(ID3D12Device* device, DXGI_FORMAT format,
                                   uint32_t width, uint32_t height) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                          IID_PPV_ARGS(&resource)),
          "CreateCommittedResource(texture)");
    return resource;
}

// One upload command list for every input texture.  The caller owns these
// textures and their initial state; the engine never creates them.
std::vector<UploadedTexture> UploadInputs(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const std::vector<std::pair<const sette::EngineResource*, InputSpec>>& files) {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEventW failed");
    Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                          IID_PPV_ARGS(&allocator)),
          "CreateCommandAllocator(upload)");
    Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(),
                                    nullptr, IID_PPV_ARGS(&list)),
          "CreateCommandList(upload)");
    Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
          "CreateFence(upload)");
    std::vector<UploadedTexture> textures;
    std::vector<ComPtr<ID3D12Resource>> uploads;
    for (const auto& item : files) {
        const sette::EngineResource* spec = item.first;
        const InputSpec& input = item.second;
        const std::vector<uint8_t> data = ReadFile(input.file);
        const uint32_t pixelBytes =
            static_cast<uint32_t>(spec->bytes / (spec->width * spec->height));
        if (data.size() != spec->bytes)
            throw std::runtime_error("input size mismatch: " + input.file.string());
        UploadedTexture texture;
        texture.resource = MakeTexture(device, spec->format, spec->width, spec->height);
        D3D12_RESOURCE_DESC desc = texture.resource->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
        D3D12_HEAP_PROPERTIES up{};
        up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC ud{};
        ud.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        ud.Width = total;
        ud.Height = 1;
        ud.DepthOrArraySize = 1;
        ud.MipLevels = 1;
        ud.SampleDesc.Count = 1;
        ud.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> upload;
        Check(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &ud,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                              IID_PPV_ARGS(&upload)),
              "CreateCommittedResource(input upload)");
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "Map(input upload)");
        auto* dst = static_cast<uint8_t*>(mapped);
        std::memset(dst, 0, static_cast<size_t>(total));
        for (uint32_t y = 0; y < spec->height; ++y)
            std::memcpy(dst + footprint.Offset + uint64_t(y) * footprint.Footprint.RowPitch,
                        data.data() + uint64_t(y) * uint64_t(spec->width) * pixelBytes,
                        static_cast<size_t>(uint64_t(spec->width) * pixelBytes));
        upload->Unmap(0, nullptr);
        uploads.push_back(upload);

        D3D12_RESOURCE_BARRIER toCopy{};
        toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toCopy.Transition.pResource = texture.resource.Get();
        toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        list->ResourceBarrier(1, &toCopy);
        D3D12_TEXTURE_COPY_LOCATION from{};
        from.pResource = upload.Get();
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION to{};
        to.pResource = texture.resource.Get();
        to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.SubresourceIndex = 0;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        D3D12_RESOURCE_BARRIER toCommon = toCopy;
        toCommon.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        toCommon.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        list->ResourceBarrier(1, &toCommon);
        textures.push_back(std::move(texture));
    }
    Check(list->Close(), "Close(upload)");
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    Check(queue->Signal(fence.Get(), 1), "Signal(upload)");
    Check(fence->SetEventOnCompletion(1, event), "SetEventOnCompletion(upload)");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    return textures;
}

void ReadbackTexture(ID3D12Device* device, ID3D12CommandQueue* queue,
                     ID3D12Resource* texture, uint32_t width, uint32_t height,
                     const fs::path& path) {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEventW failed");
    Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                          IID_PPV_ARGS(&allocator)),
          "CreateCommandAllocator(readback)");
    Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(),
                                    nullptr, IID_PPV_ARGS(&list)),
          "CreateCommandList(readback)");
    Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
          "CreateFence(readback)");
    D3D12_RESOURCE_DESC desc = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 rowBytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = total;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd,
                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                          IID_PPV_ARGS(&readback)),
          "CreateCommittedResource(readback)");
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION to{};
    to.pResource = readback.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = texture;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.SubresourceIndex = 0;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    Check(list->Close(), "Close(readback)");
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    Check(queue->Signal(fence.Get(), 1), "Signal(readback)");
    Check(fence->SetEventOnCompletion(1, event), "SetEventOnCompletion(readback)");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    void* mapped = nullptr;
    Check(readback->Map(0, nullptr, &mapped), "Map(readback)");
    const uint32_t pixel = static_cast<uint32_t>(desc.Width && width
                                                     ? rowBytes / width : 4);
    std::vector<uint8_t> packed(static_cast<size_t>(width) * height * pixel);
    const auto* src = static_cast<const uint8_t*>(mapped) + footprint.Offset;
    for (uint32_t y = 0; y < height; ++y)
        std::memcpy(packed.data() + uint64_t(y) * width * pixel,
                    src + uint64_t(y) * footprint.Footprint.RowPitch,
                    static_cast<size_t>(width) * pixel);
    readback->Unmap(0, nullptr);
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(reinterpret_cast<const char*>(packed.data()),
              static_cast<std::streamsize>(packed.size()));
}

std::string JsonEscape(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}

void WriteReport(const fs::path& path, const Options& opt, const PersistentEngine& engine,
                 const std::vector<sette::PersistentEngine::FrameStats>& runs,
                 const std::vector<std::string>& labels, const GpuMemory& afterInit,
                 const GpuMemory& after) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    const auto& init = engine.initStats();
    const auto& mem = engine.memory();
    out << "{\n";
    out << "  \"manifest\": \"" << JsonEscape(opt.manifest.string()) << "\",\n";
    out << "  \"mode\": \"" << (opt.validation ? "validation" : "production") << "\",\n";
    out << "  \"device\": \"" << JsonEscape(init.deviceName) << "\",\n";
    out << "  \"frames\": " << runs.size() << ",\n";
    out << "  \"init\": {\"total_ms\": " << init.totalMs
        << ", \"parse_ms\": " << init.parseMs
        << ", \"resource_ms\": " << init.resourceMs
        << ", \"pipeline_ms\": " << init.pipelineMs
        << ", \"upload_wait_ms\": " << init.uploadWaitMs
        << ", \"stages\": " << init.stages << ", \"resources\": " << init.resources << "},\n";
    out << "  \"memory\": {\"committed_buffers\": " << mem.committedBuffers
        << ", \"committed_textures\": " << mem.committedTextures
        << ", \"committed_bytes\": " << mem.committedBytes
        << ", \"placed_bytes\": " << mem.placedBytes
        << ", \"external_bytes\": " << mem.externalBytes
        << ", \"arena_heap_bytes\": " << mem.arenaHeapBytes
        << ", \"committed_allocations\": " << mem.committedAllocations
        << ", \"placed_allocations\": " << mem.placedAllocations
        << ", \"descriptor_heaps\": " << mem.descriptorHeaps
        << ", \"descriptor_count\": " << mem.descriptorCount
        << ", \"param_upload_bytes\": " << mem.paramUploadBytes
        << ", \"peak_live_bytes\": " << mem.peakLiveBytes
        << ", \"peak_transient_bytes\": " << mem.peakTransientBytes << "},\n";
    out << "  \"dxgi\": {\"after_init_usage\": " << afterInit.usage
        << ", \"after_init_budget\": " << afterInit.budget
        << ", \"after_execute_usage\": " << after.usage
        << ", \"after_execute_budget\": " << after.budget << "},\n";
    const auto debug = engine.debugCounts();
    out << "  \"debug\": {\"total\": " << debug.total
        << ", \"errors\": " << debug.errors
        << ", \"warnings\": " << debug.warnings << "},\n";
    out << "  \"runs\": [\n";
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        out << "    {\"index\": " << i << ", \"label\": \"" << JsonEscape(labels[i]) << "\""
            << ", \"gpu_ms\": " << r.gpuIntervalMs
            << ", \"gpu_valid\": " << (r.gpuIntervalValid ? "true" : "false")
            << ", \"record_ms\": " << r.recordMs
            << ", \"submit_wait_ms\": " << r.submitWaitMs
            << ", \"readback_ms\": " << r.readbackMs
            << ", \"dispatches\": " << r.dispatches
            << ", \"copies\": " << r.copies
            << ", \"transitions\": " << r.transitions
            << ", \"uav_barriers\": " << r.uavBarriers
            << ", \"aliasing_barriers\": " << r.aliasingBarriers << "}"
            << (i + 1 == runs.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

int Run(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--manifest") opt.manifest = next();
        else if (arg == "--shaders") opt.shaderDir = next();
        else if (arg == "--input") {
            const std::string value = next();
            const size_t eq = value.find('=');
            if (eq == std::string::npos)
                throw std::runtime_error("--input expects NAME=FILE");
            opt.inputs.push_back({value.substr(0, eq), value.substr(eq + 1)});
        } else if (arg == "--output") opt.output = next();
        else if (arg == "--output-dump") opt.outputDump = next();
        else if (arg == "--frames") opt.frames = std::stoi(next());
        else if (arg == "--dump-frame") opt.dumpFrame = std::stoi(next());
        else if (arg == "--dump-dir") opt.dumpDir = next();
        else if (arg == "--report") opt.report = next();
        else if (arg == "--mode") {
            const std::string value = next();
            if (value == "validation") opt.validation = true;
            else if (value == "production") opt.validation = false;
            else throw std::runtime_error("--mode must be production or validation");
        } else if (arg == "--recover") opt.recover = true;
        else if (arg == "--debug") opt.debug = true;
        else if (arg == "--caller-list") opt.callerList = true;
        else if (arg == "--caller-readback") opt.callerReadback = next();
        else if (arg == "--seed") opt.seed = static_cast<uint32_t>(std::stoul(next()));
        else throw std::runtime_error("unknown option " + arg);
    }
    if (opt.manifest.empty()) throw std::runtime_error("--manifest is required");
    if (opt.output.empty()) throw std::runtime_error("--output is required");
    if (opt.inputs.empty()) throw std::runtime_error("at least one --input is required");
    if (opt.frames <= 0) opt.frames = static_cast<int>(opt.inputs.size());
    if (opt.frames > static_cast<int>(opt.inputs.size()))
        throw std::runtime_error("--frames exceeds the number of inputs");
    if (opt.validation && opt.dumpDir.empty())
        throw std::runtime_error("validation mode needs --dump-dir");
    if (!opt.dumpDir.empty()) fs::create_directories(opt.dumpDir);

    PersistentEngine engine;
    PersistentEngine::Config config;
    config.manifest = opt.manifest.string();
    config.shaderDir = opt.shaderDir.string();
    config.mode = opt.validation ? PersistentEngine::Mode::Validation
                                 : PersistentEngine::Mode::Production;
    config.debug = opt.debug;
    config.seed = opt.seed;
    engine.Initialize(config);
    const GpuMemory afterInit = SampleAdapter(engine.adapter());

    // Locate external resource descriptions and create the caller-owned
    // textures before the first frame.
    std::map<std::string, const sette::EngineResource*> specs;
    for (const auto& r : engine.resources())
        if (r.external) specs[r.name] = &r;
    if (!specs.count(opt.output))
        throw std::runtime_error("output resource is not external: " + opt.output);
    std::vector<std::pair<const sette::EngineResource*, InputSpec>> files;
    for (const InputSpec& input : opt.inputs) {
        if (!specs.count(input.resource))
            throw std::runtime_error("input resource is not external: " + input.resource);
        files.push_back({specs[input.resource], input});
    }
    auto inputs = UploadInputs(engine.device(), engine.queue(), files);
    const sette::EngineResource* outSpec = specs[opt.output];
    UploadedTexture output;
    output.resource = MakeTexture(engine.device(), outSpec->format, outSpec->width,
                                  outSpec->height);

    // State every caller-owned resource is guaranteed to be in before its next
    // use.  A resource is COMMON on its first use and left in the state the
    // engine promised afterwards.
    std::vector<D3D12_RESOURCE_STATES> inputState(inputs.size(),
                                                   D3D12_RESOURCE_STATE_COMMON);
    std::vector<D3D12_RESOURCE_STATES> outState(1, D3D12_RESOURCE_STATE_COMMON);
    std::vector<PersistentEngine::FrameStats> runs;
    std::vector<std::string> labels;

    auto execute = [&](size_t index, bool dumpAll, const std::string& label,
                       const std::string& dumpSubdir) {
        PersistentEngine::FrameRequest request;
        request.inputs.push_back({opt.inputs[index].resource,
                                  inputs[index].resource.Get(), inputState[index]});
        request.outputs.push_back({opt.output, output.resource.Get(), outState[0]});
        if (opt.validation) {
            request.dumpDir = (fs::path(opt.dumpDir) / dumpSubdir).string();
            request.dumps = dumpAll ? std::vector<std::string>{}
                                    : std::vector<std::string>{opt.outputDump};
        }
        PersistentEngine::FrameStats stats = engine.ExecuteFrame(request);
        inputState[index] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        outState[0] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        runs.push_back(stats);
        labels.push_back(label);
    };

    auto frameName = [](int index) {
        char text[32];
        std::snprintf(text, sizeof(text), "frame-%02d", index);
        return std::string(text);
    };
    if (!opt.callerList) {
        for (int frame = 0; frame < opt.frames; ++frame)
            execute(static_cast<size_t>(frame), opt.validation && frame == opt.dumpFrame,
                    frameName(frame), frameName(frame));
        if (opt.recover)
            execute(0, false, "recover-frame-0", "recover");
    } else {
        if (opt.validation)
            throw std::runtime_error("--caller-list cannot use engine-owned validation dumps");
        ComPtr<ID3D12CommandAllocator> calloc;
        ComPtr<ID3D12GraphicsCommandList> clist;
        ComPtr<ID3D12Fence> cfence;
        HANDLE cevent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!cevent) throw std::runtime_error("CreateEventW failed");
        Check(engine.device()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                      IID_PPV_ARGS(&calloc)),
              "CreateCommandAllocator(caller)");
        Check(engine.device()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                 calloc.Get(), nullptr,
                                                 IID_PPV_ARGS(&clist)),
              "CreateCommandList(caller)");
        Check(engine.device()->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                           IID_PPV_ARGS(&cfence)),
              "CreateFence(caller)");
        Check(clist->Close(), "Close(caller list init)");
        uint64_t fenceValue = 0;
        auto runBorrowed = [&](size_t index, const std::string& label) {
            PersistentEngine::FrameRequest request;
            request.inputs.push_back({opt.inputs[index].resource,
                                      inputs[index].resource.Get(), inputState[index]});
            request.outputs.push_back({opt.output, output.resource.Get(), outState[0]});
            Check(calloc->Reset(), "Reset(caller allocator)");
            Check(clist->Reset(calloc.Get(), nullptr), "Reset(caller list)");
            PersistentEngine::FrameStats stats = engine.ExecuteFrame(request, clist.Get());
            Check(clist->Close(), "Close(caller list)");
            ID3D12CommandList* lists[] = {clist.Get()};
            engine.queue()->ExecuteCommandLists(1, lists);
            Check(engine.queue()->Signal(cfence.Get(), ++fenceValue), "Signal(caller)");
            Check(cfence->SetEventOnCompletion(fenceValue, cevent),
                  "SetEventOnCompletion(caller)");
            WaitForSingleObject(cevent, INFINITE);
            inputState[index] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            outState[0] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            runs.push_back(stats);
            labels.push_back(label);
        };
        for (int frame = 0; frame < opt.frames; ++frame)
            runBorrowed(static_cast<size_t>(frame), frameName(frame));
        if (opt.recover) runBorrowed(0, "recover-frame-0");
        if (!opt.callerReadback.empty())
            ReadbackTexture(engine.device(), engine.queue(), output.resource.Get(),
                            outSpec->width, outSpec->height, opt.callerReadback);
        CloseHandle(cevent);
    }

    const GpuMemory after = SampleAdapter(engine.adapter());
    const auto debug = engine.debugCounts();
    if (opt.debug) {
        if (debug.errors)
            throw std::runtime_error("D3D12 debug layer reported " +
                                     std::to_string(debug.errors) + " error(s)");
        std::printf("D3D12 debug: zero error/corruption messages (%llu total)\n",
                    static_cast<unsigned long long>(debug.total));
    }
    if (!opt.report.empty()) WriteReport(opt.report, opt, engine, runs, labels, afterInit, after);
    std::printf("engine: %zu stage(s), %zu resource(s), %zu frame(s)\n",
                engine.stages().size(), engine.resources().size(), runs.size());
    for (size_t i = 0; i < runs.size(); ++i)
        std::printf("%s: gpu %.3f ms, record %.3f ms, wait %.3f ms, %llu dispatches\n",
                    labels[i].c_str(), runs[i].gpuIntervalMs, runs[i].recordMs,
                    runs[i].submitWaitMs,
                    static_cast<unsigned long long>(runs[i].dispatches));
    engine.Destroy();
    (void)afterInit;
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
