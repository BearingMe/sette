// M05 B4.1: the smallest native path that can execute an *image* stage.
//
// `m_stage_probe` executes one buffer-to-buffer matrix stage.  This program
// executes one compute dispatch whose operands and results are 2D images and
// typed buffers, resolving each binding from a stage fixture instead of from
// hard-coded roles:
//
//   version 2
//   shader  <file>                  the .cso, relative to the fixture directory
//   threads X Y Z                   the shader's [numthreads]
//   groups  X Y Z                   the Dispatch dimensions
//   params  <file> <bytes>          one constant buffer, b0
//   srv buffer  <reg> <file> <bytes>
//   srv tex2d   <reg> <FORMAT> <w> <h> <file>
//   uav buffer  <reg> <file> <bytes>
//   uav tex2d   <reg> <FORMAT> <w> <h> <file>
//
// A `uav` declares an output: its file is written after the dispatch.  A `srv`
// declares an input and must exist.  Formats are named, never inferred, so a
// fixture cannot silently reinterpret bytes; `buffer` means a raw 32-bit view.
// Row pitch is D3D12's copyable footprint, computed by the runtime and printed,
// so a padded layout is visible rather than assumed away.
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef M_TIMING
#include "m_timing.h"
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
    if (!out) throw std::runtime_error("incomplete write: " + path.string());
}

uint32_t Align(uint64_t value) {
    return static_cast<uint32_t>((value + kAlignment - 1) & ~uint64_t(kAlignment - 1));
}

// The formats a fixture may name.  Only what the captured resources use is
// listed: an unlisted name is an error, not a guess.
struct FormatInfo {
    DXGI_FORMAT format;
    uint32_t pixelBytes;
    bool uav;
};

const std::map<std::string, FormatInfo>& Formats() {
    static const std::map<std::string, FormatInfo> table = {
        {"R16_FLOAT", {DXGI_FORMAT_R16_FLOAT, 2, true}},
        {"R16G16_FLOAT", {DXGI_FORMAT_R16G16_FLOAT, 4, true}},
        {"R16G16B16A16_FLOAT", {DXGI_FORMAT_R16G16B16A16_FLOAT, 8, true}},
        {"R32_FLOAT", {DXGI_FORMAT_R32_FLOAT, 4, true}},
        {"R32G32_FLOAT", {DXGI_FORMAT_R32G32_FLOAT, 8, true}},
        {"R32G32B32A32_FLOAT", {DXGI_FORMAT_R32G32B32A32_FLOAT, 16, true}},
        // A SNORM surface has no UAV format; its bytes are written through the
        // UNORM view, which stores the same byte for the same logical value.
        {"R8G8B8A8_UNORM", {DXGI_FORMAT_R8G8B8A8_UNORM, 4, true}},
        // The same 8-bit storage viewed as integers, so a stage can choose the
        // quantisation itself instead of leaving it to the UNORM conversion.
        {"R8G8B8A8_UINT", {DXGI_FORMAT_R8G8B8A8_UINT, 4, true}},
        // ... and the same bytes again as one 32-bit integer per pixel.  This
        // driver drops stores to an R8G8B8A8_UINT unordered-access view (a
        // constant store never lands, while the identical store through an
        // RGBA16F view does), so the byte surfaces are written this way.
        {"R32_UINT", {DXGI_FORMAT_R32_UINT, 4, true}},
    };
    return table;
}

const FormatInfo& Lookup(const std::string& name) {
    auto it = Formats().find(name);
    if (it == Formats().end()) throw std::runtime_error("unknown format " + name);
    return it->second;
}

enum class Kind { Buffer, Texture };

struct Binding {
    bool input = true;
    Kind kind = Kind::Buffer;
    uint32_t reg = 0;
    std::string formatName;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t pixelBytes = 4;
    uint32_t width = 0, height = 0;
    uint64_t bytes = 0;
    fs::path file;
    std::vector<uint8_t> data;
};

struct Fixture {
    fs::path directory;
    fs::path shader;
    uint32_t threads[3] = {1, 1, 1};
    uint32_t groups[3] = {1, 1, 1};
    std::vector<uint8_t> params;
    std::vector<Binding> bindings;
    // Fixed dec0 prefix, not another graph language. Three offline weight files.
    std::vector<fs::path> dec0Weights;
    std::vector<fs::path> enc0Weights;
    fs::path shaderDirectory;
};

Fixture Parse(const fs::path& manifest, const fs::path& shaderDir) {
    Fixture f;
    f.directory = manifest.parent_path();
    f.shaderDirectory = shaderDir;
    std::ifstream in(manifest);
    if (!in) throw std::runtime_error("cannot open " + manifest.string());
    std::string line;
    int version = 0;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::string key;
        if (!(words >> key) || key.empty() || key[0] == '#') continue;
        if (key == "version") {
            words >> version;
        } else if (key == "shader") {
            std::string name;
            words >> name;
            fs::path relative(name);
            f.shader = relative.is_absolute() ? relative : f.directory / relative;
            if (!fs::exists(f.shader) && !shaderDir.empty())
                f.shader = shaderDir / relative;
        } else if (key == "dec0_int8" || key == "enc0_int8") {
            for (int i = 0; i < 3; ++i) {
                std::string name;
                if (!(words >> name)) throw std::runtime_error("dec0_int8 requires codes scales bias");
                (key == "dec0_int8" ? f.dec0Weights : f.enc0Weights).push_back(f.directory / name);
            }
        } else if (key == "threads") {
            words >> f.threads[0] >> f.threads[1] >> f.threads[2];
        } else if (key == "groups") {
            words >> f.groups[0] >> f.groups[1] >> f.groups[2];
        } else if (key == "params") {
            std::string name;
            uint64_t bytes = 0;
            words >> name >> bytes;
            f.params = ReadFile(f.directory / name);
            if (f.params.size() != bytes)
                throw std::runtime_error("params size mismatch in " + manifest.string());
        } else if (key == "srv" || key == "uav") {
            Binding b;
            b.input = key == "srv";
            std::string kind, name;
            words >> kind >> b.reg;
            b.kind = kind == "tex2d" ? Kind::Texture : Kind::Buffer;
            if (kind != "tex2d" && kind != "buffer")
                throw std::runtime_error("unknown binding kind " + kind);
            if (b.kind == Kind::Texture) {
                words >> b.formatName >> b.width >> b.height >> name;
                const FormatInfo& info = Lookup(b.formatName);
                b.format = info.format;
                b.pixelBytes = info.pixelBytes;
                if (!b.input && !info.uav)
                    throw std::runtime_error(b.formatName + " cannot be a UAV");
                b.bytes = uint64_t(b.width) * b.height * b.pixelBytes;
            } else {
                words >> name >> b.bytes;
                b.format = DXGI_FORMAT_R32_UINT;
                if (b.bytes % 4) throw std::runtime_error("buffer binding is not 32-bit aligned");
            }
            b.file = f.directory / name;
            if (b.input) {
                b.data = ReadFile(b.file);
                if (b.data.size() != b.bytes)
                    throw std::runtime_error("input size mismatch: " + b.file.string());
            }
            f.bindings.push_back(std::move(b));
        } else {
            throw std::runtime_error("unknown manifest key " + key);
        }
    }
    if (version != 2) throw std::runtime_error("unsupported fixture version");
    if (f.shader.empty()) throw std::runtime_error("manifest has no shader");
    if (f.params.empty()) throw std::runtime_error("manifest has no params");
    return f;
}

struct GpuBuffer {
    ComPtr<ID3D12Resource> resource;
    uint64_t size = 0;
    void* mapped = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

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
                std::wprintf(L"device: %s (vendor 0x%04x)\n", desc.Description, desc.VendorId);
                break;
            }
        }
        if (!device_) throw std::runtime_error("no D3D12 device at feature level 12_0");
        if (wantDebug) Check(device_.As(&infoQueue_), "ID3D12InfoQueue");

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

    void Issue(const std::string& label, UINT x, UINT y, UINT z) {
#ifdef M_TIMING
        if (timer_) timer_->Begin(list_.Get());
#endif
        list_->Dispatch(x, y, z);
#ifdef M_TIMING
        if (timer_) timer_->End(list_.Get(), label);
#endif
    }

    void Run(Fixture& f, const fs::path& outputDir, int seed = -1,
             const std::string& timingPath = "") {
        const bool dec0Int8 = !f.dec0Weights.empty();
        const bool enc0Int8 = !f.enc0Weights.empty();
#ifdef M_TIMING
        std::unique_ptr<timing::GpuTimer> timer;
        if (!timingPath.empty())
            timer = std::make_unique<timing::GpuTimer>(device_.Get(), queue_.Get(), 32);
        timer_ = timer.get();
#endif
        if (enc0Int8) {
            if (dec0Int8 || f.enc0Weights.size()!=3 || f.bindings.size()!=3 ||
                f.params.size()!=24 || f.shader.filename()!="enc0_q8_epilogue.cso" ||
                !f.bindings[0].input || f.bindings[0].kind!=Kind::Buffer || f.bindings[0].bytes!=12416 ||
                !f.bindings[1].input || f.bindings[1].kind!=Kind::Texture ||
                f.bindings[1].format!=DXGI_FORMAT_R16G16B16A16_FLOAT ||
                f.bindings[1].width!=960 || f.bindings[1].height!=540 ||
                f.bindings[2].input || f.bindings[2].kind!=Kind::Buffer || f.bindings[2].bytes!=8355840)
                throw std::runtime_error("enc0_int8: incompatible fixture");
            const uint32_t* p=reinterpret_cast<const uint32_t*>(f.params.data());
            if (p[0]!=960 || p[1]!=540 || p[2]!=960 || p[3]!=544 || p[4]>=8 || p[5]!=1)
                throw std::runtime_error("enc0_int8: only bounded 960x544 reset contract supported");
            Binding b; b.bytes=130560ull*64*2; b.file="out-embedding.bin";
            f.bindings.push_back(b);
        }
        if (dec0Int8) {
            if (f.dec0Weights.size() != 3 || f.bindings.size() != 6 ||
                !f.bindings[0].input || !f.bindings[1].input ||
                f.bindings[0].kind != Kind::Buffer || f.bindings[1].kind != Kind::Buffer ||
                f.bindings[0].bytes != 8448 || f.bindings[1].bytes != 8355840 ||
                f.params.size() != 32 || f.shader.filename() != "dec0_q8_epilogue.cso")
                throw std::runtime_error("dec0_int8: incompatible fixture");
            for (size_t i = 2; i < 6; ++i) {
                const auto& b = f.bindings[i];
                const auto format = i == 5 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32_UINT;
                if (b.input || b.kind != Kind::Texture || b.width != 960 || b.height != 540 || b.format != format)
                    throw std::runtime_error("dec0_int8: incompatible output surface");
            }
            const int32_t* p = reinterpret_cast<const int32_t*>(f.params.data());
            if (p[0] != 960 || p[1] != 544 || p[6] != 1)
                throw std::runtime_error("dec0_int8 supports only 960x544 reset frames");
            Binding b;
            b.bytes = 130560ull * 80 * 2;
            b.file = "out-expansion.bin";
            f.bindings.push_back(b); // internal SRV t2, no host input payload
        }
        const uint32_t srvCount = Count(f, true), uavCount = Count(f, false);
        ComPtr<ID3D12RootSignature> signature = MakeRootSignature(srvCount, uavCount);
        ComPtr<ID3D12PipelineState> pipeline = MakePipeline(signature.Get(), f.shader);

        GpuBuffer params = Upload(f.params.size(), f.params.data());
        std::vector<GpuBuffer> resources(f.bindings.size());
        std::vector<GpuBuffer> readback(f.bindings.size());
        for (size_t i = 0; i < f.bindings.size(); ++i) {
            Binding& b = f.bindings[i];
            if (!b.input) {
                resources[i] = b.kind == Kind::Texture
                                   ? CommittedTexture(b, D3D12_RESOURCE_STATE_COPY_DEST)
                                    : CommittedBuffer(Align(b.bytes), D3D12_HEAP_TYPE_DEFAULT,
                                                      D3D12_RESOURCE_STATE_COMMON, true);
                readback[i] = b.kind == Kind::Texture
                                  ? TextureStaging(b, D3D12_HEAP_TYPE_READBACK,
                                                   D3D12_RESOURCE_STATE_COPY_DEST)
                                  : CommittedBuffer(Align(b.bytes), D3D12_HEAP_TYPE_READBACK,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, false);
            } else if ((dec0Int8 || enc0Int8) && i == f.bindings.size() - 1) {
                resources[i] = CommittedBuffer(b.bytes, D3D12_HEAP_TYPE_DEFAULT,
                                                D3D12_RESOURCE_STATE_COMMON, true);
            } else if (b.kind == Kind::Texture) {
                resources[i] = UploadTexture(b);
            } else {
                resources[i] = Upload(b.bytes, b.data.data());
            }
        }

        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), pipeline.Get()), "Reset(list)");
        for (size_t i=0;i<f.bindings.size();++i)
            if (!f.bindings[i].input && f.bindings[i].kind==Kind::Buffer)
                Transition(resources[i].resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);

        // Seed every surface via its actual copy footprint, so repeat runs can
        // distinguish complete writes from untouched, uninitialised GPU memory.
        std::vector<GpuBuffer> sentinels;
        if (seed >= 0) {
            for (size_t i = 0; i < f.bindings.size(); ++i) {
                const Binding& b = f.bindings[i];
                if (b.input) continue;
                GpuBuffer staging = b.kind == Kind::Texture
                    ? TextureStaging(b, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)
                    : Upload(b.bytes, nullptr);
                void* mapped = staging.mapped;
                if (!mapped) Check(staging.resource->Map(0, nullptr, &mapped), "Map(sentinel)");
                auto bytes = static_cast<uint8_t*>(mapped);
                for (uint64_t j = 0; j < staging.size; ++j)
                    bytes[j] = uint8_t((j * 73 + (j >> 8) * 19 + seed * 151 + 37) & 255);
                if (b.kind == Kind::Buffer)
                    list_->CopyBufferRegion(resources[i].resource.Get(), 0, staging.resource.Get(), 0, b.bytes);
                else {
                    auto desc = TextureDesc(b);
                    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
                    from.pResource = staging.resource.Get();
                    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    device_->GetCopyableFootprints(&desc, 0, 1, 0, &from.PlacedFootprint, nullptr, nullptr, nullptr);
                    to.pResource = resources[i].resource.Get();
                    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    list_->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                }
                sentinels.push_back(std::move(staging));
            }
        }
        Prefix prefix;
        if (dec0Int8) prefix = Dec0Prefix(f, resources, params);
        if (enc0Int8) prefix = Enc0Prefix(f, resources, params, seed);

        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = srvCount + uavCount;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> heap;
        Check(device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)),
              "CreateDescriptorHeap");
        const UINT stride = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();

        uint32_t srvIndex = 0, uavIndex = 0;
        for (size_t i = 0; i < f.bindings.size(); ++i) {
            const Binding& b = f.bindings[i];
            const uint32_t slot = b.input ? srvIndex++ : uavIndex++;
            const uint32_t at = b.input ? slot : srvCount + slot;
            D3D12_CPU_DESCRIPTOR_HANDLE handle{cpu.ptr + SIZE_T(at) * stride};
            if (b.input) {
                if (b.kind == Kind::Texture) {
                    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                    srv.Format = b.format;
                    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    srv.Texture2D.MipLevels = 1;
                    device_->CreateShaderResourceView(resources[i].resource.Get(), &srv, handle);
                } else {
                    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                    srv.Format = DXGI_FORMAT_R32_UINT;
                    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    srv.Buffer.NumElements = static_cast<UINT>(b.bytes / 4);
                    device_->CreateShaderResourceView(resources[i].resource.Get(), &srv, handle);
                }
            } else {
                if (b.kind == Kind::Texture) {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
                    uav.Format = b.format;
                    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    device_->CreateUnorderedAccessView(resources[i].resource.Get(), nullptr,
                                                       &uav, handle);
                } else {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
                    uav.Format = DXGI_FORMAT_R32_UINT;
                    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                    uav.Buffer.NumElements = static_cast<UINT>(b.bytes / 4);
                    device_->CreateUnorderedAccessView(resources[i].resource.Get(), nullptr,
                                                       &uav, handle);
                }
            }
        }

        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        list_->SetDescriptorHeaps(1, heaps);
        list_->SetPipelineState(pipeline.Get());
        list_->SetComputeRootSignature(signature.Get());
        list_->SetComputeRootConstantBufferView(0, params.address);
        list_->SetComputeRootDescriptorTable(1, gpu);
        list_->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{
                                                      gpu.ptr + SIZE_T(srvCount) * stride});
        for (size_t i = 0; i < f.bindings.size(); ++i) {
            if (f.bindings[i].input) continue;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = resources[i].resource.Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list_->ResourceBarrier(1, &barrier);
        }
        Issue("main", f.groups[0], f.groups[1], f.groups[2]);

        // ---- read back every output -------------------------------------
        for (size_t i = 0; i < f.bindings.size(); ++i) {
            const Binding& b = f.bindings[i];
            if (b.input) continue;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = resources[i].resource.Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list_->ResourceBarrier(1, &barrier);
            CopyOut(resources[i], readback[i], b);
        }
        if (dec0Int8) {
            for (size_t i = 0; i < prefix.outputs.size(); ++i) {
                auto* source = prefix.outputs[i];
                Transition(source->resource.Get(), i == 0 || i == 1 || i == 4
                    ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                prefix.readback.push_back(CommittedBuffer(prefix.bytes[i], D3D12_HEAP_TYPE_READBACK,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, false));
                list_->CopyBufferRegion(prefix.readback.back().resource.Get(), 0, source->resource.Get(), 0, prefix.bytes[i]);
            }
        }
        if (enc0Int8) {
            for (size_t i=0;i<prefix.outputs.size();++i) {
                auto* source=prefix.outputs[i];
                Transition(source->resource.Get(), i==0 || i==1 || i==2 || i==5
                    ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                prefix.readback.push_back(CommittedBuffer(prefix.bytes[i], D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST,false));
                list_->CopyBufferRegion(prefix.readback.back().resource.Get(),0,source->resource.Get(),0,prefix.bytes[i]);
            }
        }
#ifdef M_TIMING
        if (timer) timer->Resolve(list_.Get());
#endif
        Check(list_->Close(), "Close(list)");

        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion");
        WaitForSingleObject(event_, INFINITE);

#ifdef M_TIMING
        if (timer) timer->Write(timingPath);
        timer_ = nullptr;
#endif

        if (infoQueue_) {
            const UINT64 count = infoQueue_->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; ++i) {
                SIZE_T size = 0;
                Check(infoQueue_->GetMessage(i, nullptr, &size), "InfoQueue message size");
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
        fs::create_directories(outputDir);
        if (enc0Int8) {
            const char* names[]={"out-features.bin","out-q8a.bin","out-q8as.bin","out-acc.bin","out-float.bin","out-embedding.bin"};
            for (size_t i=0;i<prefix.readback.size();++i) {
                void* data=nullptr;
                Check(prefix.readback[i].resource->Map(0,nullptr,&data),"Map(enc0 diagnostics)");
                WriteFile(outputDir/names[i],data,prefix.bytes[i]);
                prefix.readback[i].resource->Unmap(0,nullptr);
            }
        }
        if (dec0Int8) {
            const char* names[] = {"out-q8a.bin", "out-q8as.bin", "out-acc.bin", "out-float.bin", "out-expansion.bin"};
            for (size_t i = 0; i < prefix.readback.size(); ++i) {
                void* data = nullptr;
                Check(prefix.readback[i].resource->Map(0, nullptr, &data), "Map(dec0 diagnostics)");
                WriteFile(outputDir / names[i], data, prefix.bytes[i]);
                prefix.readback[i].resource->Unmap(0, nullptr);
            }
        }
        for (size_t i = 0; i < f.bindings.size(); ++i) {
            const Binding& b = f.bindings[i];
            if (b.input) continue;
            std::vector<uint8_t> bytes(b.bytes);
            Gather(readback[i], b, bytes.data());
            WriteFile(outputDir / b.file.filename(), bytes.data(), bytes.size());
            std::printf("wrote %s (%llu bytes)\n", b.file.filename().string().c_str(),
                        static_cast<unsigned long long>(bytes.size()));
        }
        std::printf("dispatch %ux%ux%u threads %ux%ux%u, %zu binding(s)\n", f.groups[0],
                    f.groups[1], f.groups[2], f.threads[0], f.threads[1], f.threads[2],
                    f.bindings.size());
    }

private:
    struct Prefix {
        std::vector<GpuBuffer> buffers, readback;
        std::vector<GpuBuffer> sentinels;
        std::vector<GpuBuffer*> outputs;
        std::vector<uint64_t> bytes;
        std::vector<ComPtr<ID3D12DescriptorHeap>> heaps;
        std::vector<ComPtr<ID3D12RootSignature>> signatures;
        std::vector<ComPtr<ID3D12PipelineState>> pipelines;
    };

    void Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = resource;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = before; b.Transition.StateAfter = after;
        list_->ResourceBarrier(1, &b);
    }

    Prefix Enc0Prefix(const Fixture& f, std::vector<GpuBuffer>& external, const GpuBuffer& params, int seed) {
        Prefix p;
        p.buffers.reserve(11);
        const uint64_t sizes[]={33423360,16711680,522240,33423360,33423360,16};
        for (auto bytes:sizes) {
            p.buffers.push_back(CommittedBuffer(bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COMMON,true));
            auto& b=p.buffers.back();
            if (seed>=0) {
                p.sentinels.push_back(Upload(bytes,nullptr));
                auto& s=p.sentinels.back();
                auto* data=static_cast<uint8_t*>(s.mapped);
                for (uint64_t j=0;j<bytes;++j) data[j]=uint8_t((j*73+(j>>8)*19+seed*151+37)&255);
                Transition(b.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
                list_->CopyBufferRegion(b.resource.Get(),0,s.resource.Get(),0,bytes);
                Transition(b.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            } else Transition(b.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        Transition(external.back().resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const uint64_t weightBytes[]={8192,256,256};
        for (size_t i=0;i<3;++i) {
            auto data=ReadFile(f.enc0Weights[i]);
            if (data.size()!=weightBytes[i]) throw std::runtime_error("enc0 INT8 weight size mismatch");
            p.buffers.push_back(Upload(data.size(),data.data()));
        }
        const uint32_t shape[]={130560,128,64,0};
        p.buffers.push_back(Upload(sizeof(shape),shape));
        auto dispatch=[&](const char* shader,const GpuBuffer& cb,std::vector<GpuBuffer*> srvs,
                          std::vector<GpuBuffer*> uavs,UINT groups,bool texture) {
            p.signatures.push_back(MakeRootSignature(UINT(srvs.size()),UINT(uavs.size())));
            p.pipelines.push_back(MakePipeline(p.signatures.back().Get(),f.shaderDirectory/shader));
            D3D12_DESCRIPTOR_HEAP_DESC desc{};
            desc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors=UINT(srvs.size()+uavs.size());desc.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            Check(device_->CreateDescriptorHeap(&desc,IID_PPV_ARGS(&heap)),"CreateDescriptorHeap(enc0)");
            auto cpu=heap->GetCPUDescriptorHandleForHeapStart();auto gpu=heap->GetGPUDescriptorHandleForHeapStart();
            UINT stride=device_->GetDescriptorHandleIncrementSize(desc.Type);
            for (size_t i=0;i<srvs.size()+uavs.size();++i) {
                D3D12_CPU_DESCRIPTOR_HANDLE at{cpu.ptr+i*stride};
                if (i<srvs.size()) {
                    D3D12_SHADER_RESOURCE_VIEW_DESC v{};v.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    if (texture && i==1) {
                        v.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;v.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;v.Texture2D.MipLevels=1;
                    } else {
                        v.Format=DXGI_FORMAT_R32_UINT;v.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;v.Buffer.NumElements=UINT(srvs[i]->size/4);
                    }
                    device_->CreateShaderResourceView(srvs[i]->resource.Get(),&v,at);
                } else {
                    auto* b=uavs[i-srvs.size()];D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
                    v.Format=DXGI_FORMAT_R32_UINT;v.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;v.Buffer.NumElements=UINT(b->size/4);
                    device_->CreateUnorderedAccessView(b->resource.Get(),nullptr,&v,at);
                }
            }
            ID3D12DescriptorHeap* heaps[]={heap.Get()};list_->SetDescriptorHeaps(1,heaps);
            list_->SetPipelineState(p.pipelines.back().Get());list_->SetComputeRootSignature(p.signatures.back().Get());
            list_->SetComputeRootConstantBufferView(0,cb.address);list_->SetComputeRootDescriptorTable(1,gpu);
            list_->SetComputeRootDescriptorTable(2,D3D12_GPU_DESCRIPTOR_HANDLE{gpu.ptr+srvs.size()*stride});
            Issue(shader,groups,1,1);p.heaps.push_back(heap);
            std::printf("enc0 prefix %s: dispatch %ux1x1\n",shader,groups);
        };
        auto& b=p.buffers;
        dispatch("enc0_q8_features.cso",params,{&external[0],&external[1]},{&b[0]},2040,true);
        Transition(b[0].resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dispatch("swin_q8_quant.cso",b[9],{&b[0]},{&b[1],&b[2]},2040,false);
        for (size_t i:{1u,2u}) Transition(b[i].resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dispatch("swin_mlp_up_q8.cso",b[9],{&b[1],&b[6],&b[2],&b[7],&b[8]},
                 {&external.back(),&b[3],&b[4]},8160,false);
        Transition(external.back().resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        p.outputs={&b[0],&b[1],&b[2],&b[3],&b[4],&external.back()};
        p.bytes={33423360,16711680,522240,33423360,33423360,16711680};
        return p;
    }

    Prefix Dec0Prefix(const Fixture& f, std::vector<GpuBuffer>& external, const GpuBuffer& params) {
        Prefix p;
        p.buffers.reserve(8); // stable pointers through submission and readback
        for (uint64_t bytes : {8355840ull, 522240ull, 41779200ull, 41779200ull, 16ull})
            p.buffers.push_back(CommittedBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT,
                                                D3D12_RESOURCE_STATE_COMMON, true));
        for (auto& buffer : p.buffers)
            Transition(buffer.resource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Transition(external.back().resource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const uint64_t weightBytes[] = {5120, 320, 320};
        for (size_t i = 0; i < 3; ++i) {
            auto data = ReadFile(f.dec0Weights[i]);
            if (data.size() != weightBytes[i]) throw std::runtime_error("dec0 INT8 weight size mismatch");
            p.buffers.push_back(Upload(data.size(), data.data()));
        }
        auto dispatch = [&](const char* shader, std::vector<GpuBuffer*> srvs,
                            std::vector<GpuBuffer*> uavs, UINT y) {
            p.signatures.push_back(MakeRootSignature(UINT(srvs.size()), UINT(uavs.size())));
            p.pipelines.push_back(MakePipeline(p.signatures.back().Get(), f.shaderDirectory / shader));
            D3D12_DESCRIPTOR_HEAP_DESC desc{};
            desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors = UINT(srvs.size() + uavs.size());
            desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            Check(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap(dec0)");
            auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
            auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
            const UINT stride = device_->GetDescriptorHandleIncrementSize(desc.Type);
            for (size_t i = 0; i < srvs.size() + uavs.size(); ++i) {
                D3D12_CPU_DESCRIPTOR_HANDLE at{cpu.ptr + i * stride};
                if (i < srvs.size()) {
                    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
                    v.Format = DXGI_FORMAT_R32_UINT; v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    v.Buffer.NumElements = UINT(srvs[i]->size / 4);
                    device_->CreateShaderResourceView(srvs[i]->resource.Get(), &v, at);
                } else {
                    auto* b = uavs[i - srvs.size()];
                    D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
                    v.Format = DXGI_FORMAT_R32_UINT; v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                    v.Buffer.NumElements = UINT(b->size / 4);
                    device_->CreateUnorderedAccessView(b->resource.Get(), nullptr, &v, at);
                }
            }
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            list_->SetDescriptorHeaps(1, heaps);
            list_->SetPipelineState(p.pipelines.back().Get());
            list_->SetComputeRootSignature(p.signatures.back().Get());
            list_->SetComputeRootConstantBufferView(0, params.address);
            list_->SetComputeRootDescriptorTable(1, gpu);
            list_->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{gpu.ptr + srvs.size() * stride});
            Issue(shader, 2040, y, 1);
            std::printf("dec0 prefix %s: dispatch 2040x%ux1\n", shader, y);
            p.heaps.push_back(heap);
        };
        auto& b = p.buffers;
        dispatch("dec0_q8_quant.cso", {&external[0], &external[1]},
                 {&b[4], &b[4], &b[4], &b[4], &b[0], &b[1]}, 1);
        Transition(b[0].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Transition(b[1].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dispatch("dec0_q8_matmul.cso", {&b[0], &b[5], &b[1], &b[6], &b[7]},
                 {&external.back(), &b[2], &b[3]}, 40);
        Transition(external.back().resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        p.outputs = {&b[0], &b[1], &b[2], &b[3], &external.back()};
        p.bytes = {8355840, 522240, 41779200, 41779200, 20889600};
        return p;
    }

    static uint32_t Count(const Fixture& f, bool input) {
        uint32_t n = 0;
        for (const Binding& b : f.bindings)
            if (b.input == input) ++n;
        return n;
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

    ComPtr<ID3D12PipelineState> MakePipeline(ID3D12RootSignature* signature, const fs::path& path) {
        auto code = ReadFile(path);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = signature;
        pso.CS.pShaderBytecode = code.data();
        pso.CS.BytecodeLength = code.size();
        ComPtr<ID3D12PipelineState> pipeline;
        Check(device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)),
              "CreateComputePipelineState");
        return pipeline;
    }

    D3D12_RESOURCE_DESC BufferDesc(uint64_t size) const {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        return desc;
    }

    D3D12_RESOURCE_DESC TextureDesc(const Binding& b) const {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = b.width;
        desc.Height = b.height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = b.format;
        desc.SampleDesc.Count = 1;
        return desc;
    }

    GpuBuffer Upload(uint64_t size, const void* data) {
        GpuBuffer buffer;
        buffer.size = Align(size);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC desc = BufferDesc(buffer.size);
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&buffer.resource)),
              "CreateCommittedResource(upload)");
        Check(buffer.resource->Map(0, nullptr, &buffer.mapped), "Map(upload)");
        std::memset(buffer.mapped, 0, buffer.size);
        if (data) std::memcpy(buffer.mapped, data, static_cast<size_t>(size));
        buffer.address = buffer.resource->GetGPUVirtualAddress();
        return buffer;
    }

    GpuBuffer CommittedBuffer(uint64_t size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                              bool uav) {
        GpuBuffer buffer;
        buffer.size = Align(size);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = type;
        D3D12_RESOURCE_DESC desc = BufferDesc(buffer.size);
        if (uav) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&buffer.resource)),
              "CreateCommittedResource(buffer)");
        buffer.address = buffer.resource->GetGPUVirtualAddress();
        return buffer;
    }

    GpuBuffer CommittedTexture(const Binding& b, D3D12_RESOURCE_STATES state) {
        GpuBuffer texture;
        texture.size = b.bytes;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc = TextureDesc(b);
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&texture.resource)),
              "CreateCommittedResource(texture)");
        return texture;
    }

    // A linear buffer big enough for one texture copy, with the footprint the
    // runtime chose.  Its row pitch is what the file conversion below uses.
    GpuBuffer TextureStaging(const Binding& b, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
        D3D12_RESOURCE_DESC desc = TextureDesc(b);
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
        return CommittedBuffer(total, type, state, false);
    }

    // One row-major upload buffer, filled through D3D12's own copyable footprint.
    GpuBuffer UploadTexture(const Binding& b) {
        D3D12_RESOURCE_DESC desc = TextureDesc(b);
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
        GpuBuffer staging = CommittedBuffer(total, D3D12_HEAP_TYPE_UPLOAD,
                                            D3D12_RESOURCE_STATE_GENERIC_READ, false);
        GpuBuffer texture = CommittedTexture(b, D3D12_RESOURCE_STATE_COPY_DEST);
        void* mapped = nullptr;
        Check(staging.resource->Map(0, nullptr, &mapped), "Map(texture staging)");
        const uint32_t srcRow = b.width * b.pixelBytes;
        for (uint32_t y = 0; y < b.height; ++y) {
            std::memcpy(static_cast<uint8_t*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch,
                        b.data.data() + size_t(y) * srcRow, srcRow);
        }
        staging.resource->Unmap(0, nullptr);
        std::printf("texture %ux%u %s: %u rows, row pitch %u, %llu bytes\n", b.width, b.height,
                    b.formatName.c_str(), rows, footprint.Footprint.RowPitch,
                    static_cast<unsigned long long>(total));

        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), nullptr), "Reset(list)");
        D3D12_TEXTURE_COPY_LOCATION from{};
        from.pResource = staging.resource.Get();
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION to{};
        to.pResource = texture.resource.Get();
        to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.SubresourceIndex = 0;
        list_->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = texture.resource.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list_->ResourceBarrier(1, &barrier);
        Check(list_->Close(), "Close(upload list)");
        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion");
        WaitForSingleObject(event_, INFINITE);
        return texture;
    }

    void CopyOut(GpuBuffer& source, GpuBuffer& readback, const Binding& b) {
        if (b.kind == Kind::Buffer) {
            list_->CopyBufferRegion(readback.resource.Get(), 0, source.resource.Get(), 0, b.bytes);
            return;
        }
        D3D12_RESOURCE_DESC desc = TextureDesc(b);
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
        D3D12_TEXTURE_COPY_LOCATION from{};
        from.pResource = source.resource.Get();
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION to{};
        to.pResource = readback.resource.Get();
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = footprint;
        list_->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }

    void Gather(GpuBuffer& readback, const Binding& b, uint8_t* out) {
        void* mapped = nullptr;
        Check(readback.resource->Map(0, nullptr, &mapped), "Map(readback)");
        if (b.kind == Kind::Buffer) {
            std::memcpy(out, mapped, static_cast<size_t>(b.bytes));
        } else {
            D3D12_RESOURCE_DESC desc = TextureDesc(b);
            UINT rows = 0;
            UINT64 rowBytes = 0, total = 0;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
            device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
            const uint32_t row = b.width * b.pixelBytes;
            for (uint32_t y = 0; y < b.height; ++y) {
                std::memcpy(out + size_t(y) * row,
                            static_cast<const uint8_t*>(mapped) + footprint.Offset +
                                size_t(y) * footprint.Footprint.RowPitch,
                            row);
            }
        }
        readback.resource->Unmap(0, nullptr);
    }

    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12InfoQueue> infoQueue_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    uint64_t fenceValue_ = 0;
#ifdef M_TIMING
    timing::GpuTimer* timer_ = nullptr;
#endif
};

int Run(int argc, char** argv) {
    fs::path manifest, outputDir = ".", shaderDir;
    std::string timingPath;
    int seed = -1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fixture" && i + 1 < argc) manifest = argv[++i];
        else if (arg == "--out" && i + 1 < argc) outputDir = argv[++i];
        else if (arg == "--shaders" && i + 1 < argc) shaderDir = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
#ifdef M_TIMING
        else if (arg == "--timing" && i + 1 < argc) timingPath = argv[++i];
#endif
        else throw std::runtime_error("unknown option " + arg);
    }
    if (manifest.empty()) {
        std::fprintf(stderr,
                     "usage: m_image_stage --fixture stage.txt [--out DIR] [--shaders DIR]\n");
        return 2;
    }
    Fixture fixture = Parse(manifest, shaderDir);
    Runner runner;
    runner.Run(fixture, outputDir, seed, timingPath);
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
