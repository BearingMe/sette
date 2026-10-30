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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
}

uint32_t Align(uint64_t value) {
    return static_cast<uint32_t>((value + kAlignment - 1) & ~uint64_t(kAlignment - 1));
}

struct Resource {
    std::string name;
    uint64_t bytes = 0;
    std::string init;                 // empty = sentinel fill
    ComPtr<ID3D12Resource> gpu;
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

    Resource& Find(const std::string& name) {
        for (Resource& r : resources)
            if (r.name == name) return r;
        throw std::runtime_error("unknown resource " + name);
    }
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

    void Run(Graph& g, const fs::path& outputDir, uint32_t seed) {
        for (Resource& r : g.resources) Create(r, seed);
        for (Stage& s : g.stages) Prepare(s, g);

        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), nullptr), "Reset(list)");

        for (Stage& s : g.stages) {
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
            list_->Dispatch(s.groups[0], s.groups[1], s.groups[2]);
        }

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
        }
        Check(list_->Close(), "Close(list)");
        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion");
        WaitForSingleObject(event_, INFINITE);

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
        fs::create_directories(outputDir);
        for (size_t i = 0; i < g.dumps.size(); ++i) {
            void* mapped = nullptr;
            Check(readbacks[i]->Map(0, nullptr, &mapped), "Map(readback)");
            WriteFile(outputDir / g.dumps[i].file, mapped, g.dumps[i].bytes);
            readbacks[i]->Unmap(0, nullptr);
            std::printf("wrote %s (%llu bytes)\n", g.dumps[i].file.string().c_str(),
                        static_cast<unsigned long long>(g.dumps[i].bytes));
        }
        std::printf("graph: %zu stage(s), %zu resource(s)\n", g.stages.size(),
                    g.resources.size());
    }

private:
    static uint32_t Count(const Stage& s, bool input) {
        uint32_t n = 0;
        for (const Binding& b : s.bindings)
            if (b.input == input) ++n;
        return n;
    }

    void Transition(Resource& r, D3D12_RESOURCE_STATES to) {
        ID3D12Resource* key = r.gpu.Get();
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
        states_[key] = to;
    }

    void Create(Resource& r, uint32_t seed) {        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = r.bytes;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&r.gpu)),
              "CreateCommittedResource");
        std::vector<uint8_t> init(static_cast<size_t>(r.bytes));
        if (!r.init.empty()) {
            init = ReadFile(fs::path(r.init));
            if (init.size() != r.bytes) throw std::runtime_error("init size for " + r.name);
        } else {
            for (size_t i = 0; i < init.size(); ++i)
                init[i] = static_cast<uint8_t>((i * 37u + 11u + seed * 97u) & 0xFFu);
        }
        D3D12_HEAP_PROPERTIES up{};
        up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC ud = desc;
        ud.Flags = D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> upload;
        Check(device_->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &ud,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&upload)),
              "CreateCommittedResource(upload)");
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "Map(upload)");
        std::memcpy(mapped, init.data(), init.size());
        upload->Unmap(0, nullptr);

        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), nullptr), "Reset(list)");
        Transition(r, D3D12_RESOURCE_STATE_COPY_DEST);
        list_->CopyBufferRegion(r.gpu.Get(), 0, upload.Get(), 0, r.bytes);
        Transition(r, D3D12_RESOURCE_STATE_COMMON);
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
};

int Run(int argc, char** argv) {
    fs::path manifest, outputDir = ".", shaderDir;
    uint32_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--graph" && i + 1 < argc) manifest = argv[++i];
        else if (arg == "--out" && i + 1 < argc) outputDir = argv[++i];
        else if (arg == "--shaders" && i + 1 < argc) shaderDir = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else throw std::runtime_error("unknown option " + arg);
    }
    if (manifest.empty()) {
        std::fprintf(stderr, "usage: m_network_stage --graph graph.txt [--out DIR] "
                             "[--shaders DIR] [--seed N]\n");
        return 2;
    }
    Graph graph = Parse(manifest);
    // shader paths that are just a file name resolve against --shaders
    for (Stage& s : graph.stages)
        if (!fs::exists(s.shader) && !shaderDir.empty()) s.shader = shaderDir / s.shader.filename();
    Runner runner;
    runner.Run(graph, outputDir, seed);
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
