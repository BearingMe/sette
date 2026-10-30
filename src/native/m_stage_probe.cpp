// C18: the smallest native execution path this project needs.
//
// One compute dispatch of `gemm_f16_cs.hlsl` or `gemm_int8_cs.hlsl` on a D3D12
// queue, with explicit resource roles:
//   * the operands and the parameter block arrive in upload-heap buffers and are
//     bound as root SRVs / a root CBV;
//   * the result is a default-heap UAV buffer copied into a readback heap;
//   * one fence, one wait, one map - no frames, no swapchain, no inference
//     framework.
// Buffers are contiguous sub-allocations of three large committed resources so
// the file stays readable without pretending to be an allocator.
//
// usage: m_stage_probe --case case.bin --out result.bin [--shader x.cso]
//
// case.bin version 1 (FP16, gemm_f16_cs.hlsl):
//   "MSTAGE01" | u32 version=1 | u32 M | u32 K | u32 N | u32 flags
//   | M*K f16 X | K*N f16 W | N f16 bias
// case.bin version 2 (INT8, gemm_int8_cs.hlsl):
//   "MSTAGE01" | u32 version=2 | u32 M | u32 K | u32 N | u32 flags
//   | M*K i8 A | N*K i8 W (K-major per column) | M f32 activation scales
//   | N f32 weight scales | N f32 bias
// result.bin: M*N f32, for both versions
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kAlignment = 256;
constexpr uint32_t kMaxBuffers = 4;

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t flags;
};

void Check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char text[256];
        std::snprintf(text, sizeof(text), "%s failed (hr=0x%08lx)", what, static_cast<unsigned long>(hr));
        throw std::runtime_error(text);
    }
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteFile(const std::filesystem::path& path, const void* data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

uint32_t Align(uint32_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

// Root signature: b0 constants, t0..t2 operands as one SRV table, u0 result as
// one UAV table.  Root descriptors cannot describe every buffer view (a typed
// buffer needs its format), so the operands go through a small descriptor heap.
ComPtr<ID3D12RootSignature> MakeRootSignature(ID3D12Device* device) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 5;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable.NumDescriptorRanges = 1;
    params[2].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 3;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (FAILED(hr) && error) {
        std::fprintf(stderr, "%s\n", static_cast<const char*>(error->GetBufferPointer()));
    }
    Check(hr, "D3D12SerializeRootSignature");
    ComPtr<ID3D12RootSignature> result;
    Check(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                      IID_PPV_ARGS(&result)),
          "CreateRootSignature");
    return result;
}

class Runner {
public:
    Runner() {
        UINT flags = 0;
#if defined(_DEBUG)
        const bool wantDebug = true;
#else
        const bool wantDebug = GetEnvironmentVariableW(L"M_STAGE_DEBUG", nullptr, 0) > 0;
#endif
        ComPtr<ID3D12Debug> debug;
        if (wantDebug && SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            flags |= DXGI_CREATE_FACTORY_DEBUG;
        }
        ComPtr<IDXGIFactory6> factory;
        Check(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(
                 i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_)))) {
                std::wprintf(L"device: %s (vendor 0x%04x, device 0x%04x)\n", desc.Description,
                             desc.VendorId, desc.DeviceId);
                break;
            }
        }
        if (!device_) throw std::runtime_error("no D3D12 device at feature level 12_0");

        D3D12_COMMAND_QUEUE_DESC queue{};
        queue.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        Check(device_->CreateCommandQueue(&queue, IID_PPV_ARGS(&queue_)), "CreateCommandQueue");
        Check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator_)),
              "CreateCommandAllocator");
        Check(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator_.Get(), nullptr,
                                         IID_PPV_ARGS(&list_)),
              "CreateCommandList");
        Check(list_->Close(), "Close");
        Check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence");
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_) throw std::runtime_error("CreateEventW failed");
    }

    ~Runner() {
        if (event_) CloseHandle(event_);
    }

    void Run(const Header& header, const std::vector<std::vector<uint8_t>>& sources,
             const std::filesystem::path& shader, const std::filesystem::path& output) {
        // ---- shader ------------------------------------------------------
        auto code = ReadFile(shader);
        auto signature = MakeRootSignature(device_.Get());

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = signature.Get();
        pso.CS.pShaderBytecode = code.data();
        pso.CS.BytecodeLength = code.size();
        ComPtr<ID3D12PipelineState> pipeline;
        Check(device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)), "CreateComputePipelineState");

        // ---- parameter block ---------------------------------------------
        std::vector<uint8_t> constants(256, 0);
        const uint32_t values[4] = {header.M, header.K, header.N, header.flags & 1u};
        std::memcpy(constants.data(), values, sizeof(values));
        GpuBuffer params = Upload(constants.size(), constants.data());

        RunDispatch(header, signature.Get(), pipeline.Get(), params, sources, output);
    }

private:
    struct GpuBuffer {
        ComPtr<ID3D12Resource> resource;
        uint64_t size = 0;
        void* mapped = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    };

    GpuBuffer Upload(uint64_t size, const void* data) {
        GpuBuffer buffer;
        buffer.size = Align(static_cast<uint32_t>(size));
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = buffer.size;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
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

    void RunDispatch(const Header& header, ID3D12RootSignature* signature, ID3D12PipelineState* pipeline,
                     GpuBuffer& params, const std::vector<std::vector<uint8_t>>& sources,
                     const std::filesystem::path& output) {
        const uint32_t rows = header.M, cols = header.N;
        if (sources.empty() || sources.size() > 5) throw std::runtime_error("1..5 source buffers are supported");

        std::vector<GpuBuffer> uploads;
        for (const auto& source : sources)
            uploads.push_back(Upload(source.size(), source.data()));

        const size_t outBytes = static_cast<size_t>(rows) * cols * 4;
        GpuBuffer out = Committed(outBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, true);
        GpuBuffer readback = Committed(outBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);

        D3D12_RESOURCE_BARRIER barriers[2]{};
        barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[0].Transition.pResource = out.resource.Get();
        barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        Check(allocator_->Reset(), "Reset(allocator)");
        Check(list_->Reset(allocator_.Get(), pipeline), "Reset(list)");

        // ---- descriptors --------------------------------------------------
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = 6;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> heap;
        Check(device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");
        const UINT stride = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Format = DXGI_FORMAT_R32_UINT;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.FirstElement = 0;
        for (size_t i = 0; i < uploads.size(); ++i) {
            srv.Buffer.NumElements = static_cast<uint32_t>((sources[i].size() + 3) / 4);
            D3D12_CPU_DESCRIPTOR_HANDLE at{cpu.ptr + SIZE_T(i) * stride};
            device_->CreateShaderResourceView(uploads[i].resource.Get(), &srv, at);
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = static_cast<uint32_t>(outBytes / 4);
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        const SIZE_T uavAt = uploads.size();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle{cpu.ptr + uavAt * stride};
        device_->CreateUnorderedAccessView(out.resource.Get(), nullptr, &uav, uavHandle);

        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        list_->SetDescriptorHeaps(1, heaps);
        list_->SetComputeRootSignature(signature);
        list_->SetComputeRootConstantBufferView(0, params.address);
        list_->SetComputeRootDescriptorTable(1, gpu);
        list_->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{gpu.ptr + uavAt * stride});
        list_->ResourceBarrier(1, &barriers[0]);
        const UINT threads = 64;
        const UINT groups = static_cast<UINT>((static_cast<uint64_t>(rows) * cols + threads - 1) / threads);
        list_->Dispatch(groups, 1, 1);
        barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[1].Transition.pResource = out.resource.Get();
        barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        list_->ResourceBarrier(1, &barriers[1]);
        list_->CopyBufferRegion(readback.resource.Get(), 0, out.resource.Get(), 0, outBytes);
        Check(list_->Close(), "Close(list)");

        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion");
        WaitForSingleObject(event_, INFINITE);

        void* mapped = nullptr;
        Check(readback.resource->Map(0, nullptr, &mapped), "Map(readback)");
        WriteFile(output, mapped, outBytes);
        readback.resource->Unmap(0, nullptr);
        std::printf("dispatch M=%u K=%u N=%u groups=%u -> %zu bytes\n", rows, header.K, cols, groups, outBytes);
    }

    GpuBuffer Committed(size_t size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, bool uav) {
        GpuBuffer buffer;
        buffer.size = Align(static_cast<uint32_t>(size));
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = type;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = buffer.size;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (uav) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&buffer.resource)),
              "CreateCommittedResource");
        buffer.address = buffer.resource->GetGPUVirtualAddress();
        return buffer;
    }

    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    uint64_t fenceValue_ = 0;
};

int Run(int argc, char** argv) {
    std::filesystem::path casePath, outputPath, shaderPath = "gemm_f16_cs.cso";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) casePath = argv[++i];
        else if (arg == "--out" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--shader" && i + 1 < argc) shaderPath = argv[++i];
        else throw std::runtime_error("unknown option " + arg);
    }
    if (casePath.empty() || outputPath.empty()) {
        std::fprintf(stderr, "usage: m_stage_probe --case case.bin --out result.bin [--shader x.cso]\n");
        return 2;
    }

    auto bytes = ReadFile(casePath);
    if (bytes.size() < sizeof(Header)) throw std::runtime_error("case file is truncated");
    Header header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (std::memcmp(header.magic, "MSTAGE01", 8) != 0 || (header.version != 1 && header.version != 2)) {
        throw std::runtime_error("unexpected case header");
    }
    const size_t M = header.M, K = header.K, N = header.N;
    std::vector<size_t> sizes;
    if (header.version == 1) {
        sizes = {M * K * 2, K * N * 2, N * 2};             // X, W, bias (f16)
    } else {
        if (K % 4 != 0) throw std::runtime_error("INT8 case needs K divisible by 4");
        sizes = {M * K, N * K, M * 4, N * 4, N * 4};       // A, W, activation scales,
                                                           // weight scales, bias
    }
    size_t expected = sizeof(Header);
    for (size_t size : sizes) expected += size;
    if (bytes.size() != expected) throw std::runtime_error("case size does not match its header");

    std::vector<std::vector<uint8_t>> sources;
    size_t offset = sizeof(Header);
    for (size_t size : sizes) {
        sources.emplace_back(bytes.begin() + offset, bytes.begin() + offset + size);
        offset += size;
    }

    Runner runner;
    runner.Run(header, sources, shaderPath, outputPath);
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
