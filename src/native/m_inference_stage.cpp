// M16: persistent-engine harness with caller-owned frame resources.
//
// Initializes one engine from a composed pipeline manifest, uploads the frame
// inputs once, creates one caller-owned output texture, then executes frames in
// one process.  Every frame-varying image input is an external resource:
//
//   enc0_in_colour  RGBA16F 960x540   encoder colour
//   post_colour     RGBA16F 960x540   post render colour
//   post_motion     RG16F   960x540   post motion (bound, unread on reset)
//   down_colour     RGBA16F 960x540   downsample colour (alpha only)
//   frame_out       RGBA16F 1920x1080 caller-owned output (UAV)
//
// Only descriptor views and external resource states change between frames; no
// graph, pipeline, weight, descriptor heap or arena is rebuilt.
//
// Validation mode adds the readbacks needed to compare each frame with an
// independent standalone runner.  Production mode performs none.
//
// usage:
//   m_inference_stage --manifest engine.txt --shaders DIR --output frame_out \
//     --frames-file frames.json [--mode production|validation] \
//     [--dump-dir DIR] [--dump-frame 0] [--output-dump frame_out] \
//     [--recover] [--report FILE] [--debug] [--seed N] \
//     [--caller-list] [--caller-readback FILE] [--caller-cancel] \
//     [--external-device] [--external-selftest DIR]
//
//   m_inference_stage ... --input enc0_in_colour=a.bin --input post_colour=a.bin \
//     --input post_motion=m.bin --input down_colour=a.bin
//
// frames.json is a JSON array of objects mapping resource name to file:
//   [{"enc0_in_colour":"f0.bin","post_colour":"f0.bin", ...}, ...]
#include "m_inference_engine.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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

void WriteFile(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

double Seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// --------------------------------------------------------------------------
// Minimal JSON reader for the frames file.  It accepts an array of objects
// whose values are strings, or an object with a single "frames" array.  It
// rejects anything else rather than guessing.
// --------------------------------------------------------------------------
class Json {
public:
    explicit Json(std::string text) : text_(std::move(text)) {
        if (text_.size() >= 3 && static_cast<unsigned char>(text_[0]) == 0xEF &&
            static_cast<unsigned char>(text_[1]) == 0xBB &&
            static_cast<unsigned char>(text_[2]) == 0xBF)
            text_.erase(0, 3);
    }

    std::vector<std::map<std::string, std::string>> Frames() {
        std::vector<std::map<std::string, std::string>> frames;
        Skip();
        if (Peek() == '{') {
            Expect('{');
            Skip();
            if (String() != "frames")
                throw std::runtime_error("frames file object must contain \"frames\"");
            Skip();
            Expect(':');
            frames = Array();
            Skip();
            Expect('}');
        } else {
            frames = Array();
        }
        Skip();
        if (!Eof()) throw std::runtime_error("trailing data in frames file");
        return frames;
    }

private:
    bool Eof() const { return pos_ >= text_.size(); }
    char Peek() const {
        if (Eof()) throw std::runtime_error("unexpected end of frames file");
        return text_[pos_];
    }
    void Skip() {
        while (!Eof()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }
    void Expect(char c) {
        Skip();
        if (Eof() || text_[pos_] != c)
            throw std::runtime_error(std::string("expected '") + c + "' in frames file");
        ++pos_;
    }
    std::vector<std::map<std::string, std::string>> Array() {
        std::vector<std::map<std::string, std::string>> out;
        Expect('[');
        Skip();
        if (Peek() == ']') {
            ++pos_;
            return out;
        }
        for (;;) {
            out.push_back(Object());
            Skip();
            if (Peek() == ',') {
                ++pos_;
                continue;
            }
            Expect(']');
            break;
        }
        return out;
    }
    std::map<std::string, std::string> Object() {
        std::map<std::string, std::string> out;
        Expect('{');
        Skip();
        if (Peek() == '}') {
            ++pos_;
            return out;
        }
        for (;;) {
            Skip();
            const std::string key = String();
            Skip();
            Expect(':');
            Skip();
            const std::string value = String();
            if (out.count(key)) throw std::runtime_error("duplicate key in frames file: " + key);
            out[key] = value;
            Skip();
            if (Peek() == ',') {
                ++pos_;
                continue;
            }
            Expect('}');
            break;
        }
        return out;
    }
    static void AppendUtf8(std::string& out, uint32_t code) {
        if (code <= 0x7F) {
            out.push_back(static_cast<char>(code));
        } else if (code <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }
    uint32_t Hex4() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            if (Eof()) throw std::runtime_error("truncated \\u escape");
            const char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
            else throw std::runtime_error("invalid \\u escape");
        }
        return value;
    }
    std::string String() {
        Expect('"');
        std::string out;
        while (!Eof()) {
            const char c = text_[pos_++];
            if (c == '"') return out;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (Eof()) throw std::runtime_error("truncated escape");
            const char escape = text_[pos_++];
            switch (escape) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t code = Hex4();
                    if (code >= 0xD800 && code <= 0xDBFF) {
                        if (Eof() || text_[pos_] != '\\') throw std::runtime_error("lone surrogate");
                        ++pos_;
                        if (Eof() || text_[pos_] != 'u') throw std::runtime_error("lone surrogate");
                        ++pos_;
                        const uint32_t low = Hex4();
                        if (low < 0xDC00 || low > 0xDFFF)
                            throw std::runtime_error("invalid surrogate pair");
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                    AppendUtf8(out, code);
                    break;
                }
                default: throw std::runtime_error("unknown escape in frames file");
            }
        }
        throw std::runtime_error("unterminated string in frames file");
    }

    std::string text_;
    size_t pos_ = 0;
};

std::vector<std::map<std::string, std::string>> ParseFramesFile(const fs::path& path) {
    const std::vector<uint8_t> bytes = ReadFile(path);
    return Json(std::string(bytes.begin(), bytes.end())).Frames();
}

// --------------------------------------------------------------------------
// Resources
// --------------------------------------------------------------------------
struct InputSpec {
    std::string resource;
    fs::path file;
};

struct FrameInputs {
    std::vector<InputSpec> bindings;
};

struct UploadedTexture {
    ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct Options {
    fs::path manifest, shaderDir, report, dumpDir, framesFile, callerReadback, selftestDir;
    std::vector<InputSpec> inputs;
    std::string output;
    std::string outputDump = "frame_out";
    int frames = 0;
    int dumpFrame = 0;
    bool validation = false;
    bool recover = false;
    bool debug = false;
    bool callerList = false;
    bool callerCancel = false;
    bool externalDevice = false;
    bool externalSelftest = false;
    uint32_t seed = 0;
};

struct GpuMemory {
    uint64_t usage = 0, budget = 0;
};

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
                                   uint32_t width, uint32_t height,
                                   D3D12_RESOURCE_FLAGS flags =
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   uint32_t arraySize = 1, uint32_t mips = 1,
                                   uint32_t samples = 1) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = static_cast<UINT16>(arraySize);
    desc.MipLevels = static_cast<UINT16>(mips);
    desc.Format = format;
    desc.SampleDesc.Count = samples;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                          IID_PPV_ARGS(&resource)),
          "CreateCommittedResource(texture)");
    return resource;
}

struct StandaloneDevice {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGIAdapter1> adapter;
    std::string name;
};

StandaloneDevice CreateStandaloneDevice(bool debug) {
    StandaloneDevice out;
    UINT flags = 0;
    if (debug) {
        ComPtr<ID3D12Debug> debugInterface;
        Check(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)),
              "D3D12GetDebugInterface");
        debugInterface->EnableDebugLayer();
        flags |= DXGI_CREATE_FACTORY_DEBUG;
    }
    ComPtr<IDXGIFactory6> factory;
    Check(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapterByGpuPreference(
                i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) ==
            DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                        IID_PPV_ARGS(&out.device)))) {
            out.adapter = adapter;
            char name[256];
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name),
                                nullptr, nullptr);
            out.name = name;
            break;
        }
    }
    if (!out.device) throw std::runtime_error("no D3D12 device at feature level 12_0");
    D3D12_COMMAND_QUEUE_DESC queue{};
    queue.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Check(out.device->CreateCommandQueue(&queue, IID_PPV_ARGS(&out.queue)),
          "CreateCommandQueue");
    return out;
}

// Every frame input is uploaded once, before the first frame, through one
// command list.  The caller owns these textures and their initial state.
std::vector<UploadedTexture> UploadTextures(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const std::vector<std::pair<const sette::EngineResource*, fs::path>>& files) {
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
        const fs::path& file = item.second;
        if (spec->kind != sette::EngineResource::Kind::Texture)
            throw std::runtime_error("harness inputs must be textures: " + spec->name);
        const std::vector<uint8_t> data = ReadFile(file);
        if (data.size() != spec->bytes)
            throw std::runtime_error("input size mismatch: " + file.string());
        const uint32_t pixelBytes =
            static_cast<uint32_t>(spec->bytes / (spec->width * spec->height));
        UploadedTexture texture;
        texture.resource = MakeTexture(device, spec->format, spec->width, spec->height);
        const D3D12_RESOURCE_DESC desc = texture.resource->GetDesc();
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
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
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
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

void WriteReport(const fs::path& path, const Options& opt, const PersistentEngine& engine,
                 const std::vector<PersistentEngine::FrameStats>& runs,
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
    out << "  \"recording\": \"" << (opt.callerList ? "caller-owned" : "engine-owned")
        << "\",\n";
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
            << ", \"aliasing_barriers\": " << r.aliasingBarriers
            << ", \"external_rebinds\": " << r.externalRebinds
            << ", \"engine_submitted\": " << (r.engineSubmitted ? "true" : "false")
            << ", \"engine_waited\": " << (r.engineWaited ? "true" : "false") << "}"
            << (i + 1 == runs.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

// --------------------------------------------------------------------------
// Negative conformance cases.  Each case states whether the engine must accept
// or reject, and the observed outcome is recorded rather than thrown.
// --------------------------------------------------------------------------
struct CaseResult {
    std::string name;
    std::string expected;
    std::string observed;
    std::string message;
};

using Specs = std::map<std::string, const sette::EngineResource*>;

PersistentEngine::FrameRequest ValidRequest(
    const std::vector<FrameInputs>& frames,
    const std::vector<UploadedTexture>& textures, const std::vector<size_t>& start,
    const std::string& output, ID3D12Resource* outputResource,
    const std::vector<D3D12_RESOURCE_STATES>& inputStates,
    D3D12_RESOURCE_STATES outputState) {
    PersistentEngine::FrameRequest request;
    const size_t base = start.at(0);
    for (size_t i = 0; i < frames.at(0).bindings.size(); ++i) {
        request.inputs.push_back({frames[0].bindings[i].resource,
                                  textures[base + i].resource.Get(),
                                  inputStates[i]});
    }
    request.outputs.push_back({output, outputResource, outputState});
    return request;
}

std::vector<CaseResult> RunExternalSelftest(
    PersistentEngine& engine, const Specs& specs, const std::vector<FrameInputs>& frames,
    const std::vector<UploadedTexture>& textures, const std::vector<size_t>& start,
    const std::string& output, ID3D12Resource* outputResource,
    std::vector<D3D12_RESOURCE_STATES>& firstStates,
    D3D12_RESOURCE_STATES& outState, const fs::path& manifest,
    const fs::path& shaderDir, const fs::path& selftestDir) {
    std::vector<CaseResult> results;

    auto record = [&](const std::string& name, const std::string& expected,
                      const std::string& observed, const std::string& message) {
        results.push_back({name, expected, observed, message});
    };
    auto expectPass = [&](const std::string& name, const std::function<void()>& action) {
        try {
            action();
            record(name, "accept", "accept", "");
        } catch (const std::exception& error) {
            record(name, "accept", "reject", error.what());
        }
    };
    auto expectReject = [&](const std::string& name, const std::function<void()>& action) {
        try {
            action();
            record(name, "reject", "accept", "engine accepted an invalid resource");
        } catch (const std::exception& error) {
            record(name, "reject", "reject", error.what());
        }
    };

    const auto& inputSpec = *specs.at("enc0_in_colour");
    ID3D12Resource* validInput = textures.at(start.at(0)).resource.Get();
    ID3D12Resource* validOutput = outputResource;

    expectPass("bind_valid_inputs", [&]() {
        engine.SetExternal({"enc0_in_colour", validInput, D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("null_input", [&]() {
        engine.SetExternal({"enc0_in_colour", nullptr, D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("wrong_width", [&]() {
        auto bad = MakeTexture(engine.device(), inputSpec.format,
                               inputSpec.width - 1, inputSpec.height);
        engine.SetExternal({"enc0_in_colour", bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("wrong_height", [&]() {
        auto bad = MakeTexture(engine.device(), inputSpec.format,
                               inputSpec.width, inputSpec.height - 1);
        engine.SetExternal({"enc0_in_colour", bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("wrong_format", [&]() {
        auto bad = MakeTexture(engine.device(), DXGI_FORMAT_R8G8B8A8_UNORM,
                               inputSpec.width, inputSpec.height);
        engine.SetExternal({"enc0_in_colour", bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("array_size_two", [&]() {
        auto bad = MakeTexture(engine.device(), inputSpec.format, inputSpec.width,
                               inputSpec.height, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               2, 1, 1);
        engine.SetExternal({"enc0_in_colour", bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("mip_levels_two", [&]() {
        auto bad = MakeTexture(engine.device(), inputSpec.format, inputSpec.width,
                               inputSpec.height, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               1, 2, 1);
        engine.SetExternal({"enc0_in_colour", bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("sample_count_two", [&]() {
        // A multisampled resource is legal, but not as a compute SRV here.
        auto bad = MakeTexture(engine.device(), inputSpec.format, inputSpec.width,
                               inputSpec.height,
                               D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 1, 1, 2);
        engine.SetExternal({"enc0_in_colour", bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("output_without_uav", [&]() {
        auto bad = MakeTexture(engine.device(), specs.at(output)->format,
                               specs.at(output)->width, specs.at(output)->height,
                               D3D12_RESOURCE_FLAG_NONE);
        engine.SetExternal({output, bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("output_wrong_format", [&]() {
        auto bad = MakeTexture(engine.device(), DXGI_FORMAT_R8G8B8A8_UNORM,
                               specs.at(output)->width, specs.at(output)->height);
        engine.SetExternal({output, bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("missing_required_resource", [&]() {
        PersistentEngine::FrameRequest request = ValidRequest(
            frames, textures, start, output, validOutput, firstStates, outState);
        request.inputs.pop_back();
        engine.ExecuteFrame(request);
    });
    expectReject("output_bound_as_input", [&]() {
        PersistentEngine::FrameRequest request = ValidRequest(
            frames, textures, start, output, validOutput, firstStates, outState);
        request.inputs.push_back({output, validOutput, outState});
        engine.ExecuteFrame(request);
    });
    expectReject("foreign_device_resource", [&]() {
        // A second hardware device can be the same runtime object on this
        // driver, so use WARP: its device pointer is genuinely different and
        // the engine must reject a resource created on it.
        ComPtr<IDXGIFactory4> factory;
        Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1(warp)");
        ComPtr<IDXGIAdapter> warp;
        Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "EnumWarpAdapter");
        ComPtr<ID3D12Device> warpDevice;
        Check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0,
                                IID_PPV_ARGS(&warpDevice)),
              "D3D12CreateDevice(warp)");
        auto bad = MakeTexture(warpDevice.Get(), inputSpec.format,
                               inputSpec.width, inputSpec.height);
        engine.SetExternal({"enc0_in_colour", bad.Get(), D3D12_RESOURCE_STATE_COMMON});
    });
    expectReject("malformed_manifest", [&]() {
        fs::create_directories(selftestDir);
        const fs::path badManifest = selftestDir / "malformed-engine.txt";
        WriteFile(badManifest, "version 3\nresource broken\n");
        PersistentEngine failed;
        PersistentEngine::Config config;
        config.manifest = badManifest.string();
        config.shaderDir = shaderDir.string();
        failed.Initialize(config);
    });

    // Once every rejection has happened, a complete valid frame must still run
    // and the debug layer must report no error or corruption message.
    expectPass("valid_frame_after_rejections", [&]() {
        PersistentEngine::FrameRequest request = ValidRequest(
            frames, textures, start, output, validOutput, firstStates, outState);
        engine.ExecuteFrame(request);
        for (auto& state : firstStates)
            state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        outState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    });
    const auto debug = engine.debugCounts();
    std::string message = "errors=" + std::to_string(debug.errors) +
                          " warnings=" + std::to_string(debug.warnings);
    if (debug.total) {
        for (const std::string& text : engine.debugMessages()) message += " | " + text;
    }
    record("zero_debug_errors", "0", std::to_string(debug.errors), message);
    return results;
}

// --------------------------------------------------------------------------
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
        } else if (arg == "--frames-file") opt.framesFile = next();
        else if (arg == "--output") opt.output = next();
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
        else if (arg == "--caller-cancel") opt.callerCancel = true;
        else if (arg == "--external-device") opt.externalDevice = true;
        else if (arg == "--external-selftest") {
            opt.externalSelftest = true;
            opt.selftestDir = next();
        } else if (arg == "--seed") {
            opt.seed = static_cast<uint32_t>(std::stoul(next()));
        } else {
            throw std::runtime_error("unknown option " + arg);
        }
    }
    if (opt.manifest.empty()) throw std::runtime_error("--manifest is required");
    if (opt.output.empty()) throw std::runtime_error("--output is required");
    if (opt.inputs.empty() && opt.framesFile.empty())
        throw std::runtime_error("--frames-file or at least one --input is required");
    if (!opt.inputs.empty() && !opt.framesFile.empty())
        throw std::runtime_error("--input and --frames-file are mutually exclusive");
    if (opt.callerCancel && opt.validation)
        throw std::runtime_error("--caller-cancel cannot use validation dumps");
    if (opt.externalDevice && !opt.callerList && !opt.callerCancel)
        throw std::runtime_error("--external-device requires --caller-list or --caller-cancel");
    if (opt.validation && opt.dumpDir.empty())
        throw std::runtime_error("validation mode needs --dump-dir");
    if (!opt.dumpDir.empty()) fs::create_directories(opt.dumpDir);

    std::vector<FrameInputs> frames;
    if (!opt.framesFile.empty()) {
        const auto objects = ParseFramesFile(opt.framesFile);
        if (objects.empty()) throw std::runtime_error("frames file has no frames");
        for (const auto& object : objects) {
            FrameInputs frame;
            for (const auto& entry : object)
                frame.bindings.push_back({entry.first, entry.second});
            frames.push_back(std::move(frame));
        }
    } else {
        FrameInputs frame;
        for (const InputSpec& input : opt.inputs) frame.bindings.push_back(input);
        frames.push_back(std::move(frame));
    }
    if (opt.frames <= 0) opt.frames = static_cast<int>(frames.size());
    if (opt.frames > static_cast<int>(frames.size()))
        throw std::runtime_error("--frames exceeds the number of frames");
    if (opt.dumpFrame < 0 || opt.dumpFrame >= opt.frames)
        throw std::runtime_error("--dump-frame is outside the executed frames");

    PersistentEngine engine;
    PersistentEngine::Config config;
    config.manifest = opt.manifest.string();
    config.shaderDir = opt.shaderDir.string();
    config.mode = opt.validation ? PersistentEngine::Mode::Validation
                                 : PersistentEngine::Mode::Production;
    config.debug = opt.debug;
    config.seed = opt.seed;

    StandaloneDevice standalone;
    IDXGIAdapter1* adapter = nullptr;
    if (opt.externalDevice) {
        standalone = CreateStandaloneDevice(opt.debug);
        engine.Initialize(standalone.device.Get(), standalone.queue.Get(), config);
        adapter = standalone.adapter.Get();
    } else {
        engine.Initialize(config);
        adapter = engine.adapter();
    }
    const GpuMemory afterInit = SampleAdapter(adapter);

    Specs specs;
    for (const auto& resource : engine.resources())
        if (resource.external) specs[resource.name] = &resource;
    if (!specs.count(opt.output))
        throw std::runtime_error("output resource is not external: " + opt.output);

    std::vector<std::pair<const sette::EngineResource*, fs::path>> uploads;
    std::vector<size_t> start;
    for (const auto& frame : frames) {
        start.push_back(uploads.size());
        for (const auto& binding : frame.bindings) {
            auto it = specs.find(binding.resource);
            if (it == specs.end())
                throw std::runtime_error("input resource is not external: " +
                                         binding.resource);
            uploads.push_back({it->second, binding.file});
        }
    }
    auto textures = UploadTextures(engine.device(), engine.queue(), uploads);
    if (textures.size() != uploads.size())
        throw std::runtime_error("input upload count mismatch");
    const sette::EngineResource* outSpec = specs.at(opt.output);
    UploadedTexture output;
    output.resource = MakeTexture(engine.device(), outSpec->format, outSpec->width,
                                  outSpec->height);

    std::vector<std::vector<D3D12_RESOURCE_STATES>> inputStates(
        frames.size(), std::vector<D3D12_RESOURCE_STATES>());
    for (size_t f = 0; f < frames.size(); ++f)
        inputStates[f].assign(frames[f].bindings.size(), D3D12_RESOURCE_STATE_COMMON);
    std::vector<D3D12_RESOURCE_STATES> outState(1, D3D12_RESOURCE_STATE_COMMON);
    std::vector<PersistentEngine::FrameStats> runs;
    std::vector<std::string> labels;

    auto makeRequest = [&](int frameIndex) {
        PersistentEngine::FrameRequest request;
        const size_t base = start.at(static_cast<size_t>(frameIndex));
        for (size_t i = 0; i < frames[frameIndex].bindings.size(); ++i) {
            request.inputs.push_back({frames[frameIndex].bindings[i].resource,
                                      textures[base + i].resource.Get(),
                                      inputStates[frameIndex][i]});
        }
        request.outputs.push_back({opt.output, output.resource.Get(), outState[0]});
        return request;
    };
    auto afterFrame = [&](int frameIndex) {
        for (size_t i = 0; i < inputStates[frameIndex].size(); ++i)
            inputStates[frameIndex][i] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        outState[0] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    };

    auto frameName = [](int index) {
        char text[32];
        std::snprintf(text, sizeof(text), "frame-%02d", index);
        return std::string(text);
    };

    if (opt.externalSelftest) {
        const auto results = RunExternalSelftest(
            engine, specs, frames, textures, start, opt.output, output.resource.Get(),
            inputStates[0], outState[0], opt.manifest, opt.shaderDir, opt.selftestDir);
        bool pass = true;
        for (const auto& result : results)
            if (result.expected != result.observed) pass = false;
        std::string json = "{\n  \"status\": \"" +
                           std::string(pass ? "PASS" : "FAIL") + "\",\n  \"cases\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            json += "    {\"name\": \"" + JsonEscape(result.name) +
                    "\", \"expected\": \"" + result.expected +
                    "\", \"observed\": \"" + result.observed +
                    "\", \"message\": \"" + JsonEscape(result.message) + "\"}";
            json += (i + 1 == results.size() ? "\n" : ",\n");
        }
        json += "  ]\n}\n";
        WriteFile(opt.selftestDir / "external-selftest.json", json);
        std::printf("external selftest: %s (%zu cases)\n", pass ? "PASS" : "FAIL",
                    results.size());
        for (const auto& result : results)
            if (result.expected != result.observed)
                std::printf("  FAIL %s: expected %s, observed %s (%s)\n",
                            result.name.c_str(), result.expected.c_str(),
                            result.observed.c_str(), result.message.c_str());
        engine.Destroy();
        return pass ? 0 : 1;
    }

    if (opt.callerCancel) {
        // Record one frame on a caller-owned list and discard it without
        // submission.  The engine must not close, execute or wait on the list.
        ComPtr<ID3D12CommandAllocator> calloc;
        ComPtr<ID3D12GraphicsCommandList> clist;
        Check(engine.device()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                      IID_PPV_ARGS(&calloc)),
              "CreateCommandAllocator(cancel)");
        Check(engine.device()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                 calloc.Get(), nullptr,
                                                 IID_PPV_ARGS(&clist)),
              "CreateCommandList(cancel)");
        Check(clist->Close(), "Close(cancel init)");
        Check(calloc->Reset(), "Reset(cancel allocator)");
        Check(clist->Reset(calloc.Get(), nullptr), "Reset(cancel list)");
        PersistentEngine::FrameRequest request = makeRequest(0);
        PersistentEngine::FrameStats stats = engine.ExecuteFrame(request, clist.Get());
        if (stats.engineSubmitted || stats.engineWaited)
            throw std::runtime_error("engine submitted a borrowed list");
        Check(clist->Close(), "Close(cancel list)");
        Check(calloc->Reset(), "Reset(cancel allocator after discard)");
        runs.push_back(stats);
        labels.push_back("cancelled-frame-0");
        std::printf("caller cancellation: recorded %llu dispatches, not submitted\n",
                    static_cast<unsigned long long>(stats.dispatches));
    } else if (!opt.callerList) {
        for (int frame = 0; frame < opt.frames; ++frame) {
            PersistentEngine::FrameRequest request = makeRequest(frame);
            if (opt.validation) {
                request.dumpDir =
                    (fs::path(opt.dumpDir) / frameName(frame)).string();
                request.dumps = frame == opt.dumpFrame
                                    ? std::vector<std::string>{}
                                    : std::vector<std::string>{opt.outputDump};
            }
            PersistentEngine::FrameStats stats = engine.ExecuteFrame(request);
            afterFrame(frame);
            runs.push_back(stats);
            labels.push_back(frameName(frame));
        }
        if (opt.recover) {
            PersistentEngine::FrameRequest request = makeRequest(0);
            if (opt.validation) {
                request.dumpDir = (fs::path(opt.dumpDir) / "recover").string();
                request.dumps = std::vector<std::string>{opt.outputDump};
            }
            PersistentEngine::FrameStats stats = engine.ExecuteFrame(request);
            afterFrame(0);
            runs.push_back(stats);
            labels.push_back("recover-frame-0");
        }
    } else {
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
        auto runBorrowed = [&](int frame, const std::string& label) {
            PersistentEngine::FrameRequest request = makeRequest(frame);
            Check(calloc->Reset(), "Reset(caller allocator)");
            Check(clist->Reset(calloc.Get(), nullptr), "Reset(caller list)");
            PersistentEngine::FrameStats stats = engine.ExecuteFrame(request, clist.Get());
            if (stats.engineSubmitted || stats.engineWaited)
                throw std::runtime_error("engine submitted a borrowed list");
            Check(clist->Close(), "Close(caller list)");
            ID3D12CommandList* lists[] = {clist.Get()};
            engine.queue()->ExecuteCommandLists(1, lists);
            Check(engine.queue()->Signal(cfence.Get(), ++fenceValue), "Signal(caller)");
            Check(cfence->SetEventOnCompletion(fenceValue, cevent),
                  "SetEventOnCompletion(caller)");
            WaitForSingleObject(cevent, INFINITE);
            afterFrame(frame);
            runs.push_back(stats);
            labels.push_back(label);
        };
        for (int frame = 0; frame < opt.frames; ++frame)
            runBorrowed(frame, frameName(frame));
        if (opt.recover) runBorrowed(0, "recover-frame-0");
        if (!opt.callerReadback.empty())
            ReadbackTexture(engine.device(), engine.queue(), output.resource.Get(),
                            outSpec->width, outSpec->height, opt.callerReadback);
        CloseHandle(cevent);
    }

    const GpuMemory after = SampleAdapter(adapter);
    const auto debug = engine.debugCounts();
    if (opt.debug && !opt.callerCancel && debug.errors) {
        throw std::runtime_error("D3D12 debug layer reported " +
                                 std::to_string(debug.errors) + " error(s)");
    }
    if (opt.debug) {
        std::printf("D3D12 debug: %llu error(s), %llu warning(s), %llu message(s)\n",
                    static_cast<unsigned long long>(debug.errors),
                    static_cast<unsigned long long>(debug.warnings),
                    static_cast<unsigned long long>(debug.total));
    }
    if (!opt.report.empty())
        WriteReport(opt.report, opt, engine, runs, labels, afterInit, after);
    std::printf("engine: %zu stage(s), %zu resource(s), %zu frame(s), %zu external(s)\n",
                engine.stages().size(), engine.resources().size(), runs.size(),
                specs.size());
    for (size_t i = 0; i < runs.size(); ++i)
        std::printf("%s: gpu %.3f ms, record %.3f ms, wait %.3f ms, %llu dispatches, "
                    "%llu rebinds\n",
                    labels[i].c_str(), runs[i].gpuIntervalMs, runs[i].recordMs,
                    runs[i].submitWaitMs,
                    static_cast<unsigned long long>(runs[i].dispatches),
                    static_cast<unsigned long long>(runs[i].externalRebinds));
    engine.Destroy();
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
