// M15-C: persistent D3D12 inference engine implementation.
#include "m_inference_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

namespace sette {
namespace {

constexpr uint32_t kAlign = 256;
constexpr uint64_t kPlacement = 65536;
constexpr uint64_t kUploadBatch = 64ull << 20;

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
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.flush();
    if (!out) throw std::runtime_error("incomplete write: " + path.string());
}

uint32_t Align(uint64_t value) {
    return static_cast<uint32_t>((value + kAlign - 1) & ~uint64_t(kAlign - 1));
}

uint64_t AlignPlacement(uint64_t value) {
    return (value + kPlacement - 1) & ~(kPlacement - 1);
}

double Seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct FormatInfo {
    DXGI_FORMAT format;
    uint32_t pixelBytes;
};

const std::unordered_map<std::string, FormatInfo>& FormatTable() {
    static const std::unordered_map<std::string, FormatInfo> table = {
        {"R16_FLOAT", {DXGI_FORMAT_R16_FLOAT, 2}},
        {"R16G16_FLOAT", {DXGI_FORMAT_R16G16_FLOAT, 4}},
        {"R16G16B16A16_FLOAT", {DXGI_FORMAT_R16G16B16A16_FLOAT, 8}},
        {"R32_FLOAT", {DXGI_FORMAT_R32_FLOAT, 4}},
        {"R32G32_FLOAT", {DXGI_FORMAT_R32G32_FLOAT, 8}},
        {"R32G32B32A32_FLOAT", {DXGI_FORMAT_R32G32B32A32_FLOAT, 16}},
        {"R8G8B8A8_UNORM", {DXGI_FORMAT_R8G8B8A8_UNORM, 4}},
        {"R8G8B8A8_UINT", {DXGI_FORMAT_R8G8B8A8_UINT, 4}},
        {"R32_UINT", {DXGI_FORMAT_R32_UINT, 4}},
    };
    return table;
}

const FormatInfo& LookupFormat(const std::string& name) {
    auto it = FormatTable().find(name);
    if (it == FormatTable().end()) throw std::runtime_error("unknown format " + name);
    return it->second;
}

std::string FormatName(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R16_FLOAT: return "R16_FLOAT";
        case DXGI_FORMAT_R16G16_FLOAT: return "R16G16_FLOAT";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R32_FLOAT: return "R32_FLOAT";
        case DXGI_FORMAT_R32G32_FLOAT: return "R32G32_FLOAT";
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UINT: return "R8G8B8A8_UINT";
        case DXGI_FORMAT_R32_UINT: return "R32_UINT";
        default: return "UNKNOWN";
    }
}

D3D12_RESOURCE_DESC BufferDesc(uint64_t bytes) {
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

D3D12_RESOURCE_DESC TextureDesc(DXGI_FORMAT format, uint32_t width, uint32_t height) {
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
    return desc;
}

bool IsTexture(const EngineResource& r) {
    return r.kind == EngineResource::Kind::Texture;
}

}  // namespace

// --------------------------------------------------------------------------
// FrameTimer: one query heap and one readback buffer reused across frames, so
// measuring a frame never allocates a resource.
// --------------------------------------------------------------------------
class PersistentEngine::FrameTimer {
public:
    FrameTimer(ID3D12Device* device, ID3D12CommandQueue* queue, size_t stages)
        : capacity_(stages + 8), stages_(stages) {
        D3D12_QUERY_HEAP_DESC heap{};
        heap.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heap.Count = static_cast<UINT>(capacity_ * 2);
        Check(device->CreateQueryHeap(&heap, IID_PPV_ARGS(&heap_)),
              "CreateQueryHeap(timestamp)");
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = capacity_ * 2 * sizeof(uint64_t);
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES props{};
        props.Type = D3D12_HEAP_TYPE_READBACK;
        Check(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                              IID_PPV_ARGS(&readback_)),
              "CreateCommittedResource(timing)");
        Check(queue->GetTimestampFrequency(&frequency_), "GetTimestampFrequency");
    }

    void Reset() {
        count_ = 0;
        names_.clear();
        values_.clear();
    }

    void BeginGlobal(ID3D12GraphicsCommandList* list) {
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(capacity_ * 2 - 2));
    }
    void EndGlobal(ID3D12GraphicsCommandList* list) {
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(capacity_ * 2 - 1));
    }
    void Begin(ID3D12GraphicsCommandList* list) {
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(count_ * 2));
    }
    void End(ID3D12GraphicsCommandList* list, const std::string& name) {
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(count_ * 2 + 1));
        names_.push_back(name);
        ++count_;
    }
    void Resolve(ID3D12GraphicsCommandList* list) {
        if (!count_) return;
        list->ResolveQueryData(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                               static_cast<UINT>(count_ * 2), readback_.Get(), 0);
        list->ResolveQueryData(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                               static_cast<UINT>(capacity_ * 2 - 2), 2,
                               readback_.Get(),
                               static_cast<uint64_t>(capacity_ * 2 - 2) * sizeof(uint64_t));
    }

    // Call only after the submission that resolved the queries has completed.
    bool Read(double& globalMs, std::vector<double>& stageMs,
              std::vector<std::string>& names) {
        if (!count_) return false;
        void* mapped = nullptr;
        if (FAILED(readback_->Map(0, nullptr, &mapped))) return false;
        const auto* stamps = static_cast<const uint64_t*>(mapped);
        stageMs.clear();
        stageMs.reserve(count_);
        for (size_t i = 0; i < count_; ++i) {
            const uint64_t begin = stamps[i * 2], end = stamps[i * 2 + 1];
            stageMs.push_back(end >= begin && frequency_
                                  ? double(end - begin) * 1000.0 / double(frequency_)
                                  : 0.0);
        }
        const uint64_t gb = stamps[capacity_ * 2 - 2];
        const uint64_t ge = stamps[capacity_ * 2 - 1];
        globalMs = ge >= gb && frequency_
                       ? double(ge - gb) * 1000.0 / double(frequency_)
                       : 0.0;
        names = names_;
        readback_->Unmap(0, nullptr);
        return true;
    }

private:
    ComPtr<ID3D12QueryHeap> heap_;
    ComPtr<ID3D12Resource> readback_;
    uint64_t frequency_ = 0;
    size_t capacity_ = 0, stages_ = 0, count_ = 0;
    std::vector<std::string> names_;
    std::vector<double> values_;
};

// --------------------------------------------------------------------------
// Lifecycle
// --------------------------------------------------------------------------
PersistentEngine::PersistentEngine() = default;

PersistentEngine::~PersistentEngine() {
    Destroy();
}

void PersistentEngine::Initialize(const Config& config) {
    mode_ = config.mode;
    seed_ = config.seed;
    manifest_ = config.manifest;
    shaderDir_ = config.shaderDir;
    debug_ = config.debug;
    arenaEnabled_ = config.arena;
    ownsQueue_ = true;
    ownsDevice_ = true;

    const auto t0 = std::chrono::steady_clock::now();
    CreateDevice();
    const auto t1 = std::chrono::steady_clock::now();
    initStats_.parseMs = 0.0;
    Parse();
    Analyze();
    PlanArena();
    const auto t2 = std::chrono::steady_clock::now();
    CreateResources();
    const auto t3 = std::chrono::steady_clock::now();
    PrepareStages();
    CreateReadbacks();
    const auto t4 = std::chrono::steady_clock::now();
    UploadInitData();
    const auto t5 = std::chrono::steady_clock::now();
    FlushInit();
    const auto t6 = std::chrono::steady_clock::now();

    // The timer is created after the stage count is known.
    timer_ = std::make_unique<FrameTimer>(device_.Get(), queue_.Get(), stages_.size());

    initStats_.parseMs = Seconds(t1, t2) * 1000.0;
    initStats_.resourceMs = Seconds(t2, t3) * 1000.0;
    initStats_.pipelineMs = Seconds(t3, t4) * 1000.0;
    initStats_.uploadWaitMs = Seconds(t5, t6) * 1000.0;
    initStats_.totalMs = Seconds(t0, t6) * 1000.0;
    initStats_.stages = stages_.size();
    initStats_.resources = resources_.size();
    initialized_ = true;
}

void PersistentEngine::Initialize(ID3D12Device* device, ID3D12CommandQueue* queue,
                                  const Config& config) {
    if (!device) throw std::runtime_error("external device is null");
    if (!queue) throw std::runtime_error("external initialization requires a caller queue");
    mode_ = config.mode;
    seed_ = config.seed;
    manifest_ = config.manifest;
    shaderDir_ = config.shaderDir;
    debug_ = config.debug;
    arenaEnabled_ = config.arena;
    ownsQueue_ = false;
    ownsDevice_ = false;
    device_ = device;
    queue_ = queue;
    if (debug_) device_.As(&infoQueue_);
    // An external queue means the engine records on caller lists; it still owns
    // an allocator and list for initialization uploads.  The queue is borrowed:
    // the engine never closes or resets it, and Destroy does not drain it.
    Check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                          IID_PPV_ARGS(&allocator_)),
          "CreateCommandAllocator");
    Check(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator_.Get(),
                                     nullptr, IID_PPV_ARGS(&list_)),
          "CreateCommandList");
    Check(list_->Close(), "Close");
    Check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
          "CreateFence");
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event_) throw std::runtime_error("CreateEventW failed");

    const auto t0 = std::chrono::steady_clock::now();
    Parse();
    Analyze();
    PlanArena();
    const auto t2 = std::chrono::steady_clock::now();
    CreateResources();
    const auto t3 = std::chrono::steady_clock::now();
    PrepareStages();
    CreateReadbacks();
    const auto t4 = std::chrono::steady_clock::now();
    UploadInitData();
    const auto t5 = std::chrono::steady_clock::now();
    FlushInit();
    const auto t6 = std::chrono::steady_clock::now();
    initStats_.parseMs = Seconds(t0, t2) * 1000.0;
    initStats_.resourceMs = Seconds(t2, t3) * 1000.0;
    initStats_.pipelineMs = Seconds(t3, t4) * 1000.0;
    initStats_.uploadWaitMs = Seconds(t5, t6) * 1000.0;
    initStats_.totalMs = Seconds(t0, t6) * 1000.0;
    initStats_.stages = stages_.size();
    initStats_.resources = resources_.size();
    initialized_ = true;
}

void PersistentEngine::Destroy() {
    if (event_) {
        CloseHandle(event_);
        event_ = nullptr;
    }
    if (device_) {
        // Drain before releasing anything a command list may still reference.
        if (ownsQueue_ && queue_ && fence_ && !stages_.empty()) {
            try {
                Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal(destroy)");
                Check(fence_->SetEventOnCompletion(fenceValue_, event_),
                      "SetEventOnCompletion(destroy)");
                WaitForSingleObject(event_, INFINITE);
            } catch (...) {
            }
        }
    }
    timer_.reset();
    readbacks_.clear();
    externals_.clear();
    externalSlots_.clear();
    stages_.clear();
    resources_.clear();
    dumps_.clear();
    index_.clear();
    arenaCanary_.Reset();
    arenaHeap_.Reset();
    list_.Reset();
    allocator_.Reset();
    fence_.Reset();
    queue_.Reset();
    infoQueue_.Reset();
    adapter_.Reset();
    device_.Reset();
    initialized_ = false;
}

void PersistentEngine::CreateDevice() {
    UINT flags = 0;
    if (debug_) {
        ComPtr<ID3D12Debug> debug;
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
            char name[256];
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name),
                                nullptr, nullptr);
            initStats_.deviceName = name;
            break;
        }
    }
    if (!device_) throw std::runtime_error("no D3D12 device at feature level 12_0");
    if (debug_) Check(device_.As(&infoQueue_), "ID3D12InfoQueue");

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
    Check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
          "CreateFence");
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event_) throw std::runtime_error("CreateEventW failed");
}

// --------------------------------------------------------------------------
// Manifest parsing
// --------------------------------------------------------------------------
void PersistentEngine::Parse() {
    index_.clear();
    const fs::path manifest = configManifestPath();
    std::ifstream in(manifest);
    if (!in) throw std::runtime_error("cannot open " + manifest.string());
    const fs::path base = manifest.parent_path();
    auto resolve = [&](const std::string& name) {
        fs::path p(name);
        return p.is_absolute() ? p : base / p;
    };
    EngineStage* stage = nullptr;
    std::string line;
    int version = 0;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::string key;
        if (!(words >> key) || key.empty() || key[0] == '#') continue;
        if (key == "version") {
            words >> version;
        } else if (key == "resource") {
            EngineResource r;
            std::string kind;
            words >> r.name >> kind;
            if (kind == "buffer") {
                r.kind = EngineResource::Kind::Buffer;
                words >> r.bytes;
                r.format = DXGI_FORMAT_R32_UINT;
                if (r.bytes == 0 || (r.bytes % 4))
                    throw std::runtime_error("buffer size must be 32-bit aligned: " + r.name);
            } else if (kind == "tex2d") {
                r.kind = EngineResource::Kind::Texture;
                std::string format;
                words >> format >> r.width >> r.height;
                const FormatInfo& info = LookupFormat(format);
                r.format = info.format;
                r.bytes = uint64_t(r.width) * r.height * info.pixelBytes;
            } else {
                throw std::runtime_error("unknown resource kind " + kind);
            }
            std::string tail;
            if (words >> tail) {
                if (tail == "external") {
                    r.external = true;
                } else if (tail == "committed") {
                    r.committed = true;
                } else {
                    r.init = resolve(tail).string();
                }
            }
            if (index_.count(r.name)) throw std::runtime_error("duplicate resource " + r.name);
            index_[r.name] = resources_.size();
            resources_.push_back(std::move(r));
            stage = nullptr;
        } else if (key == "stage") {
            EngineStage s;
            words >> s.name;
            stages_.push_back(std::move(s));
            stage = &stages_.back();
        } else if (key == "dump") {
            EngineDump d;
            words >> d.resource >> d.file;
            dumps_.push_back(std::move(d));
            stage = nullptr;
        } else if (!stage) {
            throw std::runtime_error("key outside a stage: " + key);
        } else if (key == "shader") {
            words >> stage->shader;
            if (!fs::exists(stage->shader) && !shaderDir_.empty()) {
                fs::path p(stage->shader);
                stage->shader = (fs::path(shaderDir_) / p.filename()).string();
            }
        } else if (key == "threads") {
            words >> stage->threads[0] >> stage->threads[1] >> stage->threads[2];
        } else if (key == "groups") {
            words >> stage->groups[0] >> stage->groups[1] >> stage->groups[2];
        } else if (key == "params") {
            std::string file;
            uint64_t bytes = 0;
            words >> file >> bytes;
            stage->params = ReadFile(resolve(file));
            if (stage->params.size() != bytes)
                throw std::runtime_error("params size mismatch for " + stage->name);
        } else if (key == "copy") {
            std::string kind, src, dst;
            words >> kind >> src >> dst;
            if (kind != "tex2d")
                throw std::runtime_error("only tex2d byte copies are supported");
            if (!index_.count(src) || !index_.count(dst))
                throw std::runtime_error("copy of an unknown resource");
            stage->copy = true;
            EngineBinding a;
            a.input = true;
            a.kind = EngineResource::Kind::Texture;
            a.resource = src;
            EngineBinding b;
            b.input = false;
            b.kind = EngineResource::Kind::Texture;
            b.resource = dst;
            stage->bindings.push_back(a);
            stage->bindings.push_back(b);
        } else if (key == "srv" || key == "uav") {
            EngineBinding b;
            b.input = key == "srv";
            std::string kind;
            words >> kind >> b.reg >> b.resource;
            if (kind == "buffer") b.kind = EngineResource::Kind::Buffer;
            else if (kind == "tex2d") b.kind = EngineResource::Kind::Texture;
            else throw std::runtime_error("unknown binding kind " + kind);
            if (!index_.count(b.resource))
                throw std::runtime_error("unknown resource " + b.resource);
            stage->bindings.push_back(std::move(b));
        } else {
            throw std::runtime_error("unknown manifest key " + key);
        }
    }
    if (version != 3) throw std::runtime_error("unsupported engine manifest version");
    if (stages_.empty()) throw std::runtime_error("engine manifest has no stages");
}

void PersistentEngine::Analyze() {
    const int stageCount = static_cast<int>(stages_.size());
    for (EngineStage& s : stages_) {
        uint32_t srv = 0, uav = 0;
        for (EngineBinding& b : s.bindings) {
            const uint32_t expected = b.input ? srv++ : uav++;
            if (b.reg != expected)
                throw std::runtime_error("bindings must be in register order: " + s.name);
            EngineResource& r = resources_[index_.at(b.resource)];
            if (r.kind != b.kind)
                throw std::runtime_error("binding kind mismatch: " + s.name + " / " + b.resource);
            r.srv |= b.input;
            r.uav |= !b.input;
        }
    }
    for (int s = 0; s < stageCount; ++s) {
        std::map<std::string, int> mask;
        for (const EngineBinding& b : stages_[s].bindings)
            mask[b.resource] |= b.input ? 1 : 2;
        for (const auto& entry : mask)
            if (entry.second == 3)
                resources_[index_.at(entry.first)].bothStates = true;
    }
    for (EngineResource& r : resources_) {
        r.first = -1;
        r.last = -1;
        for (int s = 0; s < stageCount; ++s)
            for (const EngineBinding& b : stages_[s].bindings)
                if (b.resource == r.name) {
                    r.first = r.first < 0 ? s : std::min(r.first, s);
                    r.last = std::max(r.last, s);
                }
    }
    for (const EngineDump& d : dumps_) {
        EngineResource& r = resources_[index_.at(d.resource)];
        if (mode_ == Mode::Validation) {
            r.dumped = true;
            r.last = std::max(r.last, stageCount);
        }
    }
    for (EngineResource& r : resources_) {
        if (!r.srv && !r.uav && !r.dumped) {
            r.lifetime = EngineResource::Lifetime::Dead;
        } else if (r.external) {
            r.lifetime = EngineResource::Lifetime::External;
        } else if (!r.init.empty()) {
            r.lifetime = EngineResource::Lifetime::Persistent;
            r.first = 0;
            if (r.last < 0) r.last = 0;
        } else if (r.committed || IsTexture(r)) {
            r.lifetime = EngineResource::Lifetime::Dedicated;
            if (r.first < 0) r.first = 0;
            if (r.last < 0) r.last = 0;
        } else {
            // A transient buffer whose first use is an SRV would read memory it
            // never wrote; keep it committed so the sentinel fill is visible.
            bool writes = false;
            for (const EngineBinding& b : stages_[r.first].bindings)
                if (b.resource == r.name && !b.input) writes = true;
            r.readsBeforeWrite = !writes;
            if (r.readsBeforeWrite || r.bothStates)
                r.lifetime = EngineResource::Lifetime::Dedicated;
            else
                r.lifetime = EngineResource::Lifetime::Transient;
        }
    }
}

// --------------------------------------------------------------------------
// Static arena plan (same best-fit interval allocation as M15-A)
// --------------------------------------------------------------------------
void PersistentEngine::PlanArena() {
    const int stageCount = static_cast<int>(stages_.size());
    std::vector<uint64_t> live(stageCount + 1, 0), transient(stageCount + 1, 0);
    for (const EngineResource& r : resources_) {
        if (r.lifetime == EngineResource::Lifetime::Dead || r.first < 0) continue;
        for (int i = r.first; i <= r.last && i <= stageCount; ++i) {
            live[i] += r.bytes;
            if (r.lifetime != EngineResource::Lifetime::Persistent &&
                r.lifetime != EngineResource::Lifetime::External)
                transient[i] += r.bytes;
        }
    }
    for (uint64_t v : live) memory_.peakLiveBytes = std::max(memory_.peakLiveBytes, v);
    for (uint64_t v : transient)
        memory_.peakTransientBytes = std::max(memory_.peakTransientBytes, v);
    if (!arenaEnabled_) {
        for (EngineResource& r : resources_) {
            if (r.lifetime == EngineResource::Lifetime::Transient)
                r.lifetime = EngineResource::Lifetime::Dedicated;
        }
        return;
    }

    std::vector<std::vector<EngineResource*>> starting(stageCount + 1), ending(stageCount + 1);
    for (EngineResource& r : resources_) {
        switch (r.lifetime) {
            case EngineResource::Lifetime::Dead:
            case EngineResource::Lifetime::External:
                break;
            case EngineResource::Lifetime::Persistent:
            case EngineResource::Lifetime::Dedicated:
                r.arenaSize = AlignPlacement(r.bytes);
                break;
            case EngineResource::Lifetime::Transient:
                r.arenaSize = AlignPlacement(r.bytes);
                starting[r.first].push_back(&r);
                ending[r.last].push_back(&r);
                break;
        }
    }
    struct Range {
        uint64_t offset = 0, size = 0;
    };
    auto addFree = [](std::vector<Range>& free, uint64_t offset, uint64_t size) {
        if (!size) return;
        free.push_back({offset, size});
        std::sort(free.begin(), free.end(),
                  [](const Range& a, const Range& b) { return a.offset < b.offset; });
        std::vector<Range> merged;
        for (const Range& r : free) {
            if (!merged.empty() && r.offset <= merged.back().offset + merged.back().size)
                merged.back().size =
                    std::max(merged.back().offset + merged.back().size, r.offset + r.size) -
                    merged.back().offset;
            else
                merged.push_back(r);
        }
        free = std::move(merged);
    };
    std::vector<Range> free;
    uint64_t high = 0;
    for (int i = 0; i <= stageCount; ++i) {
        if (i > 0)
            for (EngineResource* r : ending[i - 1]) addFree(free, r->arenaOffset, r->arenaSize);
        auto& batch = starting[i];
        std::sort(batch.begin(), batch.end(), [](const EngineResource* a,
                                                 const EngineResource* b) {
            if (a->arenaSize != b->arenaSize) return a->arenaSize > b->arenaSize;
            return a->name < b->name;
        });
        for (EngineResource* r : batch) {
            int best = -1;
            for (size_t k = 0; k < free.size(); ++k)
                if (free[k].size >= r->arenaSize &&
                    (best < 0 || free[k].size < free[best].size ||
                     (free[k].size == free[best].size && free[k].offset < free[best].offset)))
                    best = static_cast<int>(k);
            if (best >= 0) {
                r->arenaOffset = free[best].offset;
                free[best].offset += r->arenaSize;
                free[best].size -= r->arenaSize;
                if (!free[best].size) free.erase(free.begin() + best);
            } else {
                r->arenaOffset = high;
                high += r->arenaSize;
            }
        }
    }
    memory_.arenaHeapBytes = AlignPlacement(high);
}

// --------------------------------------------------------------------------
// Resources
// --------------------------------------------------------------------------
void PersistentEngine::CreateResources() {
    if (arenaEnabled_ && memory_.arenaHeapBytes) {
        D3D12_HEAP_DESC hd{};
        hd.SizeInBytes = memory_.arenaHeapBytes;
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Properties.CreationNodeMask = 1;
        hd.Properties.VisibleNodeMask = 1;
        hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        Check(device_->CreateHeap(&hd, IID_PPV_ARGS(&arenaHeap_)), "CreateHeap(arena)");
        D3D12_RESOURCE_DESC canary = BufferDesc(memory_.arenaHeapBytes);
        Check(device_->CreatePlacedResource(arenaHeap_.Get(), 0, &canary,
                                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                                            IID_PPV_ARGS(&arenaCanary_)),
              "CreatePlacedResource(canary)");
        ++memory_.placedAllocations;
    }
    for (EngineResource& r : resources_) {
        if (r.lifetime == EngineResource::Lifetime::Dead) continue;
        if (r.external) {
            memory_.externalBytes += r.bytes;
            continue;
        }
        const D3D12_RESOURCE_DESC desc =
            IsTexture(r) ? TextureDesc(r.format, r.width, r.height) : BufferDesc(r.bytes);
        if (arenaEnabled_ && r.lifetime == EngineResource::Lifetime::Transient) {
            D3D12_RESOURCE_DESC placed = BufferDesc(r.arenaSize);
            Check(device_->CreatePlacedResource(arenaHeap_.Get(), r.arenaOffset, &placed,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&r.gpu)),
                  "CreatePlacedResource");
            ++memory_.placedAllocations;
            memory_.placedBytes += r.bytes;
            continue;
        }
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&r.gpu)),
              "CreateCommittedResource");
        ++memory_.committedAllocations;
        memory_.committedBytes += r.bytes;
        if (IsTexture(r)) ++memory_.committedTextures;
        else ++memory_.committedBuffers;
    }
}

void PersistentEngine::PrepareStages() {
    for (EngineStage& s : stages_) {
        if (s.copy) {
            EngineResource& src = resources_[index_.at(s.bindings[0].resource)];
            EngineResource& dst = resources_[index_.at(s.bindings[1].resource)];
            if (!IsTexture(src) || !IsTexture(dst))
                throw std::runtime_error("copy stages require textures");
            if (src.width != dst.width || src.height != dst.height || src.bytes != dst.bytes)
                throw std::runtime_error("copy stages require matching byte extents");
            const D3D12_RESOURCE_DESC srcDesc = TextureDesc(src.format, src.width, src.height);
            const D3D12_RESOURCE_DESC dstDesc = TextureDesc(dst.format, dst.width, dst.height);
            UINT rows = 0;
            uint64_t srcBytes = 0, dstBytes = 0;
            device_->GetCopyableFootprints(&srcDesc, 0, 1, 0, &s.copyFootprint, &rows,
                                            nullptr, &srcBytes);
            device_->GetCopyableFootprints(&dstDesc, 0, 1, 0, &s.copyFootprintDst, &rows,
                                            nullptr, &dstBytes);
            s.copyBytes = std::max(srcBytes, dstBytes);
            if (srcBytes != dstBytes || s.copyFootprint.Footprint.RowPitch !=
                                            s.copyFootprintDst.Footprint.RowPitch)
                throw std::runtime_error("copy stages require matching footprints");
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC staging = BufferDesc(s.copyBytes);
            staging.Flags = D3D12_RESOURCE_FLAG_NONE;
            Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &staging,
                                                   D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&s.copyStaging)),
                  "CreateCommittedResource(copy staging)");
            ++memory_.committedAllocations;
            memory_.committedBytes += s.copyBytes;
            memory_.committedBuffers += 1;
            continue;
        }
        ResolveShader(s);
        auto code = ReadFile(s.shader);
        const uint32_t srvCount = static_cast<uint32_t>(
            std::count_if(s.bindings.begin(), s.bindings.end(),
                          [](const EngineBinding& b) { return b.input; }));
        const uint32_t uavCount =
            static_cast<uint32_t>(s.bindings.size()) - srvCount;
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
        D3D12_ROOT_SIGNATURE_DESC root{};
        root.NumParameters = 3;
        root.pParameters = params;
        ComPtr<ID3DBlob> signature, error;
        HRESULT hr = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1,
                                                  &signature, &error);
        if (FAILED(hr) && error)
            std::fprintf(stderr, "%s\n", static_cast<const char*>(error->GetBufferPointer()));
        Check(hr, "D3D12SerializeRootSignature");
        Check(device_->CreateRootSignature(0, signature->GetBufferPointer(),
                                           signature->GetBufferSize(),
                                           IID_PPV_ARGS(&s.signature)),
              "CreateRootSignature");
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
        ++memory_.descriptorHeaps;
        memory_.descriptorCount += srvCount + uavCount;

        const UINT stride = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = s.heap->GetCPUDescriptorHandleForHeapStart();
        uint32_t srv = 0, uav = 0;
        for (size_t i = 0; i < s.bindings.size(); ++i) {
            const EngineBinding& b = s.bindings[i];
            const uint32_t at = b.input ? srv++ : srvCount + uav++;
            D3D12_CPU_DESCRIPTOR_HANDLE handle{cpu.ptr + SIZE_T(at) * stride};
            EngineResource& r = resources_[index_.at(b.resource)];
            if (r.external) {
                externalSlots_.push_back(
                    {static_cast<size_t>(&s - stages_.data()), i, at});
                continue;
            }
            if (b.kind == EngineResource::Kind::Buffer) {
                CreateBufferView(b, r.gpu.Get(), handle);
            } else {
                CreateTextureView(b, r.gpu.Get(), handle);
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
        if (!s.params.empty()) std::memcpy(mapped, s.params.data(), s.params.size());
        s.paramUpload->Unmap(0, nullptr);
        memory_.paramUploadBytes += Align(s.params.size());
    }
}

void PersistentEngine::ResolveShader(EngineStage& stage) {
    fs::path p(stage.shader);
    if (fs::exists(p)) return;
    if (shaderDir_.empty()) throw std::runtime_error("shader not found: " + stage.shader);
    p = fs::path(shaderDir_) / p.filename();
    if (!fs::exists(p)) throw std::runtime_error("shader not found: " + stage.shader);
    stage.shader = p.string();
}

void PersistentEngine::CreateBufferView(const EngineBinding& binding,
                                        ID3D12Resource* resource,
                                        D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    const EngineResource& r = resources_[index_.at(binding.resource)];
    if (binding.input) {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R32_UINT;
        view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Buffer.NumElements = static_cast<UINT>(r.bytes / 4);
        device_->CreateShaderResourceView(resource, &view, handle);
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R32_UINT;
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements = static_cast<UINT>(r.bytes / 4);
        device_->CreateUnorderedAccessView(resource, nullptr, &view, handle);
    }
}

void PersistentEngine::CreateTextureView(const EngineBinding& binding,
                                         ID3D12Resource* resource,
                                         D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    const EngineResource& r = resources_[index_.at(binding.resource)];
    if (binding.input) {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = r.format;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(resource, &view, handle);
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = r.format;
        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device_->CreateUnorderedAccessView(resource, nullptr, &view, handle);
    }
}

// --------------------------------------------------------------------------
// Initial uploads
// --------------------------------------------------------------------------
void PersistentEngine::CopyBufferIn(ID3D12Resource* dest, uint64_t offset,
                                    const void* data, uint64_t bytes) {
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
    Transition(dest, D3D12_RESOURCE_STATE_COPY_DEST);
    activeList_->CopyBufferRegion(dest, offset, upload.Get(), 0, bytes);
    uploads_.push_back(upload);
    pendingUpload_ += Align(bytes);
    if (pendingUpload_ >= kUploadBatch) FlushUploads();
}

void PersistentEngine::FillSentinel(ID3D12Resource* dest, uint64_t bytes,
                                    uint64_t label) {
    if (!bytes) return;
    std::vector<uint8_t> data(static_cast<size_t>(std::min<uint64_t>(bytes, 8u << 20)));
    for (uint64_t offset = 0; offset < bytes; offset += data.size()) {
        const uint64_t n = std::min<uint64_t>(data.size(), bytes - offset);
        for (uint64_t i = 0; i < n; ++i)
            data[i] = static_cast<uint8_t>(((offset + i) * 37u + 11u + label * 97u) & 0xFFu);
        CopyBufferIn(dest, offset, data.data(), n);
    }
}

void PersistentEngine::CopyTextureIn(EngineResource& resource,
                                     const std::vector<uint8_t>& data) {
    const D3D12_RESOURCE_DESC desc = TextureDesc(resource.format, resource.width,
                                                 resource.height);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 rowBytes = 0, total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
    if (data.size() != uint64_t(resource.height) * rowBytes)
        throw std::runtime_error("texture init size mismatch: " + resource.name);
    D3D12_HEAP_PROPERTIES up{};
    up.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC ud = BufferDesc(total);
    ud.Flags = D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> upload;
    Check(device_->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &ud,
                                           D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                           IID_PPV_ARGS(&upload)),
          "CreateCommittedResource(texture upload)");
    void* mapped = nullptr;
    Check(upload->Map(0, nullptr, &mapped), "Map(texture upload)");
    auto* bytes = static_cast<uint8_t*>(mapped);
    std::memset(bytes, 0, static_cast<size_t>(total));
    for (uint32_t y = 0; y < resource.height; ++y)
        std::memcpy(bytes + footprint.Offset + uint64_t(y) * footprint.Footprint.RowPitch,
                    data.data() + uint64_t(y) * rowBytes, static_cast<size_t>(rowBytes));
    upload->Unmap(0, nullptr);
    Transition(resource.gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = upload.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION to{};
    to.pResource = resource.gpu.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.SubresourceIndex = 0;
    activeList_->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    uploads_.push_back(upload);
    pendingUpload_ += total;
    if (pendingUpload_ >= kUploadBatch) FlushUploads();
}

void PersistentEngine::DumpDebug() {
    if (!infoQueue_) return;
    const UINT64 count = infoQueue_->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        infoQueue_->GetMessage(i, nullptr, &size);
        std::vector<uint8_t> bytes(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        infoQueue_->GetMessage(i, message, &size);
        std::fprintf(stderr, "D3D12 debug: %s\n", message->pDescription);
    }
}

std::vector<std::string> PersistentEngine::debugMessages() const {
    std::vector<std::string> out;
    if (!infoQueue_) return out;
    const UINT64 count = infoQueue_->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        infoQueue_->GetMessage(i, nullptr, &size);
        std::vector<uint8_t> bytes(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        infoQueue_->GetMessage(i, message, &size);
        out.emplace_back(message->pDescription ? message->pDescription : "");
    }
    return out;
}

PersistentEngine::DebugCounts PersistentEngine::debugCounts() const {
    DebugCounts out;
    if (!infoQueue_) return out;
    out.total = infoQueue_->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < out.total; ++i) {
        SIZE_T size = 0;
        infoQueue_->GetMessage(i, nullptr, &size);
        std::vector<uint8_t> bytes(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        infoQueue_->GetMessage(i, message, &size);
        if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
            message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)
            ++out.errors;
        else if (message->Severity == D3D12_MESSAGE_SEVERITY_WARNING)
            ++out.warnings;
    }
    return out;
}

void PersistentEngine::CheckDebug(HRESULT hr, const char* what) {
    if (FAILED(hr)) DumpDebug();
    Check(hr, what);
}

void PersistentEngine::UploadInitData() {
    CheckDebug(allocator_->Reset(), "Reset(allocator:init-start)");
    CheckDebug(list_->Reset(allocator_.Get(), nullptr), "Reset(list:init-start)");
    activeList_ = list_.Get();
    if (arenaEnabled_ && arenaCanary_) {
        FillSentinel(arenaCanary_.Get(), memory_.arenaHeapBytes, seed_);
        Transition(arenaCanary_.Get(), D3D12_RESOURCE_STATE_COMMON);
    }
    for (EngineResource& r : resources_) {
        if (!r.gpu) continue;
        if (r.lifetime == EngineResource::Lifetime::Transient) continue;  // canary
        if (!r.init.empty()) {
            const std::vector<uint8_t> data = ReadFile(r.init);
            if (IsTexture(r)) {
                CopyTextureIn(r, data);
            } else {
                if (data.size() != r.bytes)
                    throw std::runtime_error("init size mismatch: " + r.name);
                CopyBufferIn(r.gpu.Get(), 0, data.data(), r.bytes);
            }
        } else if (IsTexture(r)) {
            std::vector<uint8_t> blank(static_cast<size_t>(r.bytes));
            for (size_t i = 0; i < blank.size(); ++i)
                blank[i] = static_cast<uint8_t>((i * 37u + 11u + seed_ * 97u) & 0xFFu);
            CopyTextureIn(r, blank);
        } else {
            FillSentinel(r.gpu.Get(), r.bytes, seed_);
        }
        Transition(r.gpu.Get(), D3D12_RESOURCE_STATE_COMMON);
    }
    FlushUploads();
}

void PersistentEngine::FlushUploads() {
    if (uploads_.empty()) return;
    const HRESULT hr = list_->Close();
    if (FAILED(hr)) {
        if (infoQueue_) {
            const UINT64 count = infoQueue_->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; ++i) {
                SIZE_T size = 0;
                infoQueue_->GetMessage(i, nullptr, &size);
                std::vector<uint8_t> bytes(size);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
                infoQueue_->GetMessage(i, message, &size);
                std::fprintf(stderr, "D3D12 debug: %s\n", message->pDescription);
            }
        }
    }
    Check(hr, "Close(upload batch)");
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal(upload)");
    Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion(upload)");
    WaitForSingleObject(event_, INFINITE);
    uploads_.clear();
    pendingUpload_ = 0;
    CheckDebug(allocator_->Reset(), "Reset(allocator:upload)");
    CheckDebug(list_->Reset(allocator_.Get(), nullptr), "Reset(list:upload)");
    activeList_ = list_.Get();
}

void PersistentEngine::FlushInit() {
    Check(list_->Close(), "Close(init)");
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal(init)");
    Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion(init)");
    WaitForSingleObject(event_, INFINITE);
    // Leave the list closed: ExecuteFrame resets it before recording.  Opening
    // it here would make the next allocator reset illegal.
}

// --------------------------------------------------------------------------
// External resources
// --------------------------------------------------------------------------
EngineResource& PersistentEngine::Find(const std::string& name) {
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error("unknown resource " + name);
    return resources_[it->second];
}

ID3D12Resource* PersistentEngine::ResourceFor(const std::string& name) {
    EngineResource& resource = Find(name);
    if (resource.gpu) return resource.gpu.Get();
    auto it = externals_.find(name);
    return it == externals_.end() ? nullptr : it->second.resource;
}

void PersistentEngine::ValidateExternal(const External& external,
                                        const EngineResource& resource,
                                        bool output) const {
    if (!external.resource) throw std::runtime_error("null external resource " + external.name);
    ComPtr<ID3D12Device> owner;
    if (FAILED(external.resource->GetDevice(IID_PPV_ARGS(&owner))) ||
        owner.Get() != device_.Get())
        throw std::runtime_error("external resource belongs to a different device: " +
                                 external.name);
    const D3D12_RESOURCE_DESC desc = external.resource->GetDesc();
    if (IsTexture(resource)) {
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
            throw std::runtime_error("external resource is not a texture2d: " + external.name);
        if (desc.Width != resource.width || desc.Height != resource.height)
            throw std::runtime_error("external texture dimension does not match manifest: " +
                                     external.name);
        if (desc.Format != resource.format)
            throw std::runtime_error("external texture format does not match manifest: " +
                                     external.name);
        if (desc.DepthOrArraySize != 1 || desc.MipLevels != 1 ||
            desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0)
            throw std::runtime_error("external texture must have one mip, one array slice and "
                                     "one sample: " + external.name);
        if (output && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            throw std::runtime_error("external output lacks ALLOW_UNORDERED_ACCESS: " +
                                     external.name);
    } else {
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
            throw std::runtime_error("external resource is not a buffer: " + external.name);
        if (desc.Width < resource.bytes)
            throw std::runtime_error("external buffer is shorter than the manifest extent: " +
                                     external.name);
        if (output && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            throw std::runtime_error("external output buffer lacks ALLOW_UNORDERED_ACCESS: " +
                                     external.name);
    }
}

size_t PersistentEngine::RebindExternal(const External& external) {
    EngineResource& resource = Find(external.name);
    if (!resource.external)
        throw std::runtime_error("resource is not declared external: " + external.name);

    bool anyOutput = false;
    for (const ExternalSlot& slot : externalSlots_) {
        const EngineBinding& binding = stages_[slot.stage].bindings[slot.binding];
        if (binding.resource == external.name && !binding.input) anyOutput = true;
    }
    ValidateExternal(external, resource, anyOutput);

    externals_[external.name] = external;
    const UINT stride = device_->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    size_t rebinds = 0;
    for (const ExternalSlot& slot : externalSlots_) {
        const EngineStage& stage = stages_[slot.stage];
        const EngineBinding& binding = stage.bindings[slot.binding];
        if (binding.resource != external.name) continue;
        D3D12_CPU_DESCRIPTOR_HANDLE handle{
            stage.heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(slot.descriptor) * stride};
        if (binding.kind == EngineResource::Kind::Buffer)
            CreateBufferView(binding, external.resource, handle);
        else
            CreateTextureView(binding, external.resource, handle);
        ++rebinds;
    }
    return rebinds;
}

void PersistentEngine::SetExternal(const External& external) {
    RebindExternal(external);
}

// --------------------------------------------------------------------------
// Readback targets for validation dumps
// --------------------------------------------------------------------------
void PersistentEngine::CreateReadbacks() {
    readbacks_.clear();
    dumpNames_.clear();
    if (mode_ != Mode::Validation) return;
    for (const EngineDump& d : dumps_) {
        const EngineResource& r = Find(d.resource);
        Readback rb;
        rb.texture = IsTexture(r);
        rb.width = r.width;
        rb.height = r.height;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        if (rb.texture) {
            const D3D12_RESOURCE_DESC desc = TextureDesc(r.format, r.width, r.height);
            UINT rows = 0;
            device_->GetCopyableFootprints(&desc, 0, 1, 0, &rb.footprint, &rows, nullptr,
                                           &rb.bytes);
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = rb.bytes;
            rd.Height = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&rb.resource)),
                  "CreateCommittedResource(dump texture)");
        } else {
            rb.bytes = r.bytes;
            D3D12_RESOURCE_DESC rd = BufferDesc(Align(r.bytes));
            rd.Flags = D3D12_RESOURCE_FLAG_NONE;
            Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&rb.resource)),
                  "CreateCommittedResource(dump buffer)");
        }
        readbacks_.push_back(std::move(rb));
        dumpNames_.push_back(d.resource);
    }
}

void PersistentEngine::Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES to) {
    if (!resource) throw std::runtime_error("transition of a null resource");
    auto it = states_.find(resource);
    const D3D12_RESOURCE_STATES from =
        it == states_.end() ? D3D12_RESOURCE_STATE_COMMON : it->second;
    if (from == to) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    activeList_->ResourceBarrier(1, &barrier);
    ++lastTransitions_;
    states_[resource] = to;
}

void PersistentEngine::Transition(EngineResource& resource, D3D12_RESOURCE_STATES to) {
    Transition(resource.gpu.Get(), to);
}

// --------------------------------------------------------------------------
// Frame execution
// --------------------------------------------------------------------------
PersistentEngine::FrameStats PersistentEngine::ExecuteFrame(
    const FrameRequest& request, ID3D12GraphicsCommandList* borrowed) {
    if (!initialized_) throw std::runtime_error("engine is not initialized");
    if (!borrowed && !ownsQueue_)
        throw std::runtime_error("an external-queue engine needs a caller command list");
    if (borrowed && mode_ == Mode::Validation && !request.dumpDir.empty())
        throw std::runtime_error("validation readback requires engine-owned submission");

    FrameStats stats;
    lastTransitions_ = 0;
    lastUavBarriers_ = 0;
    lastAliasingBarriers_ = 0;

    // Validate the complete external set before touching the allocator or the
    // command list.  A rejected frame therefore leaves the engine reusable and
    // a borrowed list still open exactly as the caller left it.
    struct Provided {
        bool input = false, output = false;
        const External* external = nullptr;
    };
    std::map<std::string, Provided> provided;
    for (const External& e : request.inputs) {
        EngineResource& r = Find(e.name);
        if (!r.external)
            throw std::runtime_error("resource is not external: " + e.name);
        ValidateExternal(e, r, false);
        Provided& p = provided[e.name];
        p.input = true;
        p.external = &e;
    }
    for (const External& e : request.outputs) {
        EngineResource& r = Find(e.name);
        if (!r.external)
            throw std::runtime_error("resource is not external: " + e.name);
        ValidateExternal(e, r, true);
        Provided& p = provided[e.name];
        p.output = true;
        p.external = &e;
    }
    for (const auto& entry : provided) {
        if (entry.second.input && entry.second.output)
            throw std::runtime_error("external resource bound as both input and output: " +
                                     entry.first);
    }
    for (const ExternalSlot& slot : externalSlots_) {
        const std::string& name = stages_[slot.stage].bindings[slot.binding].resource;
        auto it = provided.find(name);
        if (it == provided.end())
            throw std::runtime_error("external resource is not bound: " + name);
        const bool wantsOutput = !stages_[slot.stage].bindings[slot.binding].input;
        if (wantsOutput ? !it->second.output : !it->second.input)
            throw std::runtime_error("external resource bound in the wrong direction: " +
                                     name);
    }

    activeList_ = borrowed ? borrowed : list_.Get();
    const bool owns = !borrowed;
    if (owns) {
        CheckDebug(allocator_->Reset(), "Reset(allocator:frame)");
        CheckDebug(list_->Reset(allocator_.Get(), nullptr), "Reset(list:frame)");
    }

    auto apply = [&](const External& external) {
        auto it = externals_.find(external.name);
        if (it == externals_.end() || it->second.resource != external.resource) {
            stats.externalRebinds += RebindExternal(external);
        } else {
            it->second.state = external.state;
        }
        states_[external.resource] = external.state;
    };
    for (const External& e : request.inputs) apply(e);
    for (const External& e : request.outputs) apply(e);

    const auto f0 = std::chrono::steady_clock::now();
    if (owns && timer_) {
        timer_->Reset();
        timer_->BeginGlobal(activeList_);
    }
    for (const EngineStage& stage : stages_) {
        if (arenaEnabled_) {
            D3D12_RESOURCE_BARRIER alias{};
            alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
            alias.Aliasing.pResourceBefore = nullptr;
            alias.Aliasing.pResourceAfter = nullptr;
            activeList_->ResourceBarrier(1, &alias);
            ++lastAliasingBarriers_;
        }
        if (stage.copy) {
            EngineResource& src = resources_[index_.at(stage.bindings[0].resource)];
            EngineResource& dst = resources_[index_.at(stage.bindings[1].resource)];
            Transition(src.gpu.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(dst.gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
            Transition(stage.copyStaging.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION toBuffer{};
            toBuffer.pResource = stage.copyStaging.Get();
            toBuffer.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            toBuffer.PlacedFootprint = stage.copyFootprint;
            D3D12_TEXTURE_COPY_LOCATION fromTexture{};
            fromTexture.pResource = src.gpu.Get();
            fromTexture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            fromTexture.SubresourceIndex = 0;
            activeList_->CopyTextureRegion(&toBuffer, 0, 0, 0, &fromTexture, nullptr);
            Transition(stage.copyStaging.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION fromBuffer{};
            fromBuffer.pResource = stage.copyStaging.Get();
            fromBuffer.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            fromBuffer.PlacedFootprint = stage.copyFootprintDst;
            D3D12_TEXTURE_COPY_LOCATION toTexture{};
            toTexture.pResource = dst.gpu.Get();
            toTexture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            toTexture.SubresourceIndex = 0;
            activeList_->CopyTextureRegion(&toTexture, 0, 0, 0, &fromBuffer, nullptr);
            ++stats.copies;
            continue;
        }
        ID3D12DescriptorHeap* heaps[] = {stage.heap.Get()};
        activeList_->SetDescriptorHeaps(1, heaps);
        activeList_->SetPipelineState(stage.pipeline.Get());
        activeList_->SetComputeRootSignature(stage.signature.Get());
        activeList_->SetComputeRootConstantBufferView(
            0, stage.paramUpload->GetGPUVirtualAddress());
        for (const EngineBinding& b : stage.bindings) {
            ID3D12Resource* resource = ResourceFor(b.resource);
            if (!resource) throw std::runtime_error("unbound resource " + b.resource);
            Transition(resource, b.input ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                         : D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        const uint32_t srvCount = static_cast<uint32_t>(
            std::count_if(stage.bindings.begin(), stage.bindings.end(),
                          [](const EngineBinding& b) { return b.input; }));
        const UINT stride = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const D3D12_GPU_DESCRIPTOR_HANDLE base =
            stage.heap->GetGPUDescriptorHandleForHeapStart();
        activeList_->SetComputeRootDescriptorTable(1, base);
        D3D12_GPU_DESCRIPTOR_HANDLE uav = base;
        uav.ptr += SIZE_T(srvCount) * stride;
        activeList_->SetComputeRootDescriptorTable(2, uav);
        D3D12_RESOURCE_BARRIER uavBarrier{};
        uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarrier.UAV.pResource = nullptr;
        activeList_->ResourceBarrier(1, &uavBarrier);
        ++lastUavBarriers_;
        if (owns && timer_) timer_->Begin(activeList_);
        activeList_->Dispatch(stage.groups[0], stage.groups[1], stage.groups[2]);
        if (owns && timer_) timer_->End(activeList_, stage.name);
        ++stats.dispatches;
    }
    if (owns && timer_) timer_->EndGlobal(activeList_);
    const auto f1 = std::chrono::steady_clock::now();

    std::vector<size_t> dumpIndices;
    if (mode_ == Mode::Validation && !request.dumpDir.empty())
        RecordDumps(activeList_, request, dumpIndices);

    for (const External& e : request.outputs) {
        if (e.resource) Transition(e.resource, outputState_);
    }
    if (owns && timer_) timer_->Resolve(activeList_);

    if (owns) {
        CheckDebug(activeList_->Close(), "Close(frame)");
        ID3D12CommandList* lists[] = {activeList_};
        queue_->ExecuteCommandLists(1, lists);
        stats.engineSubmitted = true;
        Check(queue_->Signal(fence_.Get(), ++fenceValue_), "Signal(frame)");
        Check(fence_->SetEventOnCompletion(fenceValue_, event_), "SetEventOnCompletion(frame)");
        WaitForSingleObject(event_, INFINITE);
        stats.engineWaited = true;
        const auto f2 = std::chrono::steady_clock::now();
        stats.recordMs = Seconds(f0, f1) * 1000.0;
        stats.submitWaitMs = Seconds(f1, f2) * 1000.0;
        if (timer_) {
            double global = 0.0;
            std::vector<double> stageMs;
            std::vector<std::string> names;
            if (timer_->Read(global, stageMs, names)) {
                stats.gpuIntervalMs = global;
                stats.gpuIntervalValid = true;
                stats.stageMs = std::move(stageMs);
                stats.stageNames = std::move(names);
            }
        }
    } else {
        stats.recordMs = Seconds(f0, std::chrono::steady_clock::now()) * 1000.0;
    }
    stats.transitions = lastTransitions_;
    stats.uavBarriers = lastUavBarriers_;
    stats.aliasingBarriers = lastAliasingBarriers_;

    if (!dumpIndices.empty()) {
        const auto r0 = std::chrono::steady_clock::now();
        WriteDumps(request, dumpIndices);
        stats.readbackMs = Seconds(r0, std::chrono::steady_clock::now()) * 1000.0;
    }
    return stats;
}

void PersistentEngine::RecordDumps(ID3D12GraphicsCommandList* list,
                                   const FrameRequest& request,
                                   std::vector<size_t>& indices) {
    std::vector<std::string> wanted = request.dumps;
    if (wanted.empty())
        for (const EngineDump& d : dumps_) wanted.push_back(d.resource);
    for (const std::string& name : wanted) {
        size_t index = dumps_.size();
        for (size_t i = 0; i < dumps_.size(); ++i)
            if (dumps_[i].resource == name) {
                index = i;
                break;
            }
        if (index == dumps_.size())
            throw std::runtime_error("no readback target for " + name);
        if (std::find(indices.begin(), indices.end(), index) != indices.end()) continue;
        EngineResource& r = Find(name);
        ID3D12Resource* source = ResourceFor(name);
        if (!source) throw std::runtime_error("dump of an unbound resource: " + name);
        Transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Readback& rb = readbacks_[index];
        if (rb.texture) {
            D3D12_TEXTURE_COPY_LOCATION to{};
            to.pResource = rb.resource.Get();
            to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            to.PlacedFootprint = rb.footprint;
            D3D12_TEXTURE_COPY_LOCATION from{};
            from.pResource = source;
            from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            from.SubresourceIndex = 0;
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        } else {
            list->CopyBufferRegion(rb.resource.Get(), 0, source, 0, r.bytes);
        }
        indices.push_back(index);
    }
}

void PersistentEngine::WriteDumps(const FrameRequest& request,
                                  const std::vector<size_t>& indices) {
    for (size_t index : indices) {
        Readback& rb = readbacks_[index];
        const EngineResource& r = Find(dumps_[index].resource);
        const fs::path path = fs::path(request.dumpDir) / dumps_[index].file;
        void* mapped = nullptr;
        Check(rb.resource->Map(0, nullptr, &mapped), "Map(readback)");
        if (rb.texture) {
            const uint32_t pixel = std::max<uint32_t>(1, static_cast<uint32_t>(r.bytes) /
                                                            (r.width * r.height));
            std::vector<uint8_t> packed(static_cast<size_t>(r.width) * r.height * pixel);
            const auto* src = static_cast<const uint8_t*>(mapped) + rb.footprint.Offset;
            const uint64_t rowBytes = uint64_t(r.width) * pixel;
            for (uint32_t y = 0; y < r.height; ++y)
                std::memcpy(packed.data() + uint64_t(y) * rowBytes,
                            src + uint64_t(y) * rb.footprint.Footprint.RowPitch,
                            static_cast<size_t>(rowBytes));
            rb.resource->Unmap(0, nullptr);
            WriteFile(path, packed.data(), packed.size());
        } else {
            WriteFile(path, mapped, static_cast<size_t>(rb.bytes));
            rb.resource->Unmap(0, nullptr);
        }
    }
}

}  // namespace sette



