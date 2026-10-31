// M15-C: persistent D3D12 inference engine.
//
// The fixture runners (`m_image_stage`, `m_network_stage`) execute one
// manifest per process and rebuild every device object on each run.  This
// engine separates that into a lifecycle:
//
//   Initialize(device, config)   parse, upload weights, build pipelines,
//                                descriptors and the transient arena once
//   ExecuteFrame(frame)          record one frame over persistent resources
//   Destroy()                    release everything
//
// A frame never recompiles a shader, recreates a pipeline, reloads a weight or
// rebuilds the arena.  The reset-frame contract has four caller-owned image
// inputs and one caller-owned output:
//
//   enc0_in_colour  RGBA16F 960x540    encoder render colour
//   post_colour     RGBA16F 960x540    post render colour
//   post_motion     RG16F   960x540    post motion (bound, unread on reset)
//   down_colour     RGBA16F 960x540    downsample colour (alpha only)
//   frame_out       RGBA16F 1920x1080  output, ALLOW_UNORDERED_ACCESS
//
// Weights and the remaining static reconstruction surfaces are immutable model
// resources loaded once.  The engine is deliberately not a general inference
// framework: it executes one composed pipeline manifest produced from the
// accepted fixtures, with the same buffers, formats, dispatch geometry and
// arithmetic as the accepted graph.  See
// docs/experiments/EXP-045-persistent-inference-engine.md and
// docs/experiments/EXP-046-backend-readiness.md.
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sette {

// One D3D12 resource in the composed pipeline.
struct EngineResource {
    enum class Kind { Buffer, Texture };
    enum class Lifetime { Persistent, Transient, Dedicated, External, Dead };

    std::string name;
    Kind kind = Kind::Buffer;
    DXGI_FORMAT format = DXGI_FORMAT_R32_UINT;
    uint64_t bytes = 0;
    uint32_t width = 0, height = 0;
    std::string init;            // empty when the resource has no file payload
    bool external = false;       // owned and supplied by the caller
    bool committed = false;      // never placed in the transient arena
    bool srv = false, uav = false, dumped = false;
    bool bothStates = false, readsBeforeWrite = false;
    int first = -1, last = -1;
    Lifetime lifetime = Lifetime::Transient;
    uint64_t arenaOffset = 0, arenaSize = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> gpu;
};

struct EngineBinding {
    bool input = true;
    EngineResource::Kind kind = EngineResource::Kind::Buffer;
    uint32_t reg = 0;
    std::string resource;
};

struct EngineStage {
    std::string name;
    std::string shader;
    uint32_t threads[3] = {1, 1, 1};
    uint32_t groups[3] = {1, 1, 1};
    std::vector<uint8_t> params;
    std::vector<EngineBinding> bindings;
    // A format-reinterpreting texture handoff (D3D12 cannot alias R32_UINT to
    // R8G8B8A8_UNORM).  The engine copies the raw bytes through one staging
    // buffer so the consumer sees exactly the producer's bytes.
    bool copy = false;
    uint64_t copyBytes = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT copyFootprint{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT copyFootprintDst{};

    Microsoft::WRL::ComPtr<ID3D12RootSignature> signature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
    Microsoft::WRL::ComPtr<ID3D12Resource> paramUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> copyStaging;
};

struct EngineDump {
    std::string resource;
    std::string file;
};

class PersistentEngine {
public:
    enum class Mode { Production, Validation };

    struct Config {
        std::string manifest;        // composed pipeline manifest (version 3)
        std::string shaderDir;       // resolved for shader file names
        bool arena = true;           // place transient buffers in one heap
        Mode mode = Mode::Production;
        bool debug = false;          // enable the D3D12 debug layer
        uint32_t seed = 0;           // sentinel fill for uninitialised memory
    };

    struct External {
        std::string name;
        ID3D12Resource* resource = nullptr;
        // State the caller guarantees before this frame's work.  The engine
        // transitions from it, and its own tracking is reset to it each frame.
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    };

    struct FrameRequest {
        std::vector<External> inputs;
        std::vector<External> outputs;
        std::string dumpDir;         // validation only; empty = no dumps
        std::vector<std::string> dumps;  // resource names; empty = manifest dumps
    };

    struct FrameStats {
        double recordMs = 0.0;
        double submitWaitMs = 0.0;
        double readbackMs = 0.0;
        uint64_t dispatches = 0;
        uint64_t copies = 0;
        uint64_t transitions = 0;
        uint64_t uavBarriers = 0;
        uint64_t aliasingBarriers = 0;
        // Descriptor views rewritten because a caller resource changed.
        uint64_t externalRebinds = 0;
        // False for a borrowed list: the engine records but never submits or
        // waits.  True when the engine owns the queue and executed the frame.
        bool engineSubmitted = false;
        bool engineWaited = false;
        double gpuIntervalMs = 0.0;
        bool gpuIntervalValid = false;
        std::vector<double> stageMs;
        std::vector<std::string> stageNames;
    };

    struct MemoryStats {
        uint64_t committedBuffers = 0, committedTextures = 0;
        uint64_t committedBytes = 0, placedBytes = 0, externalBytes = 0;
        uint64_t arenaHeapBytes = 0;
        uint64_t committedAllocations = 0, placedAllocations = 0;
        uint64_t descriptorHeaps = 0, descriptorCount = 0;
        uint64_t paramUploadBytes = 0;
        uint64_t peakLiveBytes = 0, peakTransientBytes = 0;
    };

    struct InitStats {
        double parseMs = 0.0;
        double resourceMs = 0.0;
        double pipelineMs = 0.0;
        double uploadWaitMs = 0.0;
        double totalMs = 0.0;
        size_t stages = 0, resources = 0;
        std::string deviceName;
    };

    PersistentEngine();
    ~PersistentEngine();
    PersistentEngine(const PersistentEngine&) = delete;
    PersistentEngine& operator=(const PersistentEngine&) = delete;

    // Create the device and every persistent object.  Call once.
    void Initialize(const Config& config);
    // Variant for a caller that already owns a device and compute queue.  The
    // engine then records onto the caller's command list only; it does not own
    // submission or synchronization for ExecuteFrame.
    void Initialize(ID3D12Device* device, ID3D12CommandQueue* queue,
                    const Config& config);

    // Rebind one external resource.  Cheap: rewrites that resource's SRV/UAV
    // descriptors.  The resource must match the manifest format and size.
    void SetExternal(const External& external);

    // Record and (when the engine owns submission) execute one frame.
    // `list` may be a caller-owned, open compute command list.  The engine
    // never closes, executes or waits on a borrowed list.
    FrameStats ExecuteFrame(const FrameRequest& request,
                            ID3D12GraphicsCommandList* list = nullptr);

    // State the engine leaves an external output in after the frame.
    void SetOutputState(D3D12_RESOURCE_STATES state) { outputState_ = state; }

    ID3D12Device* device() const { return device_.Get(); }
    ID3D12CommandQueue* queue() const { return queue_.Get(); }
    IDXGIAdapter1* adapter() const { return adapter_.Get(); }
    bool ownsSubmission() const { return ownsQueue_; }
    bool initialized() const { return initialized_; }

    const InitStats& initStats() const { return initStats_; }
    const MemoryStats& memory() const { return memory_; }
    const std::vector<EngineResource>& resources() const { return resources_; }
    const std::vector<EngineStage>& stages() const { return stages_; }

    struct DebugCounts {
        uint64_t total = 0, errors = 0, warnings = 0;
    };
    // Zero unless the debug layer was enabled; errors include corruption.
    DebugCounts debugCounts() const;
    // Stored debug-layer descriptions, in queue order.
    std::vector<std::string> debugMessages() const;

    // Total barriers raised by the last ExecuteFrame, for reporting.
    uint64_t lastTransitions() const { return lastTransitions_; }

    void Destroy();

private:
    struct ExternalSlot {
        size_t stage = 0;
        size_t binding = 0;
        uint32_t descriptor = 0;
    };
    struct Readback {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        uint64_t bytes = 0;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        uint32_t width = 0, height = 0;
        bool texture = false;
    };
    class FrameTimer;

    void CreateDevice();
    const std::string& configManifestPath() const { return manifest_; }
    void DumpDebug();
    void CheckDebug(HRESULT hr, const char* what);
    void Parse();
    void Analyze();
    void PlanArena();
    void CreateResources();
    void PrepareStages();
    void CreateReadbacks();
    void ResolveShader(EngineStage& stage);
    void UploadInitData();
    void FlushInit();

    EngineResource& Find(const std::string& name);
    ID3D12Resource* ResourceFor(const std::string& name);

    void Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES to);
    void Transition(EngineResource& resource, D3D12_RESOURCE_STATES to);
    void CreateViewsForExternal(const std::string& name, ID3D12Resource* resource);
    void CreateStageViews(EngineStage& stage);
    void CreateBufferView(const EngineBinding& binding, ID3D12Resource* resource,
                          D3D12_CPU_DESCRIPTOR_HANDLE handle);
    void CreateTextureView(const EngineBinding& binding, ID3D12Resource* resource,
                           D3D12_CPU_DESCRIPTOR_HANDLE handle);
    void ValidateExternal(const External& external, const EngineResource& resource,
                          bool output) const;
    size_t RebindExternal(const External& external);

    void CopyBufferIn(ID3D12Resource* dest, uint64_t offset, const void* data,
                      uint64_t bytes);
    void CopyTextureIn(EngineResource& resource, const std::vector<uint8_t>& data);
    void FillSentinel(ID3D12Resource* dest, uint64_t bytes, uint64_t label);
    void FlushUploads();

    void RecordDumps(ID3D12GraphicsCommandList* list, const FrameRequest& request,
                     std::vector<size_t>& indices);
    void WriteDumps(const FrameRequest& request, const std::vector<size_t>& indices);

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue_;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    uint64_t fenceValue_ = 0;
    bool ownsQueue_ = true;
    bool ownsDevice_ = true;
    bool debug_ = false;
    D3D12_RESOURCE_STATES outputState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    Mode mode_ = Mode::Production;
    uint32_t seed_ = 0;
    std::string manifest_;
    std::string shaderDir_;
    std::vector<EngineResource> resources_;
    std::vector<EngineStage> stages_;
    std::vector<EngineDump> dumps_;
    std::map<std::string, size_t> index_;
    std::vector<ExternalSlot> externalSlots_;

    Microsoft::WRL::ComPtr<ID3D12Heap> arenaHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> arenaCanary_;
    bool arenaEnabled_ = false;

    std::map<ID3D12Resource*, D3D12_RESOURCE_STATES> states_;
    std::map<std::string, External> externals_;
    ID3D12GraphicsCommandList* activeList_ = nullptr;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> uploads_;
    uint64_t pendingUpload_ = 0;

    std::vector<Readback> readbacks_;
    std::vector<int> readbackFor_;   // dump index -> readback slot
    std::vector<std::string> dumpNames_;

    std::unique_ptr<FrameTimer> timer_;

    InitStats initStats_;
    MemoryStats memory_;
    uint64_t lastTransitions_ = 0;
    uint64_t lastUavBarriers_ = 0;
    uint64_t lastAliasingBarriers_ = 0;
    bool initialized_ = false;
};

}  // namespace sette
