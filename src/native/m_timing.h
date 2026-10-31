// Optional D3D12 GPU timestamp capture (M12).  Included only by the `*_timed`
// executables (`-DM_TIMING`); the accepted hosts compile without it, so their
// binaries and measured outputs stay byte-identical.
#pragma once
#ifdef M_TIMING

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace timing {

using Microsoft::WRL::ComPtr;

class GpuTimer {
public:
    GpuTimer(ID3D12Device* device, ID3D12CommandQueue* queue, size_t capacity)
        : capacity_(capacity) {
        D3D12_QUERY_HEAP_DESC heap{};
        heap.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heap.Count = static_cast<UINT>(capacity_ * 2);
        if (FAILED(device->CreateQueryHeap(&heap, IID_PPV_ARGS(&heap_))))
            throw std::runtime_error("CreateQueryHeap(TIMESTAMP)");

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
        if (FAILED(device->CreateCommittedResource(
                &props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(&readback_))))
            throw std::runtime_error("CreateCommittedResource(timing readback)");
        if (FAILED(queue->GetTimestampFrequency(&frequency_)))
            throw std::runtime_error("GetTimestampFrequency");
    }

    // One query pair is reserved for the whole-command-list interval.
    bool enabled() const { return count_ + 1 < capacity_; }

    void BeginGlobal(ID3D12GraphicsCommandList* list) {
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(capacity_ * 2 - 2));
    }

    void EndGlobal(ID3D12GraphicsCommandList* list) {
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(capacity_ * 2 - 1));
    }

    void Begin(ID3D12GraphicsCommandList* list) {
        if (!enabled()) return;
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(count_ * 2));
    }

    void End(ID3D12GraphicsCommandList* list, const std::string& label) {
        if (!enabled()) return;
        list->EndQuery(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                       static_cast<UINT>(count_ * 2 + 1));
        labels_.push_back(label);
        ++count_;
    }

    void Resolve(ID3D12GraphicsCommandList* list) {
        if (count_ == 0) return;
        list->ResolveQueryData(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                               static_cast<UINT>(count_ * 2), readback_.Get(), 0);
        list->ResolveQueryData(heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                               static_cast<UINT>(capacity_ * 2 - 2), 2,
                               readback_.Get(),
                               static_cast<UINT64>(capacity_ * 2 - 2) * sizeof(uint64_t));
    }

    void Write(const std::string& path) {
        if (count_ == 0) return;
        void* mapped = nullptr;
        if (FAILED(readback_->Map(0, nullptr, &mapped)))
            throw std::runtime_error("Map(timing readback)");
        auto* stamps = static_cast<const uint64_t*>(mapped);
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cannot write " + path);
        out << "{\n  \"frequency\": " << frequency_ << ",\n  \"dispatches\": [\n";
        for (size_t i = 0; i < count_; ++i) {
            const uint64_t begin = stamps[i * 2], end = stamps[i * 2 + 1];
            const double ticks = static_cast<double>(end - begin);
            out << "    {\"index\": " << i << ", \"label\": \"" << labels_[i]
                << "\", \"begin\": " << begin << ", \"end\": " << end
                << ", \"ticks\": " << (end >= begin ? end - begin : 0)
                << ", \"ms\": " << (end >= begin && frequency_
                                        ? ticks * 1000.0 / double(frequency_) : 0.0)
                << "}";
            out << (i + 1 == count_ ? "\n" : ",\n");
        }
        const uint64_t gb = stamps[capacity_ * 2 - 2];
        const uint64_t ge = stamps[capacity_ * 2 - 1];
        out << "  ],\n  \"global\": {\"begin\": " << gb << ", \"end\": " << ge
            << ", \"ticks\": " << (ge >= gb ? ge - gb : 0)
            << ", \"ms\": " << (ge >= gb && frequency_
                                    ? double(ge - gb) * 1000.0 / double(frequency_) : 0.0)
            << "}\n}\n";
        readback_->Unmap(0, nullptr);
    }

private:
    size_t capacity_ = 0;
    size_t count_ = 0;
    uint64_t frequency_ = 0;
    ComPtr<ID3D12QueryHeap> heap_;
    ComPtr<ID3D12Resource> readback_;
    std::vector<std::string> labels_;
};

}  // namespace timing

#endif  // M_TIMING
