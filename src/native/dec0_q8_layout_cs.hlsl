// EXP-024: isolated physical-layout probes. The FP16 entry calls the existing
// Column verbatim; the encoder reads the same plane-major E4M3 resource.
#include "dec0_cs.hlsl"
#include "swin_q8_common.hlsli"

RWBuffer<uint> Encoded : register(u4);
RWBuffer<uint> RowScale : register(u5);

[numthreads(64, 1, 1)]
void Quantize(uint3 id : SV_DispatchThreadID)
{
    uint row = id.x;
    if (row >= 130560u) return;
    uint ty = row / 480u, tx = row % 480u;
    float mx = 0.0f;
    for (uint c = 0; c < 64u; ++c)
        mx = max(mx, abs(TokenValue(ty, tx, c)));
    if (mx == 0.0f) mx = 1.0f;
    float candidate = 127.0f / mx;
    double residual = 127.0 - (double)candidate * (double)mx;
    float inv = (float)((double)candidate + residual / (double)mx);
    RowScale[row] = asuint(mx * 0.0078740157f);
    for (uint g = 0; g < 16u; ++g)
    {
        uint word = 0;
        for (uint j = 0; j < 4u; ++j)
            word |= uint(QuantCode(TokenValue(ty, tx, g*4u+j) * inv) & 255) << (8u*j);
        Encoded[row*16u+g] = word;
    }
}

[numthreads(64, 1, 1)]
void FP16(uint3 id : SV_DispatchThreadID)
{
    // 2040 x 40 groups, rather than exceeding D3D12's 65535 groups/axis.
    uint pair = id.x + id.y * 130560u;
    if (pair >= 130560u * 40u) return;
    uint row = pair / 40u, n = (pair % 40u) * 2u;
    uint physical = (n / 20u) * 32u + n % 20u;
    float a = Column(row / 480u, row % 480u, physical);
    float b = Column(row / 480u, row % 480u, physical + 1u);
    Encoded[pair] = ToHalfBits(a) | (ToHalfBits(b) << 16);
}
