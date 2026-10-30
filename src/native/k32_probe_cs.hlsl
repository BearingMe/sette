// Arithmetic diagnostic only; reuse production conversion helpers unchanged.
#include "swin_layer_cs.hlsl"

[numthreads(32, 1, 1)]
void K32Probe(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint test = group.x;
    if (lane != 0) return;
    double first = 0.0, second = 0.0;
    for (uint i = 0; i < 16; ++i)
    {
        uint k = (i & 7u) < 4u ? 4u * (i >> 3u) + (i & 7u)
                                      : 16u + 4u * (i >> 3u) + (i & 7u) - 4u;
        first += (double)F16ToF32(B0[test * 65u + k]) *
                 (double)F16ToF32(B0[test * 65u + 32u + k]);
        second += (double)F16ToF32(B0[test * 65u + k + 8u]) *
                  (double)F16ToF32(B0[test * 65u + 32u + k + 8u]);
    }
    double c = (double)F16ToF32(B0[test * 65u + 64u]);
    float result = R16D(first + second + c);
    // The partials are exact mathematical diagnostics, not extra half roundings.
    U0[4u * test] = asuint((float)(first + c));
    U0[4u * test + 1u] = asuint((float)(first + second + c));
    U0[4u * test + 2u] = HalfBits(result);
    U0[4u * test + 3u] = HalfBits(R16D(first + second + (double)result));
}
