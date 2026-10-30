// EXP-029: bounded reset-frame texture adapter and original field epilogue.
#define main OriginalMain
#include "enc0_cs.hlsl"
#undef main

[numthreads(64,1,1)]
void Features(uint3 id : SV_DispatchThreadID)
{
    uint token=id.x, width=uint(padded.x)/2u;
    if (token>=width*uint(padded.y)/2u) return;
    uint tx=token%width, ty=token/width;
    for (uint sub=0u;sub<4u;++sub)
    {
        uint2 pixel=uint2(2u*tx+(sub&1u),2u*ty+(sub>>1u));
        float4 texel=Texel(pixel);
        uint n=DitherBits(pixel);
        for (uint slot=0u;slot<32u;slot+=2u)
        {
            uint pair=0u;
            for (uint j=0u;j<2u;++j)
            {
                uint ch=kSlotChannel[slot+j];
                float v=ch==3u ? 0.0f : E4M3Value(E4M3Code(Dither(texel[ch],n)));
                pair |= f32tof16(v)<<(16u*j); // E4M3 is exactly representable in f16
            }
            Tokens[(token*128u+sub*32u+slot)/2u]=pair;
        }
    }
}
