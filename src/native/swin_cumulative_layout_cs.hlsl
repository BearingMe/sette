// EXP-028: only adapters missing from the established row-major INT8 kernels.
#include "swin_layer_cs.hlsl"

[numthreads(64, 1, 1)]
void Unpack(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pLowW*pLowH) return;
    for (uint k = 0; k < pK; k += 2u)
    {
        uint a = ((k/32u)*pLowW*pLowH+tok)*32u+k%32u;
        // E4M3's floating multiply may canonicalize -0 under DXC fast math.
        // Carry the source sign in integer bits, including E4M3 code 0x80.
        uint c0 = CodeAt(B0,a), c1 = CodeAt(B0,a+1u);
        uint h0 = HalfBits(E4M3(c0&127u)) | ((c0&128u)<<8u);
        uint h1 = HalfBits(E4M3(c1&127u)) | ((c1&128u)<<8u);
        U0[(tok*pK+k)/2u] = h0 | (h1<<16u);
    }
}

[numthreads(64, 1, 1)]
void GatherMerge(uint3 id : SV_DispatchThreadID)
{
    uint row = id.x, width = pGW/2u;
    if (row >= width*(pGH/2u)) return;
    uint mx = row%width, my = row/width;
    for (uint q = 0; q < 4u; ++q)
    {
        uint x = 8u*(mx/4u)+2u*(mx%4u)+(q&1u);
        uint y = 8u*(my/4u)+2u*(my%4u)+(q>>1u);
        for (uint c = 0; c < pC; c += 2u)
            Store2(U0, row*4u*pC+q*pC+c,
                   Q8(HalfAt(B0,(y*pGW+x)*pC+c)), Q8(HalfAt(B0,(y*pGW+x)*pC+c+1u)));
    }
}

[numthreads(64, 1, 1)]
void MergeStore(uint3 id : SV_DispatchThreadID)
{
    uint row = id.x, width = pGW/2u;
    if (row >= width*(pGH/2u)) return;
    int x = int(row%width)-int(pSx/2u), y = int(row/width)-int(pSy/2u);
    if (x<0 || y<0 || x>=int(pW/2u) || y>=int(pH/2u)) return;
    for (uint c = 0; c < pNPM; c += 4u)
    {
        uint word = 0;
        for (uint j = 0; j < 4u; ++j)
            word |= Enc8(clamp(HalfAt(B0,row*pNPM+c+j),-6.2831855f,6.2831855f))<<(8u*j);
        uint at = (((c/32u)*(pH/2u)+uint(y))*(pW/2u)+uint(x))*8u+(c/4u)%8u;
        U0[at] = word;
    }
}
