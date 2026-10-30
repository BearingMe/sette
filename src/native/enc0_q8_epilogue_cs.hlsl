Buffer<uint> Embedding : register(t2);
static uint CurrentToken;
float Enc0Column(uint n, float unused[128]);
#define ENC0_LOAD_COLUMN Enc0Column
#define main FieldMain
#include "enc0_cs.hlsl"
#undef main
float Enc0Column(uint n, float unused[128])
{
    uint i=CurrentToken*64u+n;
    return f16tof32((Embedding[i/2u]>>(16u*(i&1u)))&65535u);
}
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    CurrentToken=id.x;
    FieldMain(id);
}
