// M13: PV with asymmetric probability decoding.
//
// P (softmax probabilities) is non-negative, so an asymmetric per-row encoding
// uses the whole signed byte range instead of half of it.  With
//
//     P = pcode * pscale + poffset,        V = vcode * vscale
//
// the product becomes
//
//     y = (sum(pcode*vcode) * pscale + poffset * sum(vcode)) * vscale
//
// `sum(vcode)` is accumulated with a second dot4 against a word of ones, so the
// integer work is one extra dot4 per four values and the epilogue is one
// multiply-add.  The V operand and its symmetric column scales are unchanged.

#include "swin_layer_cs.hlsl"

RWBuffer<uint> U2 : register(u2);
RWBuffer<uint> U3 : register(u3);
Buffer<uint> POff : register(t4);

uint Instances() { return (pGW/4u)*(pGH/4u)*pHeads; }
uint WindowToken(uint instance, uint row)
{
    uint w=instance/pHeads;
    return ((w/(pGW/4u))*4u+row/4u)*pGW+(w%(pGW/4u))*4u+row%4u;
}

[numthreads(64,1,1)]
void PV(uint3 id : SV_DispatchThreadID)
{
    uint row=id.x%16u, instance=id.x/16u;
    if (instance>=Instances()) return;
    precise float ps = asfloat(B2[instance*16u+row]);
    precise float po = asfloat(POff[instance*16u+row]);
    for (uint c=0u;c<32u;c+=2u)
    {
        float value[2];
        for (uint j=0u;j<2u;++j)
        {
            int acc=0;
            int vsum=0;
            for (uint k=0u;k<4u;++k)
            {
                uint vw=B1[(instance*32u+c+j)*4u+k];
                acc=dot4add_i8packed(B0[(instance*16u+row)*4u+k],vw,acc);
                vsum=dot4add_i8packed(vw,0x01010101u,vsum);
            }
            precise float vs = asfloat(B3[instance*32u+c+j]);
            precise float corr = po * float(vsum);
            precise float y = (float(acc)*ps + corr) * vs;
            uint at=instance*512u+row*32u+c+j;
            U1[at]=asuint(acc);U2[at]=asuint(y);value[j]=R16(y);
        }
        Store2(U3,instance*512u+row*32u+c,value[0],value[1]);
        uint at=WindowToken(instance,row)*pHeads*32u+(instance%pHeads)*32u+c;
        Store2(U0,at,Q8(value[0]),Q8(value[1]));
    }
}
