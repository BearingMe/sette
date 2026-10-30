// EXP-029: fixed 16-token, 32-channel dynamic attention. Both operands runtime.
#include "swin_layer_cs.hlsl"
RWBuffer<uint> U2 : register(u2);
RWBuffer<uint> U3 : register(u3);

uint Instances() { return (pGW/4u)*(pGH/4u)*pHeads; }
uint WindowToken(uint instance, uint row)
{
    uint w=instance/pHeads;
    return ((w/(pGW/4u))*4u+row/4u)*pGW+(w%(pGW/4u))*4u+row%4u;
}

[numthreads(64,1,1)]
void Pack(uint3 id : SV_DispatchThreadID)
{
    uint instance=id.x;
    if (instance>=Instances()) return;
    uint head=instance%pHeads;
    for (uint row=0u;row<16u;++row)
        for (uint k=0u;k<32u;k+=2u)
        {
            uint at=WindowToken(instance,row)*pHeads*32u+head*32u+k;
            Store2(U0,instance*512u+row*32u+k,Q8(HalfAt(B0,at)),Q8(HalfAt(B0,at+1u)));
        }
    // Dynamic V columns are rows of the transpose, each with its own scale.
    for (uint c=0u;c<32u;++c)
        for (uint row=0u;row<16u;row+=2u)
            Store2(U1,instance*512u+c*16u+row,
                HalfAt(B1,WindowToken(instance,row)*pHeads*32u+head*32u+c),
                HalfAt(B1,WindowToken(instance,row+1u)*pHeads*32u+head*32u+c));
}

[numthreads(64,1,1)]
void QK(uint3 id : SV_DispatchThreadID)
{
    uint row=id.x%16u, instance=id.x/16u;
    if (instance>=Instances()) return;
    for (uint col=0u;col<16u;col+=2u)
    {
        uint halves=0u;
        for (uint j=0u;j<2u;++j)
        {
            int acc=0;
            for (uint k=0u;k<8u;++k)
                acc=dot4add_i8packed(B0[(instance*16u+row)*8u+k],B0[(instance*16u+col+j)*8u+k],acc);
            precise float y=float(acc)*asfloat(B1[instance*16u+row])*asfloat(B1[instance*16u+col+j]);
            y=y+asfloat(B2[pOffBias+(instance%pHeads)*256u+row*16u+col+j]);
            uint at=instance*256u+row*16u+col+j;
            U1[at]=asuint(acc);U2[at]=asuint(y);halves|=HalfBits(y)<<(16u*j);
        }
        U0[(instance*256u+row*16u+col)/2u]=halves;
    }
}

[numthreads(64,1,1)]
void Softmax(uint3 id : SV_DispatchThreadID)
{
    uint row=id.x;
    if (row>=Instances()*16u) return;
    float cs=CScale(), cl=R16(0.55615234375f), p1=R16(0.92730712890625f), p0=R16(1.375f);
    float poly[16];
    for (uint c=0u;c<16u;++c)
    {
        float tv=Mul16(HalfAt(B0,row*16u+c),cs);
        tv=tv < -cl ? -cl : (tv > cl ? cl : tv);
        float e1=Fma16(tv,-tv,p1);
        poly[c]=Fma16(tv,e1,p0);
    }
    float w[16];
    for (uint j=0u;j<8u;++j)
    {
        uint lo=HalfBits(poly[2u*j]),hi=HalfBits(poly[2u*j+1u]);
        uint word=((lo|(hi<<16))<<5)+0x7FF88000u;
        w[2u*j]=F16ToF32(word&0xFFFFu);w[2u*j+1u]=F16ToF32(word>>16);
    }
    float s8[16],s2[16],s4[16],s1[16];
    for (uint c=0u;c<16u;++c) s8[c]=Add16(w[c],w[c^8u]);
    for (uint c=0u;c<16u;++c) s2[c]=Add16(s8[c],s8[c^2u]);
    for (uint c=0u;c<16u;++c) s4[c]=Add16(s2[c],s2[c^4u]);
    for (uint c=0u;c<16u;++c) s1[c]=Add16(s4[c],s4[c^1u]);
    float rinv=R16(1.0f/s1[0]);
    for (uint c=0u;c<16u;c+=2u)
        Store2(U0,row*16u+c,Mul16(w[c],rinv),Mul16(w[c+1u],rinv));
}

[numthreads(64,1,1)]
void PV(uint3 id : SV_DispatchThreadID)
{
    uint row=id.x%16u, instance=id.x/16u;
    if (instance>=Instances()) return;
    for (uint c=0u;c<32u;c+=2u)
    {
        float value[2];
        for (uint j=0u;j<2u;++j)
        {
            int acc=0;
            for (uint k=0u;k<4u;++k)
                acc=dot4add_i8packed(B0[(instance*16u+row)*4u+k],B1[(instance*32u+c+j)*4u+k],acc);
            precise float y=float(acc)*asfloat(B2[instance*16u+row])*asfloat(B3[instance*32u+c+j]);
            uint at=instance*512u+row*32u+c+j;
            U1[at]=asuint(acc);U2[at]=asuint(y);value[j]=R16(y);
        }
        Store2(U3,instance*512u+row*32u+c,value[0],value[1]);
        uint at=WindowToken(instance,row)*pHeads*32u+(instance%pHeads)*32u+c;
        Store2(U0,at,Q8(value[0]),Q8(value[1]));
    }
}
