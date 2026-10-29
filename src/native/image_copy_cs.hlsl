// M05 B4.1: the identity stage.
//
// One image or buffer in, the same bytes out.  It exists to prove that a
// captured resource survives the native path untouched - D3D12 texture upload
// through a copyable footprint, a dispatch, and a readback copy - so that any
// difference a real stage shows is arithmetic and not transport.
//
// The typed view is deliberate: a format is bound once, on the descriptor, and
// the shader never reinterprets bytes.  For the UNORM/SNORM surfaces the same
// byte encodes the value under either interpretation, so an UNORM view round
// trips bit-exactly.

Texture2D<float4> SrcTex : register(t0);
Buffer<uint> SrcBuf : register(t0);
RWTexture2D<float4> DstTex : register(u0);
RWBuffer<uint> DstBuf : register(u0);

[numthreads(16, 16, 1)]
void CopyTex(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    DstTex.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;
    DstTex[id.xy] = SrcTex[id.xy];
}

[numthreads(64, 1, 1)]
void CopyBuf(uint3 id : SV_DispatchThreadID)
{
    uint count;
    DstBuf.GetDimensions(count);
    if (id.x < count)
        DstBuf[id.x] = SrcBuf[id.x];
}
