// Diagnostic only: measure the driver's reciprocal instructions for every
// binary16 argument, widened exactly to f32. Not part of the network graph.
Buffer<uint> Inputs : register(t0);
RWBuffer<uint> Outputs : register(u0);
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    float x = f16tof32(Inputs[i]);
    Outputs[2 * i] = asuint(rsqrt(x));
    Outputs[2 * i + 1] = asuint(rcp(x));
}
