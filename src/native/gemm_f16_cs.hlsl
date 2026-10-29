// C18: minimal native FP16 matrix stage.
//
// Computes Y[M][N] = f16-step(A[M][K] * B[K][N] + bias) with the accumulator
// contract M04/M06 established for Preset M: the products of one 32-element
// step are summed in float32, then the accumulator takes one float16 rounding,
// and the next step starts from that rounded value.  That is d4r's
// `mma_chain` (model_enc3.py) expressed as a kernel.
//
// The accumulator is a native `float16_t`, not `f16tof32(f32tof16(x))`: with
// 16-bit types disabled DXC lowers that round trip to a truncating conversion,
// which biases every step toward zero (EXP-010).  Native f16 conversion rounds
// to nearest even, which is what the tensor-core path and d4r's packHalf2x16
// do.
//
// Operands arrive packed as f16 pairs in uint words so the runner can bind them
// as plain typed buffers: X holds A row-major (M x K, K even), W holds B
// row-major (K x N), Bias holds N halves.  Nothing is quantized here; this is
// the FP16 reference stage the INT8 kernels will be compared against.

RWByteAddressBuffer Y : register(u0);
Buffer<uint> X : register(t0);
Buffer<uint> W : register(t1);
Buffer<uint> Bias : register(t2);

cbuffer Params : register(b0)
{
    uint M;
    uint K;
    uint N;
    uint useBias;
};

float HalfAt(Buffer<uint> buf, uint index)
{
    uint word = buf[index >> 1];
    return f16tof32((index & 1u) ? (word >> 16) : (word & 0xFFFFu));
}

// Explicit round-to-nearest-even float32 -> float16, in integer arithmetic.
// The driver's own f32->f16 conversion is not trusted here: EXP-010 measured it
// truncating toward zero, which biases every accumulated step.
uint ToHalfBits(float value)
{
    uint u = asuint(value);
    uint sign = (u >> 16) & 0x8000u;
    int expo = int((u >> 23) & 0xFFu) - 127 + 15;
    uint mant = u & 0x7FFFFFu;
    if (expo <= 0)
    {
        // subnormal or zero: shift the implicit bit in and round once
        int shift = 14 - expo;
        if (shift > 24)
            return sign;
        uint full = (mant | 0x800000u) >> shift;
        uint rest = (mant | 0x800000u) & ((1u << shift) - 1u);
        uint half = 1u << (shift - 1);
        if (rest > half || (rest == half && (full & 1u)))
            full += 1u;
        return sign | full;
    }
    if (expo >= 31)
        return sign | 0x7C00u;
    uint h = (uint(expo) << 10) | (mant >> 13);
    uint rest = mant & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u)))
        h += 1u;
    return sign | h;
}

float RoundHalf(float value)
{
    return f16tof32(ToHalfBits(value));
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= M * N)
        return;
    uint row = index / N;
    uint col = index % N;

    float acc = useBias ? HalfAt(Bias, col) : 0.0;

    // one float16 rounding per 32 products, as the reference chain does
    for (uint k0 = 0; k0 < K; k0 += 32)
    {
        float step = 0.0;
        for (uint k = 0; k < 32; ++k)
        {
            uint kk = k0 + k;
            float a = HalfAt(X, row * K + kk);
            float b = HalfAt(W, kk * N + col);
            step += a * b;
        }
        acc = RoundHalf(step + acc);
    }
    Y.Store(index * 4u, asuint(acc));
}
