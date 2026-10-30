// M09: native INT8 matrix stage.
//
//   Y[m][n] = ( sum_k A[m][k] * W[n][k] ) * AScale[m] * WScale[n] + Bias[n]
//
// A is the activation encoded as signed INT8, row-major (M x K), four codes per
// uint.  W is the weight encoded as signed INT8, K-major per output column
// (N x K) so a four-code group runs along the reduction axis for both operands.
// The reduction is int32 and takes *no* intermediate rescale: the weight scale
// is per output column, so one integer sum is the whole product
// (|sum| <= K * 127 * 127; the accumulator bound is K <= 131071).
//
// The product uses dot4add_i8packed, which RDNA 3 executes as v_dot4_i32_i8:
// four signed 8-bit products and a 32-bit accumulate in one instruction.  There
// is no FP16 conversion and no rounding between the encodes and the rescale,
// so the only numerical difference from a CPU reference is the f32 multiply
// order at the epilogue.
//
// Both operands are quantised symmetrically (no zero point), which is the
// single-accumulation baseline M07 measured (per-output-channel weights) and
// the per-token activation scale M07/M13 use.
//
// Compiled at cs_6_4: dot4add_i8packed needs SM 6.4.

RWByteAddressBuffer Y : register(u0);
Buffer<uint> A : register(t0);
Buffer<uint> W : register(t1);
Buffer<uint> AScale : register(t2);
Buffer<uint> WScale : register(t3);
Buffer<uint> Bias : register(t4);

cbuffer Params : register(b0)
{
    uint M;
    uint K;
    uint N;
    uint UseBias;
};

int ByteAt(Buffer<uint> buf, uint index)
{
    uint word = buf[index >> 2];
    uint shift = (index & 3u) * 8u;
    return int(word << (24u - shift)) >> 24;   // sign-extend one byte
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= M * N)
        return;
    uint row = index / N;
    uint col = index % N;

    uint base = (row * K) >> 2;      // both offsets are four-code aligned
    uint wbase = (col * K) >> 2;
    int acc = 0;
    for (uint group = 0; group < (K >> 2); ++group)
        acc = dot4add_i8packed(A[base + group], W[wbase + group], acc);

    float y = float(acc) * asfloat(AScale[row]) * asfloat(WScale[col]);
    if (UseBias != 0)
        y += asfloat(Bias[col]);
    Y.Store(index * 4u, asuint(y));
}
