// M05 B5: the Preset M Swin block as native D3D12 compute.
//
// One entry point per operator; a layer is a fixed sequence of dispatches over
// GPU-resident buffers (see `m_network_stage.cpp` and `scripts/m_layer_fixture.py`).
// Every binding is a 32-bit-word view of f16 / f32 / e4m3 data, so no byte is
// reinterpreted by the host.
//
// Reconstruction contract (EXP-006/EXP-017/EXP-019): each 32-product step uses
// an exact product sum plus the f16 accumulator, then one RNE half rounding.
// PTX declares f16 C/D but leaves internal accumulation order/rounding unspecified;
// this exact model is not a claim of NVIDIA or translated-WMMA bit fidelity. Every f16
// operation rounds once; the RMS sums use the kernel's own trees; attention uses
// the clamped cubic, the bit-level exponent and the c^8,c^2,c^4,c^1 butterfly;
// GELU is the captured cubic; the patch merge clamps to +-2*pi.
//
// f16 rounding is explicit integer arithmetic (EXP-010: the driver's own f32->f16
// conversion truncates). `Add16`/`Fma16` correct the one double-rounding case -
// an f32 result landing exactly on an f16 midpoint - with the exact residual, so
// a chained f16 operation is a single rounding, as on the hardware.
//
// 16-bit integer types are deliberately not used: DXC miscompiles small masks
// under -enable-16bit-types (a 16-bit truncation evaluates to zero).

// ---------------------------------------------------------------- f16 helpers

uint ToHalfBits(float value)
{
    uint u = asuint(value);
    uint sign = (u >> 16) & 0x8000u;
    int expo = int((u >> 23) & 0xFFu) - 127 + 15;
    uint mant = u & 0x7FFFFFu;
    if (expo <= 0)
    {
        int shift = 14 - expo;
        if (shift > 24) return sign;
        uint full = (mant | 0x800000u) >> shift;
        uint rest = (mant | 0x800000u) & ((1u << shift) - 1u);
        uint half = 1u << (shift - 1);
        if (rest > half || (rest == half && (full & 1u))) full += 1u;
        return sign | full;
    }
    if (expo >= 31) return sign | 0x7C00u;
    uint h = (uint(expo) << 10) | (mant >> 13);
    uint rest = mant & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) h += 1u;
    return sign | h;
}

float R16(float v) { return f16tof32(ToHalfBits(v)); }
// exact double -> f16: one RNE conversion.  The double is first rounded to f32;
// that only double-rounds when the f32 value lands exactly on an f16 midpoint,
// where the exact residual decides the direction.
float R16D(double v)
{
    float f = (float)v;
    uint u = asuint(f);
    if ((u & 0x1FFFu) == 0x1000u)
    {
        double d = v - (double)f;
        if (d != 0.0)
        {
            bool up = ((d > 0.0) == ((u & 0x80000000u) == 0u));
            return R16(asfloat(up ? u + 1u : u - 1u));
        }
    }
    return R16(f);
}
float F16ToF32(uint bits) { return f16tof32(bits & 0xFFFFu); }
uint HalfBits(float v) { return ToHalfBits(v); }

// f16 operations: one rounding of the exact result.  The inputs are f16 values,
// so the double expressions are exact and R16D is the single rounding the
// hardware performs.
float Add16(float a, float b) { return R16D((double)a + (double)b); }
float Fma16(float a, float b, float c) { return R16D((double)a * (double)b + (double)c); }
float Mul16(float a, float b) { return R16D((double)a * (double)b); }
float HAdd(float a, float b) { return Add16(a, b); }
float HMul(float a, float b) { return Mul16(a, b); }
float HFma(float a, float b, float c) { return Fma16(a, b, c); }

float E4M3(uint code)
{
    uint bits = ((code & 0x7Fu) << 7) | ((code & 0x80u) << 8);
    return F16ToF32(bits) * 256.0f;
}

uint CodeAt(Buffer<uint> b, uint byteIndex)
{
    uint word = b[byteIndex >> 2];
    return (word >> (8u * (byteIndex & 3u))) & 0xFFu;
}

// q8_exact from swin_common.h: e4m3 RNE satfinite, NaN payload kept
uint Q8Bits(uint u)
{
    uint m = u & 0x7FFFu;
    uint r = (m + 0x3Fu + ((m >> 7) & 1u)) & 0x7F80u;
    r = r < 0x5F00u ? r : 0x5F00u;
    float mag = F16ToF32(m);
    float subf = R16(HAdd(mag, 2.0f) - 2.0f);
    uint sub = HalfBits(subf);
    r = m < 0x2400u ? sub : r;
    r = m > 0x7C00u ? m : r;
    return (u & 0x8000u) | r;
}
float Q8(float v) { return F16ToF32(Q8Bits(HalfBits(v))); }

uint Enc8(float v)
{
    uint u = Q8Bits(HalfBits(v));
    uint m = u & 0x7FFFu;
    uint code = m >= 0x2400u ? (m >> 7) - 64u : (uint)Mul16(F16ToF32(m), 512.0f);
    code = m > 0x7C00u ? 0x7Fu : code;
    return ((u >> 8) & 0x80u) | code;
}

uint GPerm(uint c)
{
    return 32u * (c >> 5) + 2u * ((c & 31u) >> 2) + (c & 1u) + 16u * ((c & 3u) >> 1);
}
uint GInv(uint n)
{
    uint m = n & 31u, e = m >> 4, a = (m & 15u) >> 1, b = m & 1u;
    return (n & ~31u) + 4u * a + 2u * e + b;
}

int Mirror(int v, int n)
{
    v = v < 0 ? -v : v;
    int o = 2 * n - 2 - v;
    return v < o ? v : o;
}

float HalfAt(Buffer<uint> b, uint i)
{
    uint w = b[i >> 1];
    return F16ToF32((i & 1u) ? (w >> 16) : (w & 0xFFFFu));
}
// two halves into one word; no read-modify-write on the UAV
void Store2(RWBuffer<uint> b, uint i, float lo, float hi)
{
    b[i >> 1] = HalfBits(lo) | (HalfBits(hi) << 16);
}

float Rms(float s)
{
    float sf = Add16(s, 0.0001220703125f);          // 2^-13, f16
    precise float root = sqrt(sf);
    precise float q = 1.0f / root;
    return R16(q);
}

// ---------------------------------------------------------------- parameters

cbuffer Params : register(b0)
{
    uint pC, pHeads, pGW, pGH, pW, pH, pSx, pSy;
    uint pTokens, pLowW, pLowH, pNC, pMode, pSrcFmt, pSrcW, pSrcH;
    uint pK, pN, pC0Mode, pOffG1, pOffBo, pOffG2, pOffB2;
    uint pOffQ, pOffV, pOffWo, pOffW1, pOffB1, pOffW2, pOffBias;
    uint pOffMergeW, pOffMergeB, pOffExpandW, pOffExpandB, pNPM, pNT8;
    uint pOffMatrix, pReserved1, pReserved2, pReserved3;
};

Buffer<uint> B0 : register(t0);
Buffer<uint> B1 : register(t1);
Buffer<uint> B2 : register(t2);
Buffer<uint> B3 : register(t3);
RWBuffer<uint> U0 : register(u0);
RWBuffer<uint> U1 : register(u1);

// ---------------------------------------------------------------- Prepare

[numthreads(64, 1, 1)]
void Prepare(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pTokens)
        return;
    uint px = tok % pGW, py = tok / pGW;
    int x = (int)px - (int)pSx, y = (int)py - (int)pSy;
    int mx = Mirror(x, (int)pW), my = Mirror(y, (int)pH);
    if (pMode == 0)
    {
        for (uint c = 0; c < pC; c += 2u)
        {
            uint off0 = ((c >> 5) * pH + (uint)my) * pW + (uint)mx;
            uint off1 = (((c + 1u) >> 5) * pH + (uint)my) * pW + (uint)mx;
            float v0 = E4M3(CodeAt(B0, off0 * 32u + (c & 31u)));
            float v1 = E4M3(CodeAt(B0, off1 * 32u + ((c + 1u) & 31u)));
            Store2(U0, tok * pC + c, v0, v1);
        }
        return;
    }
    int Yb = ((y + (int)pSy) >> 3) * 8 - (int)pSy;
    int Xb = ((x + (int)pSx) >> 3) * 8 - (int)pSx;
    int ly = Mirror((Yb >> 1) + ((y - Yb) >> 1), (int)pLowH);
    int lx = Mirror((Xb >> 1) + ((x - Xb) >> 1), (int)pLowW);
    uint q = (uint)(((y - Yb) & 1) << 1 | ((x - Xb) & 1));
    for (uint c = 0; c < pC; c += 2u)
    {
        float e0 = HalfAt(B0, ((uint)(ly * (int)pLowW + lx) * 4u + q) * pC + c);
        float e1 = HalfAt(B0, ((uint)(ly * (int)pLowW + lx) * 4u + q) * pC + c + 1u);
        uint off0 = ((c >> 5) * pH + (uint)my) * pW + (uint)mx;
        uint off1 = (((c + 1u) >> 5) * pH + (uint)my) * pW + (uint)mx;
        float s0 = E4M3(CodeAt(B1, off0 * 32u + (c & 31u)));
        float s1 = E4M3(CodeAt(B1, off1 * 32u + ((c + 1u) & 31u)));
        Store2(U0, tok * pC + c, Add16(Q8(e0), s0), Add16(Q8(e1), s1));
    }
}

// ---------------------------------------------------------------- Norm

float SquareAt(Buffer<uint> X, uint base, uint c)
{
    float v = HalfAt(X, base + c);
    return Mul16(v, v);
}

float WordSum(uint base, uint word, uint e, uint planes)
{
    float pv[4];
    for (uint p = 0; p < planes; ++p)
    {
        uint c = 32u * p + 4u * word;
        pv[p] = Add16(SquareAt(B0, base, c + e), SquareAt(B0, base, c + 2u + e));
    }
    if (planes == 4u) return Add16(Add16(pv[0], pv[1]), Add16(pv[2], pv[3]));
    if (planes == 3u) return Add16(Add16(pv[0], pv[1]), pv[2]);
    return Add16(pv[0], pv[1]);
}

float SumSqTokens(uint base)
{
    uint planes = pC / 32u;
    float lane0[4], lane1[4];
    for (uint t = 0; t < 4; ++t)
    {
        lane0[t] = Add16(WordSum(base, t, 0, planes), WordSum(base, t + 4u, 0, planes));
        lane1[t] = Add16(WordSum(base, t, 1, planes), WordSum(base, t + 4u, 1, planes));
    }
    float a0[4], a1[4];
    for (uint t = 0; t < 4; ++t)
    {
        a0[t] = Add16(lane0[t], lane0[t ^ 1u]);
        a1[t] = Add16(lane1[t], lane1[t ^ 1u]);
    }
    return Add16(Add16(a0[0], a0[2]), Add16(a1[0], a1[2]));
}

float SqPair(uint base, uint pairCol)
{
    float v = HalfAt(B0, base + GInv(pairCol));
    return Mul16(v, v);
}
float QuadPair(uint base, uint col, uint a, uint b, uint c, uint d)
{
    return Add16(Add16(SqPair(base, 8u * a + col), SqPair(base, 8u * b + col)),
                 Add16(SqPair(base, 8u * c + col), SqPair(base, 8u * d + col)));
}
float TreeNorm2(uint base)
{
    uint n = pNT8;
    float L[8];
    for (uint col = 0; col < 8u; ++col)
    {
        float even, odd;
        if (n == 16u)
        {
            even = Add16(QuadPair(base, col, 0, 2, 4, 6), QuadPair(base, col, 8, 10, 12, 14));
            odd = Add16(QuadPair(base, col, 1, 3, 5, 7), QuadPair(base, col, 9, 11, 13, 15));
        }
        else if (n == 12u)
        {
            even = Add16(QuadPair(base, col, 0, 2, 4, 6),
                         Add16(SqPair(base, 8u * 8u + col), SqPair(base, 8u * 10u + col)));
            odd = Add16(QuadPair(base, col, 1, 3, 5, 7),
                        Add16(SqPair(base, 8u * 9u + col), SqPair(base, 8u * 11u + col)));
        }
        else
        {
            even = QuadPair(base, col, 0, 2, 4, 6);
            odd = QuadPair(base, col, 1, 3, 5, 7);
        }
        L[col] = Add16(even, odd);
    }
    float b0 = Add16(Add16(L[0], L[2]), Add16(L[4], L[6]));
    float b1 = Add16(Add16(L[1], L[3]), Add16(L[5], L[7]));
    return Add16(b0, b1);
}

// mode 0: norm1 (sumsq_tokens, gamma g1, residual x0+bo)
// mode 1: norm2 (tree_norm2, gamma g2, x2 = x1+b2)
[numthreads(64, 1, 1)]
void Norm(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pTokens)
        return;
    uint base = tok * pC;
    float s = pMode == 0u ? SumSqTokens(base) : TreeNorm2(base);
    float r = Rms(s);
    uint gammaOff = pMode == 0u ? pOffG1 : pOffG2;
    uint resOff = pMode == 0u ? pOffBo : pOffB2;
    for (uint c = 0; c < pC; c += 2u)
    {
        float x0 = HalfAt(B0, base + c);
        float x1 = HalfAt(B0, base + c + 1u);
        float g0 = asfloat(B1[gammaOff + c]);
        float g1 = asfloat(B1[gammaOff + c + 1u]);
        Store2(U0, base + c, Q8(Mul16(x0, Mul16(r, g0))), Q8(Mul16(x1, Mul16(r, g1))));
        Store2(U1, base + c, Add16(x0, asfloat(B2[resOff + c])),
               Add16(x1, asfloat(B2[resOff + c + 1u])));
    }
}

// ---------------------------------------------------------------- Matmul

float AAt(uint tok, uint k)
{
    if (pSrcFmt == 0u)
        return HalfAt(B0, tok * pK + k);
    uint px = tok % pSrcW, py = tok / pSrcW;
    uint off = ((k >> 5) * pSrcH + py) * pSrcW + px;
    return E4M3(CodeAt(B0, off * 32u + (k & 31u)));
}

[numthreads(64, 1, 1)]
void Matmul(uint3 id : SV_DispatchThreadID)
{
    uint nt = pN / 16u;
    uint tile = id.x % nt, tok = id.x / nt;
    if (tok >= pTokens)
        return;
    uint n0 = tile * 16u;
    double acc[16];
    for (uint j = 0; j < 16u; ++j)
    {
        if (pC0Mode == 0u) acc[j] = 0.0;
        else if (pC0Mode == 1u) acc[j] = (double)asfloat(B2[pOffBias + n0 + j]);
        else acc[j] = (double)HalfAt(B2, tok * pN + n0 + j);
    }
    for (uint k0 = 0; k0 < pK; k0 += 32u)
    {
        double sum[16];
        for (uint j = 0; j < 16u; ++j) sum[j] = 0.0;
        for (uint k = 0; k < 32u; ++k)
        {
            double a = (double)AAt(tok, k0 + k);
            uint row = pOffMatrix + (k0 + k) * pN + n0;
            for (uint j = 0; j < 16u; ++j)
                sum[j] += a * (double)asfloat(B1[row + j]);
        }
        for (uint j = 0; j < 16u; ++j)
            acc[j] = (double)R16D(sum[j] + acc[j]);
    }
    for (uint j = 0; j < 16u; j += 2u)
        Store2(U0, tok * pN + n0 + j, (float)acc[j], (float)acc[j + 1u]);
}

// ---------------------------------------------------------------- Attention

float CScale() { return R16(0.0972222164273262f / 5.656854152679443f); }

[numthreads(64, 1, 1)]
void Attention(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pTokens)
        return;
    uint px = tok % pGW, py = tok / pGW;
    uint bx = px >> 3, tx = px & 7u;
    uint by = py >> 3, ty = py & 7u;
    uint wx = tx >> 2, wy = ty >> 2;
    uint tok2[16];
    for (uint r = 0; r < 4u; ++r)
        for (uint a = 0; a < 4u; ++a)
            tok2[r * 4u + a] = (8u * by + 4u * wy + r) * pGW + (8u * bx + 4u * wx + a);

    uint row = (ty & 3u) * 4u + (tx & 3u);
    float cs = CScale();
    float cl = R16(0.55615234375f);
    float p1 = R16(0.92730712890625f);
    float p0 = R16(1.375f);

    for (uint h = 0; h < pHeads; ++h)
    {
        uint qbase = 32u * h;
        float S[16];
        for (uint col = 0; col < 16u; ++col)
        {
            double sum = 0.0;
            for (uint k = 0; k < 32u; ++k)
            {
                float qa = Q8(HalfAt(B0, tok2[row] * 32u * pHeads + qbase + k));
                float qb = Q8(HalfAt(B0, tok2[col] * 32u * pHeads + qbase + k));
                sum += (double)qa * (double)qb;
            }
            S[col] = R16D(sum + (double)asfloat(B2[pOffBias + h * 256u + row * 16u + col]));
        }
        float poly[16];
        for (uint c = 0; c < 16u; ++c)
        {
            float tv = Mul16(S[c], cs);
            tv = tv < -cl ? -cl : (tv > cl ? cl : tv);
            float e1 = Fma16(tv, -tv, p1);
            poly[c] = Fma16(tv, e1, p0);
        }
        float w[16];
        for (uint j = 0; j < 8u; ++j)
        {
            uint lo = HalfBits(poly[2u * j]), hi = HalfBits(poly[2u * j + 1u]);
            uint word = ((lo | (hi << 16)) << 5) + 0x7FF88000u;
            w[2u * j] = F16ToF32(word & 0xFFFFu);
            w[2u * j + 1u] = F16ToF32(word >> 16);
        }
        float s8[16], s2[16], s4[16], s1[16];
        for (uint c = 0; c < 16u; ++c) s8[c] = Add16(w[c], w[c ^ 8u]);
        for (uint c = 0; c < 16u; ++c) s2[c] = Add16(s8[c], s8[c ^ 2u]);
        for (uint c = 0; c < 16u; ++c) s4[c] = Add16(s2[c], s2[c ^ 4u]);
        for (uint c = 0; c < 16u; ++c) s1[c] = Add16(s4[c], s4[c ^ 1u]);
        float rinv = R16(1.0f / s1[0]);
        float P[16];
        for (uint c = 0; c < 16u; ++c) P[c] = Mul16(w[c], rinv);
        for (uint c = 0; c < 32u; c += 2u)
        {
            double sum0 = 0.0, sum1 = 0.0;
            for (uint k = 0; k < 16u; ++k)
            {
                double pk = (double)P[k];
                sum0 += pk * (double)HalfAt(B1, tok2[k] * 32u * pHeads + qbase + c);
                sum1 += pk * (double)HalfAt(B1, tok2[k] * 32u * pHeads + qbase + c + 1u);
            }
            Store2(U0, tok * 32u * pHeads + qbase + c,
                   Q8(R16D(sum0)), Q8(R16D(sum1)));
        }
    }
}

// ---------------------------------------------------------------- MLP

float GeluQ(float z)
{
    float cz = z < -2.0f ? -2.0f : (z > 2.0f ? 2.0f : z);
    float az = abs(cz);
    float inner = Add16(0.412109375f,
                        -Mul16(0.0810546875f, az));
    float gl = Mul16(z, Add16(0.5f, Mul16(cz, inner)));
    return Q8(gl);
}

[numthreads(64, 1, 1)]
void Mlp(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pTokens)
        return;
    uint base = tok * pC;
    float xv[128];
    for (uint c = 0; c < pC; ++c) xv[c] = HalfAt(B2, base + c);
    for (uint chunk = 0; chunk < pNC; ++chunk)
    {
        uint w1 = pOffW1 + chunk * pC * 32u;
        uint b1 = pOffB1 + chunk * 32u;
        float z[32];
        for (uint j = 0; j < 32u; ++j) z[j] = asfloat(B1[b1 + j]);
        for (uint k0 = 0; k0 < pC; k0 += 32u)
        {
            double sum[32];
            for (uint j = 0; j < 32u; ++j) sum[j] = 0.0;
            for (uint k = 0; k < 32u; ++k)
            {
                double a = (double)HalfAt(B0, base + k0 + k);
                uint row = w1 + (k0 + k) * 32u;
                for (uint j = 0; j < 32u; ++j)
                    sum[j] += a * (double)asfloat(B1[row + j]);
            }
            for (uint j = 0; j < 32u; ++j)
                z[j] = R16D(sum[j] + (double)z[j]);
        }
        float g[32];
        for (uint j = 0; j < 32u; ++j) g[j] = GeluQ(z[j]);
        uint w2 = pOffW2 + chunk * 32u * pC;
        for (uint c = 0; c < pC; c += 2u)
        {
            double sum0 = 0.0, sum1 = 0.0;
            for (uint j = 0; j < 32u; ++j)
            {
                double gj = (double)g[j];
                sum0 += gj * (double)asfloat(B1[w2 + j * pC + c]);
                sum1 += gj * (double)asfloat(B1[w2 + j * pC + c + 1u]);
            }
            float x0 = xv[c];
            float x1 = xv[c + 1u];
            xv[c] = R16D(sum0 + (double)x0);
            xv[c + 1u] = R16D(sum1 + (double)x1);
        }
    }
    for (uint c = 0; c < pC; c += 2u)
        Store2(U0, base + c, xv[c], xv[c + 1u]);
}

// ---------------------------------------------------------------- Store

[numthreads(64, 1, 1)]
void Store(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pTokens)
        return;
    uint px = tok % pGW, py = tok / pGW;
    int x = (int)px - (int)pSx, y = (int)py - (int)pSy;
    if (x < 0 || x >= (int)pW || y < 0 || y >= (int)pH)
        return;
    for (uint c = 0; c < pC; c += 4u)
    {
        uint code = Enc8(HalfAt(B0, tok * pC + c)) | (Enc8(HalfAt(B0, tok * pC + c + 1u)) << 8)
                  | (Enc8(HalfAt(B0, tok * pC + c + 2u)) << 16)
                  | (Enc8(HalfAt(B0, tok * pC + c + 3u)) << 24);
        uint off = (((c >> 5) * pH + (uint)y) * pW + (uint)x) * 8u + ((c >> 2) & 7u);
        U0[off] = code;
    }
}

// ---------------------------------------------------------------- Merge

[numthreads(64, 1, 1)]
void Merge(uint3 id : SV_DispatchThreadID)
{
    uint halfW = pGW / 2u, halfH = pGH / 2u;
    uint nt = pNPM / 16u;
    uint tile = id.x % nt;
    uint merged = id.x / nt;
    if (merged >= halfW * halfH)
        return;
    uint mx2 = merged % halfW, my2 = merged / halfW;
    uint bx = mx2 >> 2, mx = mx2 & 3u, by = my2 >> 2, my = my2 & 3u;
    uint n0 = tile * 16u;
    float acc[16];
    for (uint j = 0; j < 16u; ++j) acc[j] = asfloat(B1[pOffMergeB + n0 + j]);
    for (uint k0 = 0; k0 < 4u * pC; k0 += 32u)
    {
        double sum[16];
        for (uint j = 0; j < 16u; ++j) sum[j] = 0.0;
        for (uint k = 0; k < 32u; ++k)
        {
            uint kk = k0 + k;
            uint q = kk / pC, c = kk % pC;
            uint xp = 8u * bx + 2u * mx + (q & 1u);
            uint yp = 8u * by + 2u * my + (q >> 1);
            double a = (double)Q8(HalfAt(B0, (yp * pGW + xp) * pC + c));
            uint row = pOffMergeW + kk * pNPM + n0;
            for (uint j = 0; j < 16u; ++j)
                sum[j] += a * (double)asfloat(B1[row + j]);
        }
        for (uint j = 0; j < 16u; ++j)
            acc[j] = R16D(sum[j] + (double)acc[j]);
    }
    int MX = (int)mx2 - (int)(pSx / 2u), MY = (int)my2 - (int)(pSy / 2u);
    if (MX < 0 || MX >= (int)(pW / 2u) || MY < 0 || MY >= (int)(pH / 2u))
        return;
    for (uint j = 0; j < 16u; j += 4u)
    {
        float v0 = acc[j], v1 = acc[j + 1u], v2 = acc[j + 2u], v3 = acc[j + 3u];
        v0 = v0 < -6.2831855f ? -6.2831855f : (v0 > 6.2831855f ? 6.2831855f : v0);
        v1 = v1 < -6.2831855f ? -6.2831855f : (v1 > 6.2831855f ? 6.2831855f : v1);
        v2 = v2 < -6.2831855f ? -6.2831855f : (v2 > 6.2831855f ? 6.2831855f : v2);
        v3 = v3 < -6.2831855f ? -6.2831855f : (v3 > 6.2831855f ? 6.2831855f : v3);
        uint code = Enc8(v0) | (Enc8(v1) << 8) | (Enc8(v2) << 16) | (Enc8(v3) << 24);
        uint nh = n0 + j;
        uint off = (((nh >> 5) * (pH / 2u) + (uint)MY) * (pW / 2u) + (uint)MX) * 8u
                 + ((nh >> 2) & 7u);
        U0[off] = code;
    }
}
