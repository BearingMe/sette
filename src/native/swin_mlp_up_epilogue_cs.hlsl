// EXP-026: replace only W1+b1; original GELU, W2 and residual are unchanged.
#define MLP_LOAD_UP(tok, chunk, j) HalfAt(B3, (tok) * (pNC * 32u) + (chunk) * 32u + (j))
#include "swin_layer_cs.hlsl"
