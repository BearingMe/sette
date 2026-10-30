// EXP-025: the existing dec0 epilogue, fed by its GPU-resident expansion.
Buffer<uint> Expansion : register(t2);
float ExpansionColumn(uint ty, uint tx, uint physicalColumn)
{
    uint n = (physicalColumn / 32u) * 20u + physicalColumn % 32u;
    uint index = (ty * 480u + tx) * 80u + n;
    uint word = Expansion[index >> 1];
    return f16tof32((index & 1u) ? word >> 16 : word & 65535u);
}
#define DEC0_LOAD_COLUMN ExpansionColumn
#include "dec0_cs.hlsl"
