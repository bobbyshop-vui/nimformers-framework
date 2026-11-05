// Shared MSL/GLSL implementation. All weight arithmetic and packing run here.
uint magnitude(float v) {
    return v<0.25f?0u:v<0.75f?1u:v<1.25f?2u:v<1.75f?3u:
           v<2.5f?4u:v<3.5f?5u:v<5.0f?6u:7u;
}
uint encodeScale(float v) {
    uint bits=F2U(v);
    int exponent=int((bits>>23)&255u)-127+7;
    if (exponent<=0) return 0u;
    if (exponent>=15) return 120u;
    return (uint(exponent)<<3)|((bits>>20)&7u);
}
float merge_divide(float x,float y) {
    // Correct the hardware reciprocal/divide approximation using its FMA
    // residual. This preserves legacy FP32 rounding on MoltenVK as well.
    PRECISE float q=x/y;
    PRECISE float residual=fma(-q,y,x);
    PRECISE float correction=residual/y;
    return q+correction;
}
INL void loadGroup(uint group, OUT_VALUES PARAMS) {
    uint cols=state[4], rank=state[6];
    uint row=group/(cols/32u), col=(group%(cols/32u))*32u;
    uint sb=BYTE_AT(state[3]/2u+group);
    float s=lut[16u+sb];
    for(uint i=0u;i<16u;i++) {
        uint b=BYTE_AT(group*16u+i);
        values[i]=lut[b&15u]*s;
        values[i+16u]=lut[b>>4]*s;
    }
    for(uint r=0u;r<rank;r++) {
        PRECISE float b=adapter[rank*cols+row*rank+r]*U2F(state[7]);
        if(b==0.0f) continue;
        for(uint i=0u;i<32u;i++) {
            PRECISE float product=b*adapter[r*cols+col+i];
            values[i]=values[i]+product;
        }
    }
}
// ENTRY_MAX
{
    float mx=0.0f;
    if(gid<state[3]/32u) {
        PRECISE float values[32]; loadGroup(gid,values ARGS);
        for(uint i=0u;i<32u;i++) {
            if(!isfinite(values[i])) BAD_VALUE;
            mx=max(mx,abs(values[i]));
        }
    }
    maxmem[lid]=merge_divide(mx,6.0f);
    SYNC;
    for(uint stride=64u;stride>0u;stride>>=1u) {
        if(lid<stride) maxmem[lid]=max(maxmem[lid],maxmem[lid+stride]);
        SYNC;
    }
    if(lid==0u) MAX_VALUE(maxmem[0]);
}
// ENTRY_SCALE
{
    if(gid==0u) state[1]=F2U(state[0]!=0u?merge_divide(U2F(state[0]),224.0f):1.0f);
}
// ENTRY_QUANT
{
    uint group=state[2]+gid;
    if(gid>=state[8] || group>=state[3]/32u) return;
    PRECISE float values[32]; loadGroup(group,values ARGS);
    float mx=0.0f;
    for(uint i=0u;i<32u;i++) mx=max(mx,abs(values[i]));
    float gscale=U2F(state[1]);
    PRECISE float bestError=FLOAT_INF;
    float bestScale=0.0f;
    uint bestByte=0u;
    // mx/6 is loop invariant; hoisting keeps the exact same rounding.
    PRECISE float mx6=mx>0.0f?merge_divide(mx,6.0f):0.0f;
    if(mx>0.0f) for(uint step=0u;step<8u;step++) {
        PRECISE float ratio=0.62f+merge_divide(0.62f*float(step),7.0f);
        PRECISE float candidate=mx6*ratio;
        uint code=encodeScale(merge_divide(candidate,gscale));
        float decoded=adapter[state[6]*(state[4]+state[5])+code];
        if(decoded<=0.0f || decoded>1e30f || !isfinite(decoded)) continue;
        float scale=decoded*gscale;
        if(scale<=0.0f) continue;
        PRECISE float error=0.0f;
        float invScale=merge_divide(1.0f,scale);
        for(uint i=0u;i<32u;i++) {
            PRECISE float normalized=values[i]*invScale;
            PRECISE float q=lut[magnitude(abs(normalized))]*scale;
            PRECISE float d=(normalized<0.0f?-q:q)-values[i];
            error=error+d*d;
        }
        if(error<bestError) { bestError=error; bestByte=code; bestScale=scale; }
    }
    scratch[4u*state[8]+gid]=bestByte;
    // One reciprocal per group instead of one per packed word.
    float packInv=bestScale>0.0f?merge_divide(1.0f,bestScale):0.0f;
    for(uint word=0u;word<4u;word++) {
        uint wordBits=0u;
        for(uint j=0u;j<4u;j++) {
            uint i=word*4u+j;
            uint lo=0u,hi=0u;
            if(bestScale>0.0f) {
                float a=values[i]*packInv,b=values[i+16u]*packInv;
                lo=magnitude(abs(a))|(a<0.0f?8u:0u);
                hi=magnitude(abs(b))|(b<0.0f?8u:0u);
            }
            wordBits|=(lo|(hi<<4u))<<(j*8u);
        }
        scratch[gid*4u+word]=wordBits;
    }
}
// ENTRY_COMMIT
{
    uint count=min(state[8],state[3]/32u-state[2]);
    if(gid<count*4u) weights[state[2]*4u+gid]=scratch[gid];
    if(gid<count/4u) {
        uint wordBits=0u;
        for(uint i=0u;i<4u;i++) wordBits|=scratch[4u*state[8]+gid*4u+i]<<(i*8u);
        weights[state[3]/8u+state[2]/4u+gid]=wordBits;
    }
}
// ENTRY_ADVANCE
{
    if(gid==0u) state[2]+=state[8];
}
// ENTRY_LUT
{
    if(gid>=256u) return;
    float v=adapter[state[6]*(state[4]+state[5])+gid]*U2F(state[1]);
    lut[16u+gid]=!isfinite(v)||abs(v)>1e30f?0.0f:v;
}
