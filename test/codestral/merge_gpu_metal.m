#import <Metal/Metal.h>

static id<MTLComputePipelineState> gPipelines[6];
static id<MTLDevice> gPipelinesDevice;
static id<MTLCommandQueue> gMergeQueue;
static id<MTLDevice> gMergeQueueDevice;

static void mergeEncode(id<MTLComputeCommandEncoder> encoder,
                        __strong id<MTLBuffer> *buffers,
                        int stage, int count) {
    [encoder setComputePipelineState:gPipelines[stage]];
    for (int b = 0; b < 5; b++) [encoder setBuffer:buffers[b] offset:0 atIndex:b];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

int codestral_merge_gpu_metal(const char **sources, void *scratch, void *weights,
                             void *lut, void *adapter, void *state,
                             int groups, int tile) {
    @autoreleasepool {
        __strong id<MTLBuffer> buffers[5]={(__bridge id<MTLBuffer>)scratch,(__bridge id<MTLBuffer>)weights,
            (__bridge id<MTLBuffer>)lut,(__bridge id<MTLBuffer>)adapter,(__bridge id<MTLBuffer>)state};
        id<MTLDevice> device=buffers[0].device;
        static id<MTLDevice> cachedDevice;
        const char *names[6]={"max","scale","quant","commit","advance","lut"};
        for(int i=0;i<6;i++) if(!gPipelines[i] || cachedDevice!=device) {
            NSError *error=nil;
            MTLCompileOptions *options=[MTLCompileOptions new];
            options.fastMathEnabled=NO;
            id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:sources[i]]
                options:options error:&error];
            id<MTLFunction> function=[library newFunctionWithName:
                [NSString stringWithFormat:@"codestral_merge_gpu_%s",names[i]]];
            if(function) gPipelines[i]=[device newComputePipelineStateWithFunction:function error:&error];
            if(!library || !function || !gPipelines[i]) { NSLog(@"[merge-gpu] compile: %@",error); return 0; }
        }
        cachedDevice=device;
        // One queue for the whole process: a fresh queue per tensor leaked one
        // MTLCommandQueue per merged weight (128 per run).
        if(!gMergeQueue || gMergeQueueDevice!=device) {
            gMergeQueue=[device newCommandQueue];
            if(!gMergeQueue) return 0;
            gMergeQueueDevice=device;
        }
        id<MTLCommandQueue> queue=gMergeQueue;
        int ntiles=(groups+tile-1)/tile;
        // Two submissions per tensor instead of 2+ntiles: the tile loop only
        // needs in-order execution and buffer barriers, which one command
        // buffer already provides.
        for(int pass=0;pass<2;pass++) {
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if(pass==0) {
                mergeEncode(encoder,buffers,0,((groups+127)/128)*128);
                mergeEncode(encoder,buffers,1,1);
            } else {
                for(int phase=0;phase<ntiles;phase++) {
                    int count=MIN(tile,groups-phase*tile);
                    mergeEncode(encoder,buffers,2,count);
                    mergeEncode(encoder,buffers,3,count*4);
                    mergeEncode(encoder,buffers,4,1);
                }
                mergeEncode(encoder,buffers,5,256);
            }
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status!=MTLCommandBufferStatusCompleted) { NSLog(@"[merge-gpu] %@",command.error); return 0; }
            if(pass==0 && ((uint32_t*)buffers[4].contents)[9]) return 0;
        }
        return 1;
    }
}
