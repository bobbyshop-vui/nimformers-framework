// ═══════════════════════════════════════════════════════════════════════════
// API resident (session): nhiều op mã hoá vào CÙNG 1 command buffer, chỉ
// commit+wait MỘT LẦN ở metal_session_end. Theo đúng idiom metal_matmul2() ở
// trên (đã có sẵn, đang chạy) - chỉ khác là session này mở RA NHIỀU op tuỳ ý
// thay vì cố định 2 matmul.
static id<MTLCommandBuffer> gSessionCmdBuf = nil;
static id<MTLComputeCommandEncoder> gSessionEncoder = nil;

// Đóng encoder hiện tại (nếu có) trước khi mở encoder mới hoặc commit - Metal
// yêu cầu endEncoding trước khi tạo command encoder tiếp theo trên cùng command buffer.
static void closeCurrentEncoder(void) {
    if (gSessionEncoder) {
        [gSessionEncoder endEncoding];
        gSessionEncoder = nil;
    }
}

int metal_session_begin(const char* kernel_src) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        closeCurrentEncoder();
        gSessionCmdBuf = [gQueue commandBuffer];
        return gSessionCmdBuf != nil ? 1 : 0;
    }
}

MetalBufferHandle metal_upload(const float* data, int n) {
    @autoreleasepool {
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) return NULL;
        return (void*)CFBridgingRetain(buf);
    }
}

MetalBufferHandle metal_upload_indices(const int* data, int n) {
    @autoreleasepool {
        size_t bytes = (size_t)n * sizeof(int32_t);
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) return NULL;
        return (void*)CFBridgingRetain(buf);
    }
}

MetalBufferHandle metal_alloc_scratch(int n) {
    @autoreleasepool {
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> buf = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) return NULL;
        return (void*)CFBridgingRetain(buf);
    }
}

static id<MTLBuffer> asBuf(MetalBufferHandle h) {
    return (__bridge id<MTLBuffer>)h;
}

int metal_vecop_enc(int op, MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int n) {
    @autoreleasepool {
        if (!gSessionCmdBuf) return 0;
        NSString *fname = nil;
        switch (op) {
            case 0: fname = @"vecop_add"; break;
            case 1: fname = @"vecop_sub"; break;
            case 2: fname = @"vecop_mul"; break;
            case 3: fname = @"vecop_div"; break;
            default: fname = @"vecop_add"; break;
        }
        id<MTLComputePipelineState> pipeline = getPipeline(fname);
        if (!pipeline) return 0;
        closeCurrentEncoder();
        id<MTLComputeCommandEncoder> encoder = [gSessionCmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:asBuf(a) offset:0 atIndex:0];
        [encoder setBuffer:asBuf(b) offset:0 atIndex:1];
        [encoder setBuffer:asBuf(c) offset:0 atIndex:2];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        return 1;
    }
}

int metal_activation_enc(int op, MetalBufferHandle x, MetalBufferHandle y, int n) {
    @autoreleasepool {
        if (!gSessionCmdBuf) return 0;
        NSString *fname = nil;
        switch (op) {
            case 0: fname = @"vecop_relu"; break;
            case 1: fname = @"vecop_sigmoid"; break;
            case 2: fname = @"vecop_tanh"; break;
            default: fname = @"vecop_relu"; break;
        }
        id<MTLComputePipelineState> pipeline = getPipeline(fname);
        if (!pipeline) return 0;
        closeCurrentEncoder();
        id<MTLComputeCommandEncoder> encoder = [gSessionCmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:asBuf(x) offset:0 atIndex:0];
        [encoder setBuffer:asBuf(y) offset:0 atIndex:1];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        return 1;
    }
}

int metal_softmax_enc(MetalBufferHandle x, MetalBufferHandle y, int rows, int cols) {
    @autoreleasepool {
        if (!gSessionCmdBuf) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"softmax_kernel");
        if (!pipeline) return 0;
        closeCurrentEncoder();
        id<MTLComputeCommandEncoder> encoder = [gSessionCmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:asBuf(x) offset:0 atIndex:0];
        [encoder setBuffer:asBuf(y) offset:0 atIndex:1];
        [encoder setBytes:&cols length:sizeof(int) atIndex:2];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)rows);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        return 1;
    }
}

int metal_layernorm_enc(MetalBufferHandle x, MetalBufferHandle gamma, MetalBufferHandle beta,
                         MetalBufferHandle y, int rows, int cols, float eps) {
    @autoreleasepool {
        if (!gSessionCmdBuf) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_kernel");
        if (!pipeline) return 0;
        closeCurrentEncoder();
        id<MTLComputeCommandEncoder> encoder = [gSessionCmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:asBuf(x) offset:0 atIndex:0];
        [encoder setBuffer:asBuf(gamma) offset:0 atIndex:1];
        [encoder setBuffer:asBuf(beta) offset:0 atIndex:2];
        [encoder setBuffer:asBuf(y) offset:0 atIndex:3];
        [encoder setBytes:&cols length:sizeof(int) atIndex:4];
        [encoder setBytes:&eps length:sizeof(float) atIndex:5];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)rows);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        return 1;
    }
}

int metal_embedding_lookup_enc(MetalBufferHandle table, MetalBufferHandle indices,
                                MetalBufferHandle y, int vocab, int dim, int num_indices) {
    @autoreleasepool {
        if (!gSessionCmdBuf) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"embedding_lookup_kernel");
        if (!pipeline) return 0;
        closeCurrentEncoder();
        id<MTLComputeCommandEncoder> encoder = [gSessionCmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:asBuf(table) offset:0 atIndex:0];
        [encoder setBuffer:asBuf(indices) offset:0 atIndex:1];
        [encoder setBuffer:asBuf(y) offset:0 atIndex:2];
        [encoder setBytes:&vocab length:sizeof(int) atIndex:3];
        [encoder setBytes:&dim length:sizeof(int) atIndex:4];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_indices);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)num_indices, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        return 1;
    }
}

int metal_matmul_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n) {
    @autoreleasepool {
        if (!gSessionCmdBuf) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul_naive");
        if (!pipeline) return 0;
        closeCurrentEncoder();
        id<MTLComputeCommandEncoder> encoder = [gSessionCmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:asBuf(a) offset:0 atIndex:0];
        [encoder setBuffer:asBuf(b) offset:0 atIndex:1];
        [encoder setBuffer:asBuf(c) offset:0 atIndex:2];
        [encoder setBytes:&m length:sizeof(int) atIndex:3];
        [encoder setBytes:&k length:sizeof(int) atIndex:4];
        [encoder setBytes:&n length:sizeof(int) atIndex:5];
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, (NSUInteger)m, 1);
        NSUInteger tw = MIN((NSUInteger)16, (NSUInteger)n);
        NSUInteger th = MIN((NSUInteger)16, (NSUInteger)m);
        if (tw == 0) tw = 1;
        if (th == 0) th = 1;
        MTLSize groupSize = MTLSizeMake(tw, th, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        return 1;
    }
}

int metal_session_end(void) {
    @autoreleasepool {
        if (!gSessionCmdBuf) return 0;
        closeCurrentEncoder();
        [gSessionCmdBuf commit];
        [gSessionCmdBuf waitUntilCompleted];
        BOOL ok = (gSessionCmdBuf.status == MTLCommandBufferStatusCompleted);
        if (!ok) {
            NSLog(@"[metal_session_end] command buffer FAILED, status=%ld, error=%@",
                  (long)gSessionCmdBuf.status, gSessionCmdBuf.error);
        }
        gSessionCmdBuf = nil;
        return ok ? 1 : 0;
    }
}

int metal_buffer_read(MetalBufferHandle h, float* outData, int n) {
    @autoreleasepool {
        id<MTLBuffer> buf = asBuf(h);
        if (!buf) return 0;
        memcpy(outData, [buf contents], (size_t)n * sizeof(float));
        return 1;
    }
}

void metal_buffer_free(MetalBufferHandle h) {
    if (!h) return;
    // Trả quyền sở hữu lại cho ARC - Metal đã tự retain buffer bên trong lệnh
    // encode nếu command buffer đang chạy còn tham chiếu tới nó, nên release ở
    // đây không làm buffer biến mất giữa chừng khi GPU còn đang đọc/ghi nó.
    CFRelease(h);
}
