// metal_shim.m - Metal compute backend implementation (macOS only).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "metal_shim.h"

// ─────────────────────────────────────────────────────────────
// State toàn cục, khởi tạo 1 lần, sống tới khi process thoát.
// ─────────────────────────────────────────────────────────────
static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;
static id<MTLLibrary> gLibrary = nil;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *gPipelineCache = nil;
static dispatch_once_t gInitOnce;

// MTLCreateSystemDefaultDevice() chỉ trả về ĐÚNG MỘT GPU "mặc định" theo lựa
// chọn của hệ thống. Trên Mac Apple Silicon chỉ có 1 GPU nên luôn ổn, nhưng
// trên Mac Intel máy có thể có NHIỀU GPU (iGPU Intel + dGPU AMD, hoặc thêm
// eGPU rời) -- default device đôi khi không phải GPU thực sự đang "sống"
// (vd. dGPU đang ngủ do auto graphics-switching, hoặc app không có quyền
// power-on dGPU). Dùng MTLCopyAllDevices() (chỉ có trên macOS, không có trên
// iOS) để liệt kê toàn bộ GPU và tự chọn/thử lần lượt thay vì tin vào 1 default.
static id<MTLDevice> pickMetalDevice(void) {
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    if (all.count == 0) {
        NSLog(@"[metal] MTLCopyAllDevices() rong, thu MTLCreateSystemDefaultDevice()");
        return MTLCreateSystemDefaultDevice();
    }

    NSLog(@"[metal] Tim thay %lu GPU:", (unsigned long)all.count);
    for (id<MTLDevice> d in all) {
        NSLog(@"[metal]   - %@ (lowPower=%d, removable=%d, headless=%d)",
              d.name, d.isLowPower, d.isRemovable, d.isHeadless);
    }

    // Ưu tiên: GPU rời/hiệu năng cao (không lowPower) và không headless.
    for (id<MTLDevice> d in all) {
        if (!d.isLowPower) {
            NSLog(@"[metal] Chon GPU hieu nang cao: %@", d.name);
            return d;
        }
    }
    // Không có GPU rời -> lấy GPU đầu tiên tìm được (vd. iGPU tích hợp).
    NSLog(@"[metal] Khong co GPU rieng, dung GPU dau tien: %@", all[0].name);
    return all[0];
}

// Khởi tạo device + queue + library MỘT LẦN DUY NHẤT cho toàn bộ process.
// Trả về NO nếu thất bại (vd. không có GPU, hoặc compile MSL lỗi).
static BOOL ensureMetalInit(const char* kernel_src) {
    __block BOOL ok = YES;
    dispatch_once(&gInitOnce, ^{
        gDevice = pickMetalDevice();
        if (!gDevice) { NSLog(@"[metal] khong tim thay GPU Metal nao"); ok = NO; return; }

        gQueue = [gDevice newCommandQueue];
        if (!gQueue) { NSLog(@"[metal] newCommandQueue tra ve nil"); ok = NO; return; }

        NSString *source = [NSString stringWithUTF8String:kernel_src];
        NSError *error = nil;
        gLibrary = [gDevice newLibraryWithSource:source options:nil error:&error];
        if (!gLibrary) { NSLog(@"[metal] newLibraryWithSource FAILED (compile 1 lan luc khoi dong): %@", error); ok = NO; return; }

        gPipelineCache = [NSMutableDictionary dictionary];
        NSLog(@"[metal] Khoi tao xong: device=%@, kernel library da compile 1 lan.", gDevice.name);
    });
    return ok && gDevice != nil && gLibrary != nil;
}

// Lấy pipeline state đã cache theo tên hàm; compile pipeline (không compile
// lại source) chỉ ở lần đầu gặp tên hàm đó, các lần sau lấy thẳng từ cache.
static id<MTLComputePipelineState> getPipeline(NSString *fname) {
    id<MTLComputePipelineState> cached = gPipelineCache[fname];
    if (cached) return cached;

    id<MTLFunction> fn = [gLibrary newFunctionWithName:fname];
    if (!fn) { NSLog(@"[metal] newFunctionWithName:%@ tra ve nil", fname); return nil; }

    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [gDevice newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) { NSLog(@"[metal] newComputePipelineStateWithFunction FAILED cho %@: %@", fname, error); return nil; }

    gPipelineCache[fname] = pipeline;
    return pipeline;
}

int metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = pickMetalDevice();
        return device != nil ? 1 : 0;
    }
}

int metal_vecop(const char* kernel_src, int op, const float* a, const float* b, float* c, int n) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;

        id<MTLComputePipelineState> pipeline = getPipeline(@"vecop_kernel");
        if (!pipeline) return 0;

        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [gDevice newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) { NSLog(@"[metal_vecop] tao buffer FAILED (bytes=%zu, n=%d)", bytes, n); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&op length:sizeof(int) atIndex:3];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_vecop] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c, [bufC contents], bytes);
        return 1;
    }
}

int metal_matmul(const char* kernel_src, const float* a, const float* b, float* c, int m, int k, int n) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;

        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul_kernel");
        if (!pipeline) return 0;

        size_t bytesA = (size_t)m * (size_t)k * sizeof(float);
        size_t bytesB = (size_t)k * (size_t)n * sizeof(float);
        size_t bytesC = (size_t)m * (size_t)n * sizeof(float);

        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [gDevice newBufferWithBytes:b length:bytesB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) { NSLog(@"[metal_matmul] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&m length:sizeof(int) atIndex:3];
        [encoder setBytes:&k length:sizeof(int) atIndex:4];
        [encoder setBytes:&n length:sizeof(int) atIndex:5];

        // Kernel uses a fixed threadgroup float tileA/tileB[16][16]; the
        // threadgroup size must always be exactly 16x16 (padded groups),
        // matching BB_TILE in the .metal source. Using a shrunk group for
        // small m/n (old code) left tile slots uninitialized -> wrong results.
        MTLSize groupSize = MTLSizeMake(8, 8, 1);  // SUA: khop BB_TILE=8 (fix GPU Timeout tren iGPU yeu)
        MTLSize threadgroupCount = MTLSizeMake(((NSUInteger)n + 7) / 8, ((NSUInteger)m + 7) / 8, 1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_matmul] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c, [bufC contents], bytesC);
        return 1;
    }
}

int metal_matmul2(const char* kernel_src,
                   const float* a1, const float* b1, float* c1, int m1, int k1, int n1,
                   const float* a2, const float* b2, float* c2, int m2, int k2, int n2) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;

        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul2_kernel");
        if (!pipeline) return 0;

        size_t bytesA1 = (size_t)m1 * (size_t)k1 * sizeof(float);
        size_t bytesB1 = (size_t)k1 * (size_t)n1 * sizeof(float);
        size_t bytesC1 = (size_t)m1 * (size_t)n1 * sizeof(float);
        size_t bytesA2 = (size_t)m2 * (size_t)k2 * sizeof(float);
        size_t bytesB2 = (size_t)k2 * (size_t)n2 * sizeof(float);
        size_t bytesC2 = (size_t)m2 * (size_t)n2 * sizeof(float);

        id<MTLBuffer> bufA1 = [gDevice newBufferWithBytes:a1 length:bytesA1 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB1 = [gDevice newBufferWithBytes:b1 length:bytesB1 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC1 = [gDevice newBufferWithLength:bytesC1 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufA2 = [gDevice newBufferWithBytes:a2 length:bytesA2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB2 = [gDevice newBufferWithBytes:b2 length:bytesB2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC2 = [gDevice newBufferWithLength:bytesC2 options:MTLResourceStorageModeShared];
        if (!bufA1 || !bufB1 || !bufC1 || !bufA2 || !bufB2 || !bufC2) {
            NSLog(@"[metal_matmul2] tao buffer FAILED"); return 0;
        }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:bufA1 offset:0 atIndex:0];
        [encoder setBuffer:bufB1 offset:0 atIndex:1];
        [encoder setBuffer:bufC1 offset:0 atIndex:2];
        int32_t dims1[3] = {m1, k1, n1};
        [encoder setBytes:dims1 length:sizeof(dims1) atIndex:3];

        [encoder setBuffer:bufA2 offset:0 atIndex:4];
        [encoder setBuffer:bufB2 offset:0 atIndex:5];
        [encoder setBuffer:bufC2 offset:0 atIndex:6];
        int32_t dims2[3] = {m2, k2, n2};
        [encoder setBytes:dims2 length:sizeof(dims2) atIndex:7];

        int max_m = (m1 > m2) ? m1 : m2;
        int max_n = (n1 > n2) ? n1 : n2;

        MTLSize gridSize = MTLSizeMake((NSUInteger)max_n, (NSUInteger)max_m, 1);
        NSUInteger tw = MIN((NSUInteger)16, (NSUInteger)max_n); if (tw == 0) tw = 1;
        NSUInteger th = MIN((NSUInteger)16, (NSUInteger)max_m); if (th == 0) th = 1;
        MTLSize groupSize = MTLSizeMake(tw, th, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_matmul2] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c1, [bufC1 contents], bytesC1);
        memcpy(c2, [bufC2 contents], bytesC2);
        return 1;
    }
}

int metal_matmul_q4(const char* kernel_src, const float* a, const unsigned char* wq,
                     const float* scales, const float* zeros, float* c,
                     int m, int k, int n, int groupSize, int nGroupsPerRow) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;

        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul_q4_kernel");
        if (!pipeline) return 0;

        size_t bytesPerRow = (size_t)((k + 1) / 2);
        size_t bytesA = (size_t)m * (size_t)k * sizeof(float);
        size_t bytesWq = (size_t)n * bytesPerRow;
        size_t bytesScales = (size_t)n * (size_t)nGroupsPerRow * sizeof(float);
        size_t bytesC = (size_t)m * (size_t)n * sizeof(float);

        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufWq = [gDevice newBufferWithBytes:wq length:bytesWq options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScales = [gDevice newBufferWithBytes:scales length:bytesScales options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufZeros = [gDevice newBufferWithBytes:zeros length:bytesScales options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufWq || !bufScales || !bufZeros || !bufC) {
            NSLog(@"[metal_matmul_q4] tao buffer FAILED");
            return 0;
        }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufWq offset:0 atIndex:1];
        [encoder setBuffer:bufScales offset:0 atIndex:2];
        [encoder setBuffer:bufZeros offset:0 atIndex:3];
        [encoder setBuffer:bufC offset:0 atIndex:4];
        [encoder setBytes:&m length:sizeof(int) atIndex:5];
        [encoder setBytes:&k length:sizeof(int) atIndex:6];
        [encoder setBytes:&n length:sizeof(int) atIndex:7];
        [encoder setBytes:&groupSize length:sizeof(int) atIndex:8];
        [encoder setBytes:&nGroupsPerRow length:sizeof(int) atIndex:9];

        // Khong dung threadgroup memory (khong tile) - moi thread tu doc
        // het 1 hang K phan tu tu wq, don gian/dung truoc, toi uu tile sau.
        NSUInteger tgW = MIN((NSUInteger)16, (NSUInteger)n);
        NSUInteger tgH = MIN((NSUInteger)16, (NSUInteger)m);
        if (tgW == 0) tgW = 1;
        if (tgH == 0) tgH = 1;
        MTLSize groupSz = MTLSizeMake(tgW, tgH, 1);
        MTLSize gridCount = MTLSizeMake(((NSUInteger)n + tgW - 1) / tgW, ((NSUInteger)m + tgH - 1) / tgH, 1);
        [encoder dispatchThreadgroups:gridCount threadsPerThreadgroup:groupSz];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_matmul_q4] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c, [bufC contents], bytesC);
        return 1;
    }
}

// SỬA BUG NGHIÊM TRỌNG: hàm này TRƯỚC ĐÂY thiếu hẳn dòng khai báo signature
// (chỉ còn "@autoreleasepool {" trơ trọi, không nằm trong bất kỳ hàm C nào)
// -> file .m này không compile được bằng clang, hoặc tệ hơn là bị gộp nhầm
// vào thân hàm metal_matmul_q4() phía trên tuỳ trình biên dịch/preprocessor,
// khiến metal_activation() (dùng cho sigmoid trong SwiGLU FFN) không tồn tại
// hoặc chạy sai hoàn toàn -> đây nhiều khả năng chính là nguyên nhân sinh
// token rác (SwiGLU sai ngay từ sigmoid thì toàn bộ FFN mỗi layer đều sai).
int metal_activation(const char* kernel_src, int op, const float* x, float* y, int n) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"activation_kernel");
        if (!pipeline) return 0;
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufX || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufY offset:0 atIndex:1];
        [encoder setBytes:&op length:sizeof(int) atIndex:2];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytes);
        return 1;
    }
}

int metal_softmax(const char* kernel_src, const float* x, float* y, int rows, int cols) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"softmax_kernel");
        if (!pipeline) return 0;
        size_t bytes = (size_t)rows * (size_t)cols * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufX || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufY offset:0 atIndex:1];
        [encoder setBytes:&cols length:sizeof(int) atIndex:2];
        // Kernel is block-per-row (BB_WG=256 threads reduce one row via
        // threadgroup memory) -> dispatch one threadgroup per row, not one
        // thread per row. Must use dispatchThreadgroups (not dispatchThreads)
        // since the kernel reads threadgroup_position_in_grid for the row.
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytes);
        return 1;
    }
}

int metal_layernorm(const char* kernel_src, const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_kernel");
        if (!pipeline) return 0;
        size_t bytesX = (size_t)rows * (size_t)cols * sizeof(float);
        size_t bytesC = (size_t)cols * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufGamma = [gDevice newBufferWithBytes:gamma length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBeta = [gDevice newBufferWithBytes:beta length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytesX options:MTLResourceStorageModeShared];
        if (!bufX || !bufGamma || !bufBeta || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufGamma offset:0 atIndex:1];
        [encoder setBuffer:bufBeta offset:0 atIndex:2];
        [encoder setBuffer:bufY offset:0 atIndex:3];
        [encoder setBytes:&cols length:sizeof(int) atIndex:4];
        [encoder setBytes:&eps length:sizeof(float) atIndex:5];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytesX);
        return 1;
    }
}

int metal_embedding_lookup(const char* kernel_src, const float* table, const int* indices, float* y, int vocab, int dim, int num_indices) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"embedding_lookup_kernel");
        if (!pipeline) return 0;
        size_t bytesTable = (size_t)vocab * (size_t)dim * sizeof(float);
        size_t bytesIndices = (size_t)num_indices * sizeof(int);
        size_t bytesY = (size_t)num_indices * (size_t)dim * sizeof(float);
        id<MTLBuffer> bufTable = [gDevice newBufferWithBytes:table length:bytesTable options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufIndices = [gDevice newBufferWithBytes:indices length:bytesIndices options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytesY options:MTLResourceStorageModeShared];
        if (!bufTable || !bufIndices || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufTable offset:0 atIndex:0];
        [encoder setBuffer:bufIndices offset:0 atIndex:1];
        [encoder setBuffer:bufY offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_indices);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)num_indices, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytesY);
        return 1;
    }
}

int metal_activation_backward(const char* kernel_src, int op, const float* x, const float* dy, float* dx, int n) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"activation_backward_kernel");
        if (!pipeline) return 0;
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDy = [gDevice newBufferWithBytes:dy length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDx = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufX || !bufDy || !bufDx) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufDy offset:0 atIndex:1];
        [encoder setBuffer:bufDx offset:0 atIndex:2];
        [encoder setBytes:&op length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(dx, [bufDx contents], bytes);
        return 1;
    }
}

int metal_layernorm_backward(const char* kernel_src, const float* dy, const float* x,
                              const float* gamma, const float* beta,
                              float* dx, float* dgamma, float* dbeta,
                              int rows, int cols, float eps) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_backward_kernel");
        if (!pipeline) return 0;
        size_t bytesX = (size_t)rows * (size_t)cols * sizeof(float);
        size_t bytesC = (size_t)cols * sizeof(float);
        id<MTLBuffer> bufDy = [gDevice newBufferWithBytes:dy length:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufGamma = [gDevice newBufferWithBytes:gamma length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBeta = [gDevice newBufferWithBytes:beta length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDx = [gDevice newBufferWithLength:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDgamma = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDbeta = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufDy || !bufX || !bufGamma || !bufBeta || !bufDx || !bufDgamma || !bufDbeta) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufDy offset:0 atIndex:0];
        [encoder setBuffer:bufX offset:0 atIndex:1];
        [encoder setBuffer:bufGamma offset:0 atIndex:2];
        [encoder setBuffer:bufBeta offset:0 atIndex:3];
        [encoder setBuffer:bufDx offset:0 atIndex:4];
        [encoder setBuffer:bufDgamma offset:0 atIndex:5];
        [encoder setBuffer:bufDbeta offset:0 atIndex:6];
        [encoder setBytes:&rows length:sizeof(int) atIndex:7];
        [encoder setBytes:&cols length:sizeof(int) atIndex:8];
        [encoder setBytes:&eps length:sizeof(float) atIndex:9];
        // Kernel dùng 1 thread cho MỖI hàng (dx) VÀ, khi id < cols, cùng
        // thread đó gộp reduction dgamma/dbeta theo cột -> dispatch phải phủ
        // max(rows, cols).
        int gridN = rows > cols ? rows : cols;
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)gridN);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)gridN, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(dx, [bufDx contents], bytesX);
        memcpy(dgamma, [bufDgamma contents], bytesC);
        memcpy(dbeta, [bufDbeta contents], bytesC);
        return 1;
    }
}

int metal_attention_fused(const char* kernel_src, const float* q, const float* k, const float* v,
                           float* o, float* s_matrix, int B, int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        if (S > 256) { NSLog(@"[metal_attention_fused] S=%d vuot gioi han 256 cua kernel (mang scores[256] cung)", S); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_kernel");
        if (!pipeline) return 0;
        size_t bytesQKV = (size_t)B * (size_t)H * (size_t)S * (size_t)D * sizeof(float);
        size_t bytesS = (size_t)B * (size_t)H * (size_t)S * (size_t)S * sizeof(float);
        id<MTLBuffer> bufQ = [gDevice newBufferWithBytes:q length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [gDevice newBufferWithBytes:k length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [gDevice newBufferWithBytes:v length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufS = [gDevice newBufferWithLength:bytesS options:MTLResourceStorageModeShared];
        if (!bufQ || !bufK || !bufV || !bufO || !bufS) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufQ offset:0 atIndex:0];
        [encoder setBuffer:bufK offset:0 atIndex:1];
        [encoder setBuffer:bufV offset:0 atIndex:2];
        [encoder setBuffer:bufO offset:0 atIndex:3];
        [encoder setBuffer:bufS offset:0 atIndex:4];
        [encoder setBytes:&B length:sizeof(int) atIndex:5];
        [encoder setBytes:&H length:sizeof(int) atIndex:6];
        [encoder setBytes:&S length:sizeof(int) atIndex:7];
        [encoder setBytes:&D length:sizeof(int) atIndex:8];
        [encoder setBytes:&scale length:sizeof(float) atIndex:9];
        // Grid: x = bh (0..B*H-1), y = ti (0..S-1). Mỗi thread tự lo trọn 1
        // hàng causal - không cần threadgroup memory / barrier ở kernel này
        // (khác softmax/layernorm) nên threadgroup size 1 là an toàn và đơn
        // giản nhất, tránh phải tính chia hết cho grid.
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_attention_fused] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(o, [bufO contents], bytesQKV);
        memcpy(s_matrix, [bufS contents], bytesS);
        return 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// API resident (session): nhiều op mã hoá vào CÙNG 1 MTLCommandBuffer, chỉ
// commit+waitUntilCompleted MỘT LẦN ở metal_session_end(). Bám sát 100%
// idiom đã chạy được ở metal_matmul/metal_softmax phía trên (cùng cách tạo
// buffer, cùng cách lấy pipeline, cùng cách dispatch) - chỉ khác dùng
// chung 1 encoder/command buffer thay vì tạo mới mỗi lệnh.
static id<MTLCommandBuffer> gSessionCmdBuf = nil;
static id<MTLComputeCommandEncoder> gSessionEncoder = nil;

int metal_session_begin(const char* kernel_src) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        if (gSessionCmdBuf != nil) { NSLog(@"[metal_session_begin] session cu chua end, bo qua"); return 0; }
        gSessionCmdBuf = [gQueue commandBuffer];
        if (!gSessionCmdBuf) { NSLog(@"[metal_session_begin] commandBuffer tra ve nil"); return 0; }
        gSessionEncoder = [gSessionCmdBuf computeCommandEncoder];
        if (!gSessionEncoder) { NSLog(@"[metal_session_begin] computeCommandEncoder tra ve nil"); gSessionCmdBuf = nil; return 0; }
        return 1;
    }
}

MetalBufferHandle metal_upload(const float* data, int n) {
    @autoreleasepool {
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_upload] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

MetalBufferHandle metal_upload_indices(const int* data, int n) {
    @autoreleasepool {
        size_t bytes = (size_t)n * sizeof(int32_t);
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_upload_indices] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

MetalBufferHandle metal_alloc_scratch(int n) {
    @autoreleasepool {
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> buf = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_alloc_scratch] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

int metal_matmul_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_enc] chua goi metal_session_begin"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufB offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufC offset:0 atIndex:2];
        [gSessionEncoder setBytes:&m length:sizeof(int) atIndex:3];
        [gSessionEncoder setBytes:&k length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&n length:sizeof(int) atIndex:5];
        // Giống hệt metal_matmul: threadgroup CỐ ĐỊNH 16x16 (tileA/tileB[16][16]
        // trong kernel), không được co lại theo m/n.
        MTLSize groupSize = MTLSizeMake(8, 8, 1);  // SUA: khop BB_TILE=8 (fix GPU Timeout tren iGPU yeu)
        MTLSize threadgroupCount = MTLSizeMake(((NSUInteger)n + 7) / 8, ((NSUInteger)m + 7) / 8, 1);
        [gSessionEncoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:groupSize];
        // SỬA BUG THẬT: 1 encoder dùng chung cho nhiều dispatch (matmul rồi
        // softmax rồi matmul...) KHÔNG tự động đảm bảo write của dispatch
        // này visible cho read của dispatch SAU trên GPU không phải Apple
        // Silicon (vd Intel Iris) - Metal chỉ đảm bảo THỨ TỰ thực thi, không
        // đảm bảo memory visibility giữa các pipeline khác nhau nếu thiếu
        // barrier tường minh. Đây đúng là nguyên nhân "kết quả sai trên
        // driver Intel Iris" mà comment cũ nghi ngờ nhưng chưa fix - insert
        // memoryBarrierWithScope: ngay sau MỖI dispatch ghi vào buffer mà
        // dispatch kế tiếp sẽ đọc, để ép GPU hoàn tất + flush cache trước khi
        // bước tiếp theo (matmul->softmax->matmul, hoặc mul->mul->add...) đọc.
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_softmax_enc(MetalBufferHandle x, MetalBufferHandle y, int rows, int cols) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_softmax_enc] chua goi metal_session_begin"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"softmax_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:1];
        [gSessionEncoder setBytes:&cols length:sizeof(int) atIndex:2];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_activation_enc(int op, MetalBufferHandle x, MetalBufferHandle y, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_activation_enc] chua goi metal_session_begin"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"activation_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:1];
        [gSessionEncoder setBytes:&op length:sizeof(int) atIndex:2];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_vecop_enc(int op, MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_vecop_enc] chua goi metal_session_begin"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"vecop_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufB offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufC offset:0 atIndex:2];
        [gSessionEncoder setBytes:&op length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_layernorm_enc(MetalBufferHandle x, MetalBufferHandle gamma, MetalBufferHandle beta,
                         MetalBufferHandle y, int rows, int cols, float eps) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_layernorm_enc] chua goi metal_session_begin"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufGamma = (__bridge id<MTLBuffer>)gamma;
        id<MTLBuffer> bufBeta = (__bridge id<MTLBuffer>)beta;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufGamma offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufBeta offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:3];
        [gSessionEncoder setBytes:&cols length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&eps length:sizeof(float) atIndex:5];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_embedding_lookup_enc(MetalBufferHandle table, MetalBufferHandle indices,
                                MetalBufferHandle y, int vocab, int dim, int num_indices) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_embedding_lookup_enc] chua goi metal_session_begin"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"embedding_lookup_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufTable = (__bridge id<MTLBuffer>)table;
        id<MTLBuffer> bufIdx = (__bridge id<MTLBuffer>)indices;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufTable offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufIdx offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:2];
        [gSessionEncoder setBytes:&dim length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)dim);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)num_indices, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_session_end(void) {
    @autoreleasepool {
        if (!gSessionEncoder || !gSessionCmdBuf) { NSLog(@"[metal_session_end] chua co session dang mo"); return 0; }
        [gSessionEncoder endEncoding];
        [gSessionCmdBuf commit];
        [gSessionCmdBuf waitUntilCompleted];
        BOOL ok = (gSessionCmdBuf.status == MTLCommandBufferStatusCompleted);
        if (!ok) {
            NSLog(@"[metal_session_end] command buffer FAILED, status=%ld, error=%@",
                  (long)gSessionCmdBuf.status, gSessionCmdBuf.error);
        }
        gSessionEncoder = nil;
        gSessionCmdBuf = nil;
        return ok ? 1 : 0;
    }
}

int metal_buffer_read(MetalBufferHandle h, float* outData, int n) {
    @autoreleasepool {
        if (!h) return 0;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)h;
        memcpy(outData, [buf contents], (size_t)n * sizeof(float));
        return 1;
    }
}

void metal_buffer_free(MetalBufferHandle h) {
    @autoreleasepool {
        if (!h) return;
        // __bridge_transfer trả quyền sở hữu ARC lại cho biến id<MTLBuffer>
        // tạm rồi để nó tự release khi ra khỏi scope - đúng cặp với
        // __bridge_retained lúc metal_upload/metal_alloc_scratch tạo ra handle.
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)h;
        (void)buf;
    }
}

int metal_attention_fused_backward(const char* kernel_src, const float* q, const float* k, const float* v,
                                    const float* s_matrix, const float* dy,
                                    float* dq, float* dk, float* dv,
                                    int B, int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        if (S > 256) { NSLog(@"[metal_attention_fused_backward] S=%d vuot gioi han 256 cua kernel", S); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_backward_kernel");
        if (!pipeline) return 0;
        size_t bytesQKV = (size_t)B * (size_t)H * (size_t)S * (size_t)D * sizeof(float);
        size_t bytesS = (size_t)B * (size_t)H * (size_t)S * (size_t)S * sizeof(float);
        id<MTLBuffer> bufQ = [gDevice newBufferWithBytes:q length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [gDevice newBufferWithBytes:k length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [gDevice newBufferWithBytes:v length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufS = [gDevice newBufferWithBytes:s_matrix length:bytesS options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDy = [gDevice newBufferWithBytes:dy length:bytesQKV options:MTLResourceStorageModeShared];
        // dq/dk/dv: kernel dùng "+=" (dq) / atomic-add (dk, dv) -> PHẢI
        // zero-init buffer trước khi launch (khác bufO/bufS ở forward, ghi
        // đè hoàn toàn nên không cần). newBufferWithLength không đảm bảo
        // zero theo spec Metal -> memset tường minh cho chắc.
        id<MTLBuffer> bufDq = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDk = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDv = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        if (!bufQ || !bufK || !bufV || !bufS || !bufDy || !bufDq || !bufDk || !bufDv) return 0;
        memset([bufDq contents], 0, bytesQKV);
        memset([bufDk contents], 0, bytesQKV);
        memset([bufDv contents], 0, bytesQKV);
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufQ offset:0 atIndex:0];
        [encoder setBuffer:bufK offset:0 atIndex:1];
        [encoder setBuffer:bufV offset:0 atIndex:2];
        [encoder setBuffer:bufS offset:0 atIndex:3];
        [encoder setBuffer:bufDy offset:0 atIndex:4];
        [encoder setBuffer:bufDq offset:0 atIndex:5];
        [encoder setBuffer:bufDk offset:0 atIndex:6];
        [encoder setBuffer:bufDv offset:0 atIndex:7];
        [encoder setBytes:&B length:sizeof(int) atIndex:8];
        [encoder setBytes:&H length:sizeof(int) atIndex:9];
        [encoder setBytes:&S length:sizeof(int) atIndex:10];
        [encoder setBytes:&D length:sizeof(int) atIndex:11];
        [encoder setBytes:&scale length:sizeof(float) atIndex:12];
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_attention_fused_backward] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(dq, [bufDq contents], bytesQKV);
        memcpy(dk, [bufDk contents], bytesQKV);
        memcpy(dv, [bufDv contents], bytesQKV);
        return 1;
    }
}