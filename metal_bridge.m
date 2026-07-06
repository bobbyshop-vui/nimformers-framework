// metal_bridge.m
// Cài đặt Objective-C cho metal_bridge.h. Biên dịch với clang, mục tiêu macOS,
// link -framework Metal -framework Foundation.
//
// LƯU Ý VỀ BỘ NHỚ: các hàm dưới đây trả về id đã "retain" (giữ sống) qua
// CFBridgingRetain để con trỏ hợp lệ ở phía Nim. Đây là bridge đơn giản cho
// mục đích build 1 model sống suốt vòng đời chương trình (giống cách
// metal_ai.py giữ self.device/self.queue/_shader_pipelines làm global).
// Nếu cần giải phóng sớm, thêm mtl_release(ref) gọi CFBridgingRelease.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_bridge.h"

MTLDeviceRef mtl_create_device(void) {
#if TARGET_CPU_ARM64
    NSLog(@"mtl_create_device: kiến trúc arm64 (Apple Silicon)");
#elif TARGET_CPU_X86_64
    NSLog(@"mtl_create_device: kiến trúc x86_64 (Intel)");
#else
    NSLog(@"mtl_create_device: kiến trúc không xác định");
#endif

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev) {
        return (__bridge_retained void*)dev;
    }

    // Fallback: MTLCreateSystemDefaultDevice() phụ thuộc vào display/session
    // hiện tại nên có thể trả nil trong môi trường headless/không có GUI
    // session sống (kể cả trên Mac thật). MTLCopyAllDevices() là API chỉ
    // có trên Mac Intel (không có trên Apple Silicon) và enumerate TOÀN BỘ
    // GPU vật lý trong máy (kể cả integrated Intel GPU) độc lập với display
    // session, nên có thể tìm thấy device khi hàm trên thất bại.
#if TARGET_OS_OSX && !TARGET_CPU_ARM64
    NSArray<id<MTLDevice>>* allDevices = MTLCopyAllDevices();
    if (allDevices.count == 0) {
        NSLog(@"mtl_create_device: MTLCopyAllDevices() cũng không tìm thấy GPU nào.");
        return NULL;
    }
    // Ưu tiên GPU rời (discrete/low power = NO) nếu có, không thì lấy cái đầu.
    id<MTLDevice> chosen = allDevices[0];
    for (id<MTLDevice> d in allDevices) {
        if (!d.isLowPower) { chosen = d; break; }
    }
    NSLog(@"mtl_create_device: dùng fallback MTLCopyAllDevices() -> %@", chosen.name);
    return (__bridge_retained void*)chosen;
#else
    NSLog(@"mtl_create_device: MTLCreateSystemDefaultDevice() trả nil và máy này "
          @"không hỗ trợ MTLCopyAllDevices() (chỉ có trên Mac Intel).");
    return NULL;
#endif
}

MTLQueueRef mtl_create_queue(MTLDeviceRef device) {
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) return NULL;
        return (__bridge_retained void*)q;
    }
}

MTLBufferRef mtl_new_buffer(MTLDeviceRef device, size_t length) {
    // Hàm này gọi RẤT nhiều lần mỗi step train (mỗi upload/alloc buffer tạm),
    // nên đặc biệt quan trọng phải có pool ở đây để dọn temporaries của
    // newBufferWithLength/options ngay sau mỗi lần gọi.
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        id<MTLBuffer> buf = [dev newBufferWithLength:length
                                              options:MTLResourceStorageModeShared];
        if (!buf) return NULL;
        return (__bridge_retained void*)buf;
    }
}

void* mtl_buffer_contents(MTLBufferRef buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return [buf contents];
}

size_t mtl_buffer_length(MTLBufferRef buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return [buf length];
}

MTLLibraryRef mtl_compile_library(MTLDeviceRef device, const char* source, char** errorOut) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    NSString* src = [NSString stringWithUTF8String:source];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        if (errorOut) {
            NSString* msg = err ? [err localizedDescription] : @"unknown metal compile error";
            *errorOut = strdup([msg UTF8String]);
        }
        return NULL;
    }
    return (__bridge_retained void*)lib;
}

void mtl_free_cstr(char* s) {
    free(s);
}

MTLPipelineRef mtl_get_pipeline(MTLDeviceRef device, MTLLibraryRef library, const char* fnName) {
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;
        NSString* name = [NSString stringWithUTF8String:fnName];
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) return NULL;
        NSError* err = nil;
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) return NULL;
        return (__bridge_retained void*)pipe;
    }
}

void mtl_dispatch(MTLQueueRef queue, MTLPipelineRef pipeline,
                   MTLBufferRef* buffers, int nbuffers,
                   size_t gx, size_t gy, size_t gz,
                   size_t tx, size_t ty, size_t tz) {
    // QUAN TRỌNG: đây là binary Nim thuần, không phải app Cocoa, nên KHÔNG có
    // run loop nào tự drain autorelease pool. Mọi object ObjC tạm mà Metal tự
    // tạo bên trong (commandBuffer, encoder, object nội bộ khi setBuffer/
    // dispatchThreads/commit/waitUntilCompleted...) sẽ không bao giờ được
    // giải phóng nếu thiếu @autoreleasepool ở đây -> đây là nguồn RAM phình
    // dần mỗi step train. Khác với leak buffer đã fix ở tầng Nim bằng
    // releaseBufs() (object CHÚNG TA tự __bridge_retained), object ở đây do
    // chính Metal tạo ra "sau lưng", không ai retain/release thủ công được,
    // chỉ autorelease pool mới dọn được.
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;

        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        for (int i = 0; i < nbuffers; i++) {
            id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffers[i];
            [enc setBuffer:b offset:0 atIndex:i];
        }
        MTLSize gridSize = MTLSizeMake(gx, gy, gz);
        MTLSize tgSize = MTLSizeMake(tx > 0 ? tx : 1, ty > 0 ? ty : 1, tz > 0 ? tz : 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ─────────────────────────────────────────────────────────────
// API gộp nhiều dispatch vào 1 command buffer / 1 encoder — xem giải thích
// đầy đủ trong metal_bridge.h. Đây chính là mtl_dispatch() ở trên nhưng
// tách commit/wait ra khỏi từng dispatch riêng lẻ.
// ─────────────────────────────────────────────────────────────

MTLCmdBufRef mtl_command_buffer_create(MTLQueueRef queue) {
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        if (!cmd) return NULL;
        return (__bridge_retained void*)cmd;
    }
}

MTLEncoderRef mtl_encoder_create(MTLCmdBufRef cmdBuf) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmdBuf;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return NULL;
        return (__bridge_retained void*)enc;
    }
}

void mtl_encoder_dispatch(MTLEncoderRef encoder, MTLPipelineRef pipeline,
                           MTLBufferRef* buffers, int nbuffers,
                           size_t gx, size_t gy, size_t gz,
                           size_t tx, size_t ty, size_t tz) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
        id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)pipeline;
        [enc setComputePipelineState:pipe];
        for (int i = 0; i < nbuffers; i++) {
            id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffers[i];
            [enc setBuffer:b offset:0 atIndex:i];
        }
        MTLSize gridSize = MTLSizeMake(gx, gy, gz);
        MTLSize tgSize = MTLSizeMake(tx > 0 ? tx : 1, ty > 0 ? ty : 1, tz > 0 ? tz : 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    }
}

void mtl_encoder_end(MTLEncoderRef encoder) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
        [enc endEncoding];
    }
}

void mtl_command_buffer_commit_and_wait(MTLCmdBufRef cmdBuf) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmdBuf;
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ─────────────────────────────────────────────────────────────
// Giải phóng object đã "retain" thủ công lúc tạo (__bridge_retained ở các
// hàm mtl_new_buffer/mtl_command_buffer_create/mtl_encoder_create...).
// __bridge_transfer trả quyền sở hữu lại cho ARC; gán vào biến local rồi để
// nó ra khỏi scope là ARC tự release đúng 1 lần (đúng bù cho đúng 1 lần
// __bridge_retained lúc tạo) — xem giải thích đầy đủ trong metal_bridge.h.
// ─────────────────────────────────────────────────────────────
void mtl_release(void* ref) {
    if (!ref) return;
    @autoreleasepool {
        id obj = (__bridge_transfer id)ref;
        obj = nil;
    }
}