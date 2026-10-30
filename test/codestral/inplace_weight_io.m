#import <Metal/Metal.h>
#include <string.h>

// Packed weights use Shared storage; their tiny decode LUT can use Private.
// Call only while the GPU is idle.
// memcpy preserves every packed bit, including bit patterns that encode FP32 NaNs.
int codestral_write_shared_bytes(void *handle, const void *data,
                                size_t offset, size_t count) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    if (!buffer || offset > buffer.length || count > buffer.length - offset) return 0;
    if (!count) return 1;
    if (buffer.storageMode == MTLStorageModeShared) {
        memcpy((char *)buffer.contents + offset, data, count);
        return 1;
    }
    @autoreleasepool {
        id<MTLBuffer> staging = [buffer.device newBufferWithBytes:data length:count
            options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [buffer.device newCommandQueue];
        if (!staging || !queue) return 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!command || !blit) return 0;
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:buffer destinationOffset:offset size:count];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return command.status == MTLCommandBufferStatusCompleted;
    }
    return 1;
}
