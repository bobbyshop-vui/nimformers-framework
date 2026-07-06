// metal_bridge.h
// API C generic để Nim gọi vào Metal (qua metal_bridge.m, Objective-C).
// KHÔNG chứa logic Nim/Python — chỉ là header C thuần, an toàn để
// #include từ cả .m (Objective-C) và làm header cho {.importc.} bên Nim.
//
// Quy ước bộ nhớ: các hàm mtl_create_device/mtl_create_queue/mtl_new_buffer/
// mtl_compile_library/mtl_get_pipeline/mtl_command_buffer_create/
// mtl_encoder_create trả về con trỏ đã "retain" (CFBridgingRetain /
// __bridge_retained) ở phía metal_bridge.m — dùng mtl_release() để giải
// phóng khi không cần nữa (xem chi tiết trong metal_bridge.m).

#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* MTLDeviceRef;
typedef void* MTLQueueRef;
typedef void* MTLBufferRef;
typedef void* MTLLibraryRef;
typedef void* MTLPipelineRef;
typedef void* MTLCmdBufRef;
typedef void* MTLEncoderRef;

// ── Device / queue / buffer ─────────────────────────────────────────
MTLDeviceRef mtl_create_device(void);
MTLQueueRef  mtl_create_queue(MTLDeviceRef device);
MTLBufferRef mtl_new_buffer(MTLDeviceRef device, size_t length);
void*        mtl_buffer_contents(MTLBufferRef buffer);
size_t       mtl_buffer_length(MTLBufferRef buffer);

// ── Compile kernel source -> library -> pipeline ────────────────────
MTLLibraryRef  mtl_compile_library(MTLDeviceRef device, const char* source, char** errorOut);
void           mtl_free_cstr(char* s);
MTLPipelineRef mtl_get_pipeline(MTLDeviceRef device, MTLLibraryRef library, const char* fnName);

// ── Dispatch đơn (tự tạo command buffer + encoder, commit + wait) ───
void mtl_dispatch(MTLQueueRef queue, MTLPipelineRef pipeline,
                   MTLBufferRef* buffers, int nbuffers,
                   size_t gx, size_t gy, size_t gz,
                   size_t tx, size_t ty, size_t tz);

// ── API gộp nhiều dispatch vào 1 command buffer / 1 encoder ─────────
// Cho phép nhiều lệnh dispatch chia sẻ cùng 1 encoder/command buffer,
// chỉ commit + wait một lần ở cuối, thay vì mỗi dispatch một round-trip
// CPU<->GPU riêng như mtl_dispatch().
MTLCmdBufRef  mtl_command_buffer_create(MTLQueueRef queue);
MTLEncoderRef mtl_encoder_create(MTLCmdBufRef cmdBuf);
void          mtl_encoder_dispatch(MTLEncoderRef encoder, MTLPipelineRef pipeline,
                                    MTLBufferRef* buffers, int nbuffers,
                                    size_t gx, size_t gy, size_t gz,
                                    size_t tx, size_t ty, size_t tz);
void          mtl_encoder_end(MTLEncoderRef encoder);
void          mtl_command_buffer_commit_and_wait(MTLCmdBufRef cmdBuf);

// ── Giải phóng object đã retain thủ công ────────────────────────────
void mtl_release(void* ref);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H