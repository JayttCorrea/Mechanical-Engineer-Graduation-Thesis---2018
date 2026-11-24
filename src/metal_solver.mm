#include "tcc/metal_solver.hpp"
#include "tcc/field_guard.hpp"
#include "metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tcc {
namespace {
std::runtime_error failure(NSString* message) {
    return std::runtime_error(message ? [message UTF8String] : "Metal operation failed");
}
}
std::vector<MetalDeviceInfo> list_metal_devices() {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d || !d.hasUnifiedMemory) return {};
        return {{0, "Apple Metal", [d.name UTF8String], true}};
    }
}
struct MetalSolver::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> pending;
    id<MTLComputePipelineState> b_kernel, p_kernel, v_kernel;
    id<MTLBuffer> u, v, p, b, un, vn, pn;
    Fields host;
    FieldGuard gu, gv, gp, gb;
    bool dirty = true;
    explicit Impl(const Config& c, const MetalOptions& options) {
        if (options.device_index > 0 || options.device_index < -1)
            throw std::runtime_error("invalid Metal device index");
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no Metal GPU available");
        if (!device.hasUnifiedMemory) throw std::runtime_error("Metal backend requires unified memory (Apple Silicon)");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("cannot create Metal command queue");
        MTLCompileOptions* compile = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            compile.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            compile.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        compile.preprocessorMacros = @{ @"CASE_TUBE": @(c.type == CaseType::Tube) };
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:metal_source] options:compile error:&error];
        if (!library) throw failure(error.localizedDescription);
        auto pipeline = [&](NSString* name) {
            id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            if (!state) throw failure(error.localizedDescription);
            return state;
        };
        b_kernel = pipeline(@"compute_b"); p_kernel = pipeline(@"pressure_iterate"); v_kernel = pipeline(@"velocity_update");
        const auto bytes = c.cells() * sizeof(float);
        for (auto* buffer : {&u, &v, &p, &b, &un, &vn, &pn}) {
            *buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!*buffer) throw std::runtime_error("Metal buffer allocation failed");
            std::memset([*buffer contents], 0, bytes);
        }
        host.assign(c.cells());
    }
    void finish() {
        if (!pending) return;
        [pending waitUntilCompleted];
        if (pending.status == MTLCommandBufferStatusError) throw failure(pending.error.localizedDescription);
        pending = nil;
    }
    void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, int nx, int ny) {
        [enc setComputePipelineState:pipeline];
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = std::min<NSUInteger>(8, pipeline.maxTotalThreadsPerThreadgroup / w);
        [enc dispatchThreads:MTLSizeMake(nx, ny, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
};
MetalSolver::MetalSolver(const Config& c, const MetalOptions& options) : Solver(c) {
    @autoreleasepool { impl_ = std::make_unique<Impl>(c, options); }
}
MetalSolver::~MetalSolver() { if (impl_->pending) [impl_->pending waitUntilCompleted]; }
std::string MetalSolver::backend_name() const { return "GPU (Metal) - " + device_name(); }
std::string MetalSolver::device_name() const { return [impl_->device.name UTF8String]; }
void MetalSolver::wait() { @autoreleasepool { impl_->finish(); } }
void MetalSolver::reset() {
    wait(); auto& m = *impl_;
    for (id<MTLBuffer> buffer : {m.u,m.v,m.p,m.b,m.un,m.vn,m.pn}) std::memset(buffer.contents,0,cfg_.cells()*sizeof(float));
    for (FieldGuard* g : {&m.gu,&m.gv,&m.gp,&m.gb}) g->clear();
    steps_=0; repaired_=0; m.dirty=true;
}
void MetalSolver::set_velocity_limit(float limit) { impl_->gu.set_limit(limit); impl_->gv.set_limit(limit); }
void MetalSolver::set_velocity(const std::vector<float>& u, const std::vector<float>& v) {
    check_velocity_size(u,v); wait();
    std::memcpy(impl_->u.contents,u.data(),u.size()*sizeof(float));
    std::memcpy(impl_->v.contents,v.data(),v.size()*sizeof(float)); impl_->dirty=true;
}
const Fields& MetalSolver::fields() {
    wait(); auto& m=*impl_;
    if (!m.dirty) return m.host;
    auto copy = [&](std::vector<float>& target,id<MTLBuffer> buffer,FieldGuard& guard) {
        std::memcpy(target.data(),buffer.contents,target.size()*sizeof(float));
        const auto n=guard.repair(target); repaired_+=n;
        if(n) std::memcpy(buffer.contents,target.data(),target.size()*sizeof(float));
    };
    copy(m.host.u,m.u,m.gu); copy(m.host.v,m.v,m.gv); copy(m.host.p,m.p,m.gp); copy(m.host.b,m.b,m.gb);
    m.dirty=false; return m.host;
}
void MetalSolver::step() {
    @autoreleasepool {
        auto& m=*impl_;
        // Bound queued work and check asynchronous errors before reusing shared storage.
        m.finish();
        id<MTLCommandBuffer> cb=[m.queue commandBuffer];
        if (!cb) throw std::runtime_error("cannot create Metal command buffer");
        if (!cfg_.warm_start_pressure) {
            id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
            [blit fillBuffer:m.p range:NSMakeRange(0,cfg_.cells()*sizeof(float)) value:0]; [blit endEncoding];
        }
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        auto buffer=[&](int i,id<MTLBuffer> value){ [enc setBuffer:value offset:0 atIndex:i]; };
        auto integer=[&](int i,int value){ [enc setBytes:&value length:sizeof(value) atIndex:i]; };
        auto scalar=[&](int i,float value){ [enc setBytes:&value length:sizeof(value) atIndex:i]; };
        buffer(0,m.b); buffer(1,m.u); buffer(2,m.v);
        integer(3,cfg_.nx); integer(4,cfg_.ny); scalar(5,cfg_.dx()); scalar(6,cfg_.dy()); scalar(7,cfg_.rho); scalar(8,cfg_.dt); scalar(9,cfg_.u0); scalar(10,cfg_.v0);
        m.dispatch(enc,m.b_kernel,cfg_.nx,cfg_.ny);
        for(int k=0;k<cfg_.pressure_iterations;++k) {
            buffer(0,m.b); buffer(1,m.p); buffer(2,m.pn);
            integer(3,cfg_.nx); integer(4,cfg_.ny); scalar(5,cfg_.dx()); scalar(6,cfg_.dy());
            m.dispatch(enc,m.p_kernel,cfg_.nx,cfg_.ny); std::swap(m.p,m.pn);
        }
        buffer(0,m.u); buffer(1,m.v); buffer(2,m.p); buffer(3,m.un); buffer(4,m.vn);
        integer(5,cfg_.nx); integer(6,cfg_.ny); scalar(7,cfg_.dx()); scalar(8,cfg_.dy()); scalar(9,cfg_.rho); scalar(10,cfg_.nu); scalar(11,cfg_.dt); scalar(12,cfg_.u0); scalar(13,cfg_.v0); integer(14,cfg_.advection==Advection::Upwind);
        m.dispatch(enc,m.v_kernel,cfg_.nx,cfg_.ny); [enc endEncoding];
        std::swap(m.u,m.un); std::swap(m.v,m.vn);
        m.pending=cb; [cb commit]; m.dirty=true; ++steps_;
    }
}
} // namespace tcc
