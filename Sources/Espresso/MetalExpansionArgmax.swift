import Foundation
import Metal
import MetalPerformanceShaders
import IOSurface
import ANETypes

public enum MetalExpansionArgmaxError: Error, Equatable {
    case metalUnavailable
    case commandQueueUnavailable
    case libraryBuildFailed(String)
    case pipelineBuildFailed(String)
    case bufferAllocationFailed
    case surfaceLockFailed(Int32)
    case bufferBindingFailed
    case commandBufferUnavailable
    case commandEncoderUnavailable
    case commandExecutionFailed(String)
    case invalidArguments(String)
}

/// MPS-based expansion [bneck→vocab] via optimized matmul + fused GPU argmax.
///
/// Two-pass design:
///  1. MPSMatrixMultiplication: proj^T [spatial, bneck] × w^T [bneck, vocab] → logits [spatial, vocab] fp16
///  2. Custom Metal kernel: row-wise argmax reduction → uint16 token IDs
///
/// Input: IOSurface [1, bneck, 1, spatial] fp16 channel-first (ANE native layout).
/// Output: uint16 token IDs, one per spatial lane.
public final class MPSExpansionArgmax {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let matmul: MPSMatrixMultiplication
    private let argmaxPipeline: MTLComputePipelineState
    private let projBuffer: MTLBuffer
    private let wBuffer: MTLBuffer
    private let logitsBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer

    public let bottleneck: Int
    public let vocabSize: Int
    public let spatial: Int

    public init(
        wExpand: UnsafeBufferPointer<Float16>,
        bottleneck: Int,
        vocabSize: Int,
        spatial: Int
    ) throws(MetalExpansionArgmaxError) {
        guard bottleneck > 0, vocabSize > 0, spatial > 0 else {
            throw .invalidArguments("all dimensions must be > 0")
        }
        guard wExpand.count >= vocabSize * bottleneck else {
            throw .invalidArguments("wExpand too small")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw .commandQueueUnavailable
        }

        // MPS matmul: proj^T [spatial, bneck] × w^T [bneck, vocab] → [spatial, vocab]
        let mpsMatmul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: true,
            transposeRight: true,
            resultRows: spatial,
            resultColumns: vocabSize,
            interiorColumns: bottleneck,
            alpha: 1.0,
            beta: 0.0)

        // Argmax kernel
        let argmaxLib: MTLLibrary
        do {
            argmaxLib = try device.makeLibrary(source: Self.argmaxShaderSource, options: nil)
        } catch {
            throw .libraryBuildFailed(String(describing: error))
        }
        guard let argmaxFn = argmaxLib.makeFunction(name: "row_argmax_fp16") else {
            throw .libraryBuildFailed("missing row_argmax_fp16")
        }
        let pso: MTLComputePipelineState
        do {
            pso = try device.makeComputePipelineState(function: argmaxFn)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }

        // Buffers
        let projSize = bottleneck * spatial * MemoryLayout<Float16>.stride
        guard let projBuf = device.makeBuffer(length: projSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let wSize = vocabSize * bottleneck * MemoryLayout<Float16>.stride
        guard let wBuf = device.makeBuffer(bytes: wExpand.baseAddress!, length: wSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let logitsSize = spatial * vocabSize * MemoryLayout<Float16>.stride
        guard let logBuf = device.makeBuffer(length: logitsSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let outSize = spatial * MemoryLayout<UInt16>.stride
        guard let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }

        self.device = device
        self.commandQueue = commandQueue
        self.matmul = mpsMatmul
        self.argmaxPipeline = pso
        self.projBuffer = projBuf
        self.wBuffer = wBuf
        self.logitsBuffer = logBuf
        self.outputBuffer = outBuf
        self.bottleneck = bottleneck
        self.vocabSize = vocabSize
        self.spatial = spatial
    }

    /// Run MPS matmul + GPU argmax on projected ANE output surface.
    public func run(projectedSurface: IOSurfaceRef) throws(MetalExpansionArgmaxError) -> UnsafeBufferPointer<UInt16> {
        let elementCount = bottleneck * spatial
        let length = elementCount * MemoryLayout<Float16>.stride

        // Copy IOSurface data into Metal buffer
        let status = IOSurfaceLock(projectedSurface, [.readOnly], nil)
        guard status == 0 else { throw .surfaceLockFailed(status) }
        memcpy(projBuffer.contents(), IOSurfaceGetBaseAddress(projectedSurface), length)
        IOSurfaceUnlock(projectedSurface, [.readOnly], nil)

        // MPS matrix descriptors
        let projDesc = MPSMatrixDescriptor(
            rows: bottleneck, columns: spatial,
            rowBytes: spatial * MemoryLayout<Float16>.stride,
            dataType: .float16)
        let wDesc = MPSMatrixDescriptor(
            rows: vocabSize, columns: bottleneck,
            rowBytes: bottleneck * MemoryLayout<Float16>.stride,
            dataType: .float16)
        let outDesc = MPSMatrixDescriptor(
            rows: spatial, columns: vocabSize,
            rowBytes: vocabSize * MemoryLayout<Float16>.stride,
            dataType: .float16)

        let projMat = MPSMatrix(buffer: projBuffer, descriptor: projDesc)
        let wMat = MPSMatrix(buffer: wBuffer, descriptor: wDesc)
        let logitsMat = MPSMatrix(buffer: logitsBuffer, descriptor: outDesc)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        // Pass 1: MPS matmul
        matmul.encode(commandBuffer: commandBuffer, leftMatrix: projMat, rightMatrix: wMat, resultMatrix: logitsMat)

        // Pass 2: row-wise argmax
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(argmaxPipeline)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var vocabU32 = UInt32(vocabSize)
        encoder.setBytes(&vocabU32, length: 4, index: 2)
        // One threadgroup per row, 256 threads per group
        let tpg = min(256, Int(argmaxPipeline.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreadgroups(
            MTLSize(width: spatial, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial)
    }

    // MARK: - GPU Argmax Shader

    private static let argmaxShaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    /// Row-wise argmax over [rows, vocab] fp16 matrix.
    /// One threadgroup per row, cooperative reduction within the group.
    kernel void row_argmax_fp16(
        const device half *logits [[buffer(0)]],
        device ushort *tokenIds [[buffer(1)]],
        constant uint &vocab [[buffer(2)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]]
    ) {
        const device half *rowPtr = logits + row * vocab;

        // Each thread finds local max over its portion
        float localMax = -INFINITY;
        uint localIdx = 0;

        for (uint v = tid; v < vocab; v += tpg) {
            float val = float(rowPtr[v]);
            if (val > localMax) {
                localMax = val;
                localIdx = v;
            }
        }

        // SIMD reduction within each simd_group
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float otherVal = simd_shuffle_down(localMax, offset);
            uint otherIdx = simd_shuffle_down(localIdx, offset);
            if (otherVal > localMax) {
                localMax = otherVal;
                localIdx = otherIdx;
            }
        }

        // Threadgroup reduction: lane 0 of each simd_group writes to shared mem
        threadgroup float sharedMax[8];  // up to 256 threads / 32 = 8 simd groups
        threadgroup uint sharedIdx[8];

        uint simdGroupId = tid / 32;
        uint laneInSimd = tid % 32;

        if (laneInSimd == 0) {
            sharedMax[simdGroupId] = localMax;
            sharedIdx[simdGroupId] = localIdx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Final reduction by thread 0
        if (tid == 0) {
            float bestVal = sharedMax[0];
            uint bestIdx = sharedIdx[0];
            uint numSimdGroups = (tpg + 31) / 32;
            for (uint s = 1; s < numSimdGroups; s++) {
                if (sharedMax[s] > bestVal) {
                    bestVal = sharedMax[s];
                    bestIdx = sharedIdx[s];
                }
            }
            tokenIds[row] = ushort(bestIdx);
        }
    }
    """
}

/// Metal GPU expansion [bneck→vocab] + argmax on projected ANE output (single-pass, legacy).
///
/// Input: IOSurface [1, bneck, 1, spatial] fp16 channel-first (ANE native layout).
/// Output: uint16 token IDs, one per spatial lane.
///
/// Single-pass design: one thread per spatial lane. Adjacent SIMD threads read
/// adjacent proj elements (perfect coalescing). All threads read the same weight
/// row (broadcast). Weights stay in GPU L2 after first SIMD group.
public final class MetalExpansionArgmax {
    private struct Params {
        var vocab: UInt32
        var spatial: UInt32
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let wExpandBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer
    /// Pre-allocated shared input buffer for the pre-bound path (avoids per-call IOSurface lock + makeBuffer).
    public let inputBuffer: MTLBuffer
    private let params: Params

    public let bottleneck: Int
    public let vocabSize: Int
    public let spatial: Int

    public init(
        wExpand: UnsafeBufferPointer<Float16>,
        bottleneck: Int,
        vocabSize: Int,
        spatial: Int,
        threadsPerGroup: Int = 256
    ) throws(MetalExpansionArgmaxError) {
        guard bottleneck > 0, vocabSize > 0, spatial > 0 else {
            throw .invalidArguments("all dimensions must be > 0")
        }
        guard bottleneck <= 64 else {
            throw .invalidArguments("bottleneck must be <= 64")
        }
        guard wExpand.count >= vocabSize * bottleneck else {
            throw .invalidArguments("wExpand too small")
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw .commandQueueUnavailable
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource(bneck: bottleneck), options: nil)
        } catch {
            throw .libraryBuildFailed(String(describing: error))
        }

        guard let fn = library.makeFunction(name: "expansion_argmax") else {
            throw .libraryBuildFailed("missing expansion_argmax")
        }

        let pso: MTLComputePipelineState
        do {
            pso = try device.makeComputePipelineState(function: fn)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }

        let expandSize = vocabSize * bottleneck * MemoryLayout<Float16>.stride
        guard let expandBuf = device.makeBuffer(
            bytes: wExpand.baseAddress!, length: expandSize, options: .storageModeShared
        ) else { throw .bufferAllocationFailed }

        let outputSize = spatial * MemoryLayout<UInt16>.stride
        guard let outBuf = device.makeBuffer(
            length: outputSize, options: .storageModeShared
        ) else { throw .bufferAllocationFailed }

        let inputSize = bottleneck * spatial * MemoryLayout<Float16>.stride
        guard let inBuf = device.makeBuffer(
            length: inputSize, options: .storageModeShared
        ) else { throw .bufferAllocationFailed }

        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pso
        self.wExpandBuffer = expandBuf
        self.outputBuffer = outBuf
        self.inputBuffer = inBuf
        self.bottleneck = bottleneck
        self.vocabSize = vocabSize
        self.spatial = spatial
        self.params = Params(
            vocab: UInt32(vocabSize),
            spatial: UInt32(spatial)
        )
    }

    /// Run expansion+argmax on projected ANE output surface.
    /// Returns pointer to internal output buffer containing uint16 token IDs.
    public func run(projectedSurface: IOSurfaceRef) throws(MetalExpansionArgmaxError) -> UnsafeBufferPointer<UInt16> {
        let elementCount = bottleneck * spatial
        let length = elementCount * MemoryLayout<Float16>.stride

        let status = IOSurfaceLock(projectedSurface, [.readOnly], nil)
        guard status == 0 else { throw .surfaceLockFailed(status) }
        let baseAddress = IOSurfaceGetBaseAddress(projectedSurface)
        guard let inputBuffer = device.makeBuffer(
            bytesNoCopy: baseAddress,
            length: length,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw .bufferBindingFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw .commandBufferUnavailable
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(wExpandBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var p = params
        withUnsafeBytes(of: &p) { rawBytes in
            encoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        let tpg = min(spatial, Int(pipeline.maxTotalThreadsPerThreadgroup))
        let groups = (spatial + tpg - 1) / tpg
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        IOSurfaceUnlock(projectedSurface, [.readOnly], nil)

        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)"
            )
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial)
    }

    /// Run expansion+argmax from pre-allocated inputBuffer (caller must fill it first).
    /// Avoids IOSurface lock and per-call buffer creation overhead.
    public func runPreBound() throws(MetalExpansionArgmaxError) -> UnsafeBufferPointer<UInt16> {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(wExpandBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var p = params
        withUnsafeBytes(of: &p) { rawBytes in
            encoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        let tpg = min(spatial, Int(pipeline.maxTotalThreadsPerThreadgroup))
        let groups = (spatial + tpg - 1) / tpg
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)"
            )
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial)
    }

    // MARK: - Metal Shader

    private static func shaderSource(bneck: Int) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        #define BNECK \(bneck)

        struct ExpParams {
            uint vocab;
            uint spatial;
        };

        /// Single-pass expansion [bneck → vocab] + fused argmax.
        ///
        /// BNECK is injected as a compile-time constant so the Metal compiler can:
        ///  1. Fully unroll the inner k-loop
        ///  2. Allocate projLocal in registers (not thread-private device memory)
        ///  3. Use fused multiply-add instructions
        ///
        /// One thread per spatial lane. Adjacent SIMD threads read adjacent
        /// proj elements (coalesced). All threads read the same weight row (broadcast).
        /// The 4MB weight tensor stays in GPU L2 after the first SIMD group.
        ///
        /// proj layout: channel-first [1, bneck, 1, spatial] → proj[k * spatial + lane]
        /// wExpand layout: row-major [vocab, bneck] → wExpand[v * bneck + k]
        kernel void expansion_argmax(
            const device half *proj [[buffer(0)]],
            const device half *wExpand [[buffer(1)]],
            device ushort *tokenIds [[buffer(2)]],
            constant ExpParams &params [[buffer(3)]],
            uint lane [[thread_position_in_grid]]
        ) {
            if (lane >= params.spatial) return;

            uint vocab = params.vocab;
            uint spatial = params.spatial;

            // Preload projection vector for this lane into registers.
            // BNECK is compile-time constant → compiler allocates registers, not device memory.
            float projLocal[BNECK];
            for (uint k = 0; k < BNECK; k++) {
                projLocal[k] = float(proj[k * spatial + lane]);
            }

            float bestVal = -INFINITY;
            uint bestIdx = 0;

            for (uint v = 0; v < vocab; v++) {
                float dot = 0.0f;
                const device half *wRow = wExpand + v * BNECK;
                for (uint k = 0; k < BNECK; k++) {
                    dot = fma(projLocal[k], float(wRow[k]), dot);
                }
                if (dot > bestVal) {
                    bestVal = dot;
                    bestIdx = v;
                }
            }

            tokenIds[lane] = ushort(bestIdx);
        }
        """
    }
}
