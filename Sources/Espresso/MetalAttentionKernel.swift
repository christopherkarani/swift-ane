import Foundation
import Darwin
import Metal
import IOSurface
import ANETypes

public enum MetalAttentionError: Error, Equatable {
    case invalidShape(String)
    case invalidInputCount(String)
    case metalUnavailable
    case commandQueueUnavailable
    case libraryBuildFailed(String)
    case pipelineBuildFailed(String)
    case surfaceCreateFailed
    case surfaceLockFailed(Int32)
    case surfaceBaseAddressNil
    case bufferBindingFailed
    case temporaryBufferAllocationFailed
    case commandBufferUnavailable
    case commandEncoderUnavailable
    case commandExecutionFailed(String)
}

public struct MetalAttentionShape: Sendable, Equatable {
    public let heads: Int
    public let headDim: Int
    public let seqLen: Int

    public init(heads: Int, headDim: Int, seqLen: Int) throws(MetalAttentionError) {
        guard heads > 0 else {
            throw .invalidShape("heads must be > 0")
        }
        guard headDim > 0 else {
            throw .invalidShape("headDim must be > 0")
        }
        guard seqLen > 0 else {
            throw .invalidShape("seqLen must be > 0")
        }
        self.heads = heads
        self.headDim = headDim
        self.seqLen = seqLen
    }
}

public struct MetalAttentionBenchmarkResult: Sendable, Equatable {
    public let meanMs: Double
    public let medianMs: Double
    public let warmup: Int
    public let iterations: Int
    public let zeroCopyBindings: Bool
}

public final class MetalAttentionKernel {
    private struct AttentionParams {
        var heads: UInt32
        var headDim: UInt32
        var seqLen: UInt32
        var pad0: UInt32 = 0
        var scale: Float
        var pad1: Float = 0
        var pad2: Float = 0
        var pad3: Float = 0
    }

    private final class SurfaceBinding {
        let surface: IOSurfaceRef
        let buffer: MTLBuffer

        init(surface: IOSurfaceRef, elementCount: Int, device: MTLDevice) throws(MetalAttentionError) {
            self.surface = surface
            let status = IOSurfaceLock(surface, [], nil)
            guard status == 0 else {
                throw .surfaceLockFailed(status)
            }

            let baseAddress = IOSurfaceGetBaseAddress(surface)

            let length = elementCount * MemoryLayout<UInt16>.stride
            guard let buffer = device.makeBuffer(
                bytesNoCopy: baseAddress,
                length: length,
                options: .storageModeShared,
                deallocator: nil
            ) else {
                IOSurfaceUnlock(surface, [], nil)
                throw .bufferBindingFailed
            }

            self.buffer = buffer
        }

        deinit {
            IOSurfaceUnlock(surface, [], nil)
        }
    }

    private final class RunResources {
        let qSurface: IOSurfaceRef
        let kSurface: IOSurfaceRef
        let vSurface: IOSurfaceRef
        let maskSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        let qBinding: SurfaceBinding
        let kBinding: SurfaceBinding
        let vBinding: SurfaceBinding
        let maskBinding: SurfaceBinding
        let outputBinding: SurfaceBinding
        let scoresBuffer: MTLBuffer
        let weightsBuffer: MTLBuffer

        init(device: MTLDevice, shape: MetalAttentionShape) throws(MetalAttentionError) {
            let qCount = shape.heads * shape.headDim
            let kvCount = shape.heads * shape.seqLen * shape.headDim
            let maskCount = shape.heads * shape.seqLen
            let outputCount = qCount

            qSurface = try Self.makeSurface(channels: shape.heads, spatial: shape.headDim)
            kSurface = try Self.makeSurface(channels: shape.heads * shape.seqLen, spatial: shape.headDim)
            vSurface = try Self.makeSurface(channels: shape.heads * shape.seqLen, spatial: shape.headDim)
            maskSurface = try Self.makeSurface(channels: shape.heads, spatial: shape.seqLen)
            outputSurface = try Self.makeSurface(channels: shape.heads, spatial: shape.headDim)

            qBinding = try SurfaceBinding(surface: qSurface, elementCount: qCount, device: device)
            kBinding = try SurfaceBinding(surface: kSurface, elementCount: kvCount, device: device)
            vBinding = try SurfaceBinding(surface: vSurface, elementCount: kvCount, device: device)
            maskBinding = try SurfaceBinding(surface: maskSurface, elementCount: maskCount, device: device)
            outputBinding = try SurfaceBinding(surface: outputSurface, elementCount: outputCount, device: device)

            let floatStride = MemoryLayout<Float>.stride
            guard let scoresBuffer = device.makeBuffer(
                length: maskCount * floatStride,
                options: .storageModeShared
            ) else {
                throw .temporaryBufferAllocationFailed
            }
            guard let weightsBuffer = device.makeBuffer(
                length: maskCount * floatStride,
                options: .storageModeShared
            ) else {
                throw .temporaryBufferAllocationFailed
            }

            self.scoresBuffer = scoresBuffer
            self.weightsBuffer = weightsBuffer
        }

        private static func makeSurface(channels: Int, spatial: Int) throws(MetalAttentionError) -> IOSurfaceRef {
            guard channels > 0, spatial > 0 else {
                throw .surfaceCreateFailed
            }
            let bytesPerElement = MemoryLayout<UInt16>.stride
            let bytesPerRow = spatial * bytesPerElement
            let allocSize = channels * bytesPerRow
            let properties: [CFString: Any] = [
                kIOSurfaceWidth: spatial,
                kIOSurfaceHeight: channels,
                kIOSurfaceBytesPerElement: bytesPerElement,
                kIOSurfaceBytesPerRow: bytesPerRow,
                kIOSurfaceAllocSize: allocSize,
            ]
            guard let surface = IOSurfaceCreate(properties as CFDictionary) else {
                throw .surfaceCreateFailed
            }
            return surface
        }
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let logitsPipeline: MTLComputePipelineState
    private let softmaxPipeline: MTLComputePipelineState
    private let outputPipeline: MTLComputePipelineState

    public init() throws(MetalAttentionError) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw .commandQueueUnavailable
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        } catch {
            throw .libraryBuildFailed(String(describing: error))
        }

        guard let logitsFunction = library.makeFunction(name: "attention_logits") else {
            throw .libraryBuildFailed("missing attention_logits")
        }
        guard let softmaxFunction = library.makeFunction(name: "attention_softmax") else {
            throw .libraryBuildFailed("missing attention_softmax")
        }
        guard let outputFunction = library.makeFunction(name: "attention_output") else {
            throw .libraryBuildFailed("missing attention_output")
        }

        do {
            logitsPipeline = try device.makeComputePipelineState(function: logitsFunction)
            softmaxPipeline = try device.makeComputePipelineState(function: softmaxFunction)
            outputPipeline = try device.makeComputePipelineState(function: outputFunction)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }

        self.device = device
        self.commandQueue = commandQueue
    }

    public func run(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        shape: MetalAttentionShape
    ) throws(MetalAttentionError) -> [Float] {
        try validateInputCounts(q: q, k: k, v: v, mask: mask, shape: shape)
        let resources = try RunResources(device: device, shape: shape)
        try loadInputs(q: q, k: k, v: v, mask: mask, resources: resources, shape: shape)
        try encodeAndWait(resources: resources, shape: shape)
        return readOutput(resources: resources, shape: shape)
    }

    public func benchmark(
        shape: MetalAttentionShape,
        warmup: Int,
        iterations: Int,
        seed: UInt64
    ) throws(MetalAttentionError) -> MetalAttentionBenchmarkResult {
        guard warmup >= 0 else {
            throw .invalidShape("warmup must be >= 0")
        }
        guard iterations > 0 else {
            throw .invalidShape("iterations must be > 0")
        }

        let qCount = shape.heads * shape.headDim
        let kvCount = shape.heads * shape.seqLen * shape.headDim
        let maskCount = shape.heads * shape.seqLen

        let q = Self.makeInput(count: qCount, seed: seed ^ 0x13579BDF)
        let k = Self.makeInput(count: kvCount, seed: seed ^ 0x2468ACE0)
        let v = Self.makeInput(count: kvCount, seed: seed ^ 0xDEADBEEF)
        let mask = Self.makeMask(count: maskCount)

        let resources = try RunResources(device: device, shape: shape)
        try loadInputs(q: q, k: k, v: v, mask: mask, resources: resources, shape: shape)

        for _ in 0..<warmup {
            try encodeAndWait(resources: resources, shape: shape)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = mach_absolute_time()
            try encodeAndWait(resources: resources, shape: shape)
            let end = mach_absolute_time()
            samples.append(Self.elapsedMilliseconds(start: start, end: end))
        }

        let mean = samples.reduce(0, +) / Double(samples.count)
        let median = Self.median(samples)
        return MetalAttentionBenchmarkResult(
            meanMs: mean,
            medianMs: median,
            warmup: warmup,
            iterations: iterations,
            zeroCopyBindings: true
        )
    }

    private func validateInputCounts(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        shape: MetalAttentionShape
    ) throws(MetalAttentionError) {
        let qExpected = shape.heads * shape.headDim
        let kvExpected = shape.heads * shape.seqLen * shape.headDim
        let maskExpected = shape.heads * shape.seqLen
        guard q.count == qExpected else {
            throw .invalidInputCount("q count \(q.count) != expected \(qExpected)")
        }
        guard k.count == kvExpected else {
            throw .invalidInputCount("k count \(k.count) != expected \(kvExpected)")
        }
        guard v.count == kvExpected else {
            throw .invalidInputCount("v count \(v.count) != expected \(kvExpected)")
        }
        guard mask.count == maskExpected else {
            throw .invalidInputCount("mask count \(mask.count) != expected \(maskExpected)")
        }
    }

    private func loadInputs(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        resources: RunResources,
        shape: MetalAttentionShape
    ) throws(MetalAttentionError) {
        q.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(to: resources.qSurface, data: ptr, channels: shape.heads, spatial: shape.headDim)
        }
        k.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(
                to: resources.kSurface,
                data: ptr,
                channels: shape.heads * shape.seqLen,
                spatial: shape.headDim
            )
        }
        v.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(
                to: resources.vSurface,
                data: ptr,
                channels: shape.heads * shape.seqLen,
                spatial: shape.headDim
            )
        }
        mask.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(to: resources.maskSurface, data: ptr, channels: shape.heads, spatial: shape.seqLen)
        }
    }

    private func readOutput(resources: RunResources, shape: MetalAttentionShape) -> [Float] {
        var output = [Float](repeating: 0, count: shape.heads * shape.headDim)
        output.withUnsafeMutableBufferPointer { ptr in
            SurfaceIO.readFP16(
                from: resources.outputSurface,
                into: ptr,
                channelOffset: 0,
                channels: shape.heads,
                spatial: shape.headDim
            )
        }
        return output
    }

    private func encodeAndWait(resources: RunResources, shape: MetalAttentionShape) throws(MetalAttentionError) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }
        let params = AttentionParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            seqLen: UInt32(shape.seqLen),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )

        guard let logitsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        logitsEncoder.setComputePipelineState(logitsPipeline)
        logitsEncoder.setBuffer(resources.qBinding.buffer, offset: 0, index: 0)
        logitsEncoder.setBuffer(resources.kBinding.buffer, offset: 0, index: 1)
        logitsEncoder.setBuffer(resources.maskBinding.buffer, offset: 0, index: 2)
        logitsEncoder.setBuffer(resources.scoresBuffer, offset: 0, index: 3)
        withUnsafeBytes(of: params) { rawBytes in
            logitsEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 4)
        }
        logitsEncoder.dispatchThreads(
            MTLSize(width: shape.seqLen, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(logitsPipeline.threadExecutionWidth, shape.seqLen)),
                height: 1,
                depth: 1
            )
        )
        logitsEncoder.endEncoding()

        guard let softmaxEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        softmaxEncoder.setComputePipelineState(softmaxPipeline)
        softmaxEncoder.setBuffer(resources.scoresBuffer, offset: 0, index: 0)
        softmaxEncoder.setBuffer(resources.weightsBuffer, offset: 0, index: 1)
        withUnsafeBytes(of: params) { rawBytes in
            softmaxEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 2)
        }
        softmaxEncoder.dispatchThreads(
            MTLSize(width: shape.heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(softmaxPipeline.maxTotalThreadsPerThreadgroup, shape.heads)),
                height: 1,
                depth: 1
            )
        )
        softmaxEncoder.endEncoding()

        guard let outputEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        outputEncoder.setComputePipelineState(outputPipeline)
        outputEncoder.setBuffer(resources.weightsBuffer, offset: 0, index: 0)
        outputEncoder.setBuffer(resources.vBinding.buffer, offset: 0, index: 1)
        outputEncoder.setBuffer(resources.outputBinding.buffer, offset: 0, index: 2)
        withUnsafeBytes(of: params) { rawBytes in
            outputEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        outputEncoder.dispatchThreads(
            MTLSize(width: shape.headDim, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(outputPipeline.threadExecutionWidth, shape.headDim)),
                height: 1,
                depth: 1
            )
        )
        outputEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }
    }

    private static func makeInput(count: Int, seed: UInt64) -> [Float] {
        var values = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let mixed = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407 &+ seed
            let unit = Float(mixed & 0xffff) / Float(0xffff)
            values[i] = unit * 2 - 1
        }
        return values
    }

    private static func makeMask(count: Int) -> [Float] {
        [Float](repeating: 0, count: count)
    }

    private static func median(_ values: [Double]) -> Double {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) * 0.5
        }
        return sorted[mid]
    }

    private static func elapsedMilliseconds(start: UInt64, end: UInt64) -> Double {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let elapsed = end &- start
        let nanos = elapsed &* UInt64(info.numer) / UInt64(info.denom)
        return Double(nanos) / 1_000_000.0
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct AttentionParams {
        uint heads;
        uint headDim;
        uint seqLen;
        uint pad0;
        float scale;
        float pad1;
        float pad2;
        float pad3;
    };

    kernel void attention_logits(
        const device half *q [[buffer(0)]],
        const device half *k [[buffer(1)]],
        const device half *mask [[buffer(2)]],
        device float *scores [[buffer(3)]],
        constant AttentionParams &params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint token = gid.x;
        uint head = gid.y;
        if (head >= params.heads || token >= params.seqLen) {
            return;
        }

        uint qBase = head * params.headDim;
        uint kBase = (head * params.seqLen + token) * params.headDim;
        float dot = 0.0f;
        for (uint dim = 0; dim < params.headDim; ++dim) {
            dot += float(q[qBase + dim]) * float(k[kBase + dim]);
        }

        uint maskIndex = head * params.seqLen + token;
        scores[maskIndex] = dot * params.scale + float(mask[maskIndex]);
    }

    kernel void attention_softmax(
        const device float *scores [[buffer(0)]],
        device float *weights [[buffer(1)]],
        constant AttentionParams &params [[buffer(2)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint head = gid;
        if (head >= params.heads) {
            return;
        }

        uint base = head * params.seqLen;
        float maxScore = -INFINITY;
        for (uint token = 0; token < params.seqLen; ++token) {
            maxScore = max(maxScore, scores[base + token]);
        }

        float denom = 0.0f;
        for (uint token = 0; token < params.seqLen; ++token) {
            float value = exp(scores[base + token] - maxScore);
            weights[base + token] = value;
            denom += value;
        }

        float invDenom = denom > 0.0f ? (1.0f / denom) : 0.0f;
        for (uint token = 0; token < params.seqLen; ++token) {
            weights[base + token] *= invDenom;
        }
    }

    kernel void attention_output(
        const device float *weights [[buffer(0)]],
        const device half *v [[buffer(1)]],
        device half *output [[buffer(2)]],
        constant AttentionParams &params [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint head = gid.y;
        if (head >= params.heads || dim >= params.headDim) {
            return;
        }

        float accum = 0.0f;
        uint weightBase = head * params.seqLen;
        for (uint token = 0; token < params.seqLen; ++token) {
            uint vBase = (head * params.seqLen + token) * params.headDim;
            accum += weights[weightBase + token] * float(v[vBase + dim]);
        }

        output[head * params.headDim + dim] = half(accum);
    }
    """
}
