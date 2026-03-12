import Foundation
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes

public struct RWKVStyleFusedThreeLayerThreeStepSurfaceHandles {
    public let x0In: IOSurfaceRef
    public let x1In: IOSurfaceRef
    public let x2In: IOSurfaceRef
    public let stateIn0: IOSurfaceRef
    public let stateIn1: IOSurfaceRef
    public let stateIn2: IOSurfaceRef
    public let x0Out: IOSurfaceRef
    public let x1Out: IOSurfaceRef
    public let x2Out: IOSurfaceRef
    public let stateMid00: IOSurfaceRef
    public let stateMid10: IOSurfaceRef
    public let stateMid20: IOSurfaceRef
    public let stateMid01: IOSurfaceRef
    public let stateMid11: IOSurfaceRef
    public let stateMid21: IOSurfaceRef
    public let stateOut0: IOSurfaceRef
    public let stateOut1: IOSurfaceRef
    public let stateOut2: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleFusedThreeLayerThreeStepKernelSet) throws(ANEError) {
        self.x0In = try kernels.step.inputSurface(at: 0)
        self.x1In = try kernels.step.inputSurface(at: 1)
        self.x2In = try kernels.step.inputSurface(at: 2)
        self.stateIn0 = try kernels.step.inputSurface(at: 3)
        self.stateIn1 = try kernels.step.inputSurface(at: 4)
        self.stateIn2 = try kernels.step.inputSurface(at: 5)
        self.x0Out = try kernels.step.outputSurface(at: 0)
        self.x1Out = try kernels.step.outputSurface(at: 1)
        self.x2Out = try kernels.step.outputSurface(at: 2)
        self.stateMid00 = try kernels.step.outputSurface(at: 3)
        self.stateMid10 = try kernels.step.outputSurface(at: 4)
        self.stateMid20 = try kernels.step.outputSurface(at: 5)
        self.stateMid01 = try kernels.step.outputSurface(at: 6)
        self.stateMid11 = try kernels.step.outputSurface(at: 7)
        self.stateMid21 = try kernels.step.outputSurface(at: 8)
        self.stateOut0 = try kernels.step.outputSurface(at: 9)
        self.stateOut1 = try kernels.step.outputSurface(at: 10)
        self.stateOut2 = try kernels.step.outputSurface(at: 11)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleFusedThreeLayerThreeStepSession: ~Copyable {
    public let kernels: RWKVStyleFusedThreeLayerThreeStepKernelSet
    public let handles: RWKVStyleFusedThreeLayerThreeStepSurfaceHandles
    public private(set) var stepCount: Int
    private var hasPreparedState: Bool

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleFusedThreeLayerThreeStepKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleFusedThreeLayerThreeStepKernelSet(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial
        )
        let handles = try RWKVStyleFusedThreeLayerThreeStepSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
        self.hasPreparedState = false
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(dst: handles.x0In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.x1In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.x2In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn0, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn1, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn2, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
        } catch {
            throw .invalidArguments("fused three-layer three-step recurrent zero reset failed: \(error)")
        }
        self.stepCount = 0
        self.hasPreparedState = false
    }

    public mutating func prepare(
        tokenInput0: borrowing TensorBuffer,
        tokenInput1: borrowing TensorBuffer,
        tokenInput2: borrowing TensorBuffer,
        output0: borrowing TensorBuffer,
        output1: borrowing TensorBuffer,
        output2: borrowing TensorBuffer,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(tokenInput0.count == ModelConfig.dim)
        precondition(tokenInput1.count == ModelConfig.dim)
        precondition(tokenInput2.count == ModelConfig.dim)
        precondition(output0.count == ModelConfig.dim)
        precondition(output1.count == ModelConfig.dim)
        precondition(output2.count == ModelConfig.dim)

        var t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(dst: handles.x0In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.x1In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.x2In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try tokenInput0.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(to: handles.x0In, channelOffset: 0, spatialIndex: 0, spatial: handles.laneSpatial, data: tokenBuf, channels: ModelConfig.dim)
            }
            try tokenInput1.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(to: handles.x1In, channelOffset: 0, spatialIndex: 0, spatial: handles.laneSpatial, data: tokenBuf, channels: ModelConfig.dim)
            }
            try tokenInput2.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(to: handles.x2In, channelOffset: 0, spatialIndex: 0, spatial: handles.laneSpatial, data: tokenBuf, channels: ModelConfig.dim)
            }
        } catch {
            throw .invalidArguments("fused three-layer three-step recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("fused three-layer three-step recurrent eval failed at step \(stepCount): \(error)")
        }
        timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try output0.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(from: handles.x0Out, channelOffset: 0, spatialIndex: 0, spatial: handles.laneSpatial, into: outBuf, channels: ModelConfig.dim)
            }
            try output1.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(from: handles.x1Out, channelOffset: 0, spatialIndex: 0, spatial: handles.laneSpatial, into: outBuf, channels: ModelConfig.dim)
            }
            try output2.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(from: handles.x2Out, channelOffset: 0, spatialIndex: 0, spatial: handles.laneSpatial, into: outBuf, channels: ModelConfig.dim)
            }
        } catch {
            throw .invalidArguments("fused three-layer three-step recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.hasPreparedState = true
    }

    public mutating func promotePreparedState(commitCount: Int) throws(ANEError) {
        guard hasPreparedState else {
            throw .invalidArguments("fused three-layer three-step recurrent state promotion requested without a prepared branch")
        }
        guard (1...3).contains(commitCount) else {
            throw .invalidArguments("fused three-layer three-step recurrent promotion commitCount must be in 1...3")
        }

        let source0: IOSurfaceRef
        let source1: IOSurfaceRef
        let source2: IOSurfaceRef
        switch commitCount {
        case 1:
            source0 = handles.stateMid00
            source1 = handles.stateMid10
            source2 = handles.stateMid20
        case 2:
            source0 = handles.stateMid01
            source1 = handles.stateMid11
            source2 = handles.stateMid21
        default:
            source0 = handles.stateOut0
            source1 = handles.stateOut1
            source2 = handles.stateOut2
        }

        do {
            try SurfaceIO.copyFP16(dst: handles.stateIn0, dstChannelOffset: 0, src: source0, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn1, dstChannelOffset: 0, src: source1, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn2, dstChannelOffset: 0, src: source2, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
        } catch {
            throw .invalidArguments("fused three-layer three-step recurrent state promotion failed: \(error)")
        }

        self.stepCount += commitCount
        self.hasPreparedState = false
    }
}
