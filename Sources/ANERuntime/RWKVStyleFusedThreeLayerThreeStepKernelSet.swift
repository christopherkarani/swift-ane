import Foundation
import ANETypes
import MILGenerator

public struct RWKVStyleFusedThreeLayerThreeStepKernelSet: ~Copyable {
    public static let defaultLaneSpatial = 32

    internal enum KernelKind: String, CaseIterable {
        case fusedThreeLayerThreeStep
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

    public let step: ANEKernel
    public let laneSpatial: Int

    private init(step: consuming ANEKernel, laneSpatial: Int) {
        self.step = step
        self.laneSpatial = laneSpatial
    }

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = defaultLaneSpatial
    ) throws(ANEError) {
        guard laneSpatial > 0 else {
            throw .invalidArguments("fused three-layer three-step laneSpatial must be > 0")
        }
        let spec = Self.compileSpecs(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial
        )[0]
        let step = try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
        self.init(step: step, laneSpatial: laneSpatial)
    }

    internal static func compileSpecs(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int
    ) -> [CompileSpec] {
        precondition(laneSpatial > 0)
        let dim = ModelConfig.dim
        let generator = RWKVStyleFusedThreeLayerThreeStepGenerator(laneSpatial: laneSpatial)

        return [
            CompileSpec(
                kind: .fusedThreeLayerThreeStep,
                milText: generator.milText,
                weights: [
                    (path: "@model_path/weights/rwkv_rms0.bin", data: buildBlob(from: weights0.rms, rows: 1, cols: dim)),
                    (path: "@model_path/weights/wx0.bin", data: buildBlob(from: weights0.Wx, rows: dim, cols: dim)),
                    (path: "@model_path/weights/ws0.bin", data: buildBlob(from: weights0.Ws, rows: dim, cols: dim)),
                    (path: "@model_path/weights/wd0.bin", data: buildBlob(from: weights0.Wd, rows: dim, cols: dim)),
                    (path: "@model_path/weights/wo0.bin", data: buildBlob(from: weights0.Wo, rows: dim, cols: dim)),
                    (path: "@model_path/weights/rwkv_rms1.bin", data: buildBlob(from: weights1.rms, rows: 1, cols: dim)),
                    (path: "@model_path/weights/wx1.bin", data: buildBlob(from: weights1.Wx, rows: dim, cols: dim)),
                    (path: "@model_path/weights/ws1.bin", data: buildBlob(from: weights1.Ws, rows: dim, cols: dim)),
                    (path: "@model_path/weights/wd1.bin", data: buildBlob(from: weights1.Wd, rows: dim, cols: dim)),
                    (path: "@model_path/weights/wo1.bin", data: buildBlob(from: weights1.Wo, rows: dim, cols: dim)),
                    (path: "@model_path/weights/rwkv_rms2.bin", data: buildBlob(from: weights2.rms, rows: 1, cols: dim)),
                    (path: "@model_path/weights/wx2.bin", data: buildBlob(from: weights2.Wx, rows: dim, cols: dim)),
                    (path: "@model_path/weights/ws2.bin", data: buildBlob(from: weights2.Ws, rows: dim, cols: dim)),
                    (path: "@model_path/weights/wd2.bin", data: buildBlob(from: weights2.Wd, rows: dim, cols: dim)),
                    (path: "@model_path/weights/wo2.bin", data: buildBlob(from: weights2.Wo, rows: dim, cols: dim)),
                ],
                inputSizes: generator.inputByteSizes,
                outputSizes: generator.outputByteSizes
            )
        ]
    }

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }
}
