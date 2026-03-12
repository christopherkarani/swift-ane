import Foundation
import ANETypes
import MILGenerator

public struct FactoredGenerationRMSNormClassifierKernelSet: ~Copyable {
    public let rmsNormClassifier: ANEKernel
    public let vocabSize: Int
    public let bottleneck: Int
    public let laneSpatial: Int
    public let groups: Int

    public init(
        rmsFinal: borrowing TensorBuffer,
        classifierProjection: borrowing TensorBuffer,
        classifierExpansion: borrowing TensorBuffer,
        vocabSize: Int,
        bottleneck: Int = 128,
        laneSpatial: Int = 32,
        groups: Int = 1
    ) throws(ANEError) {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard bottleneck > 0 else {
            throw .invalidArguments("bottleneck must be > 0")
        }
        guard laneSpatial > 0 else {
            throw .invalidArguments("laneSpatial must be > 0")
        }
        guard groups > 0 else {
            throw .invalidArguments("groups must be > 0")
        }
        guard rmsFinal.count == ModelConfig.dim else {
            throw .invalidArguments("rmsFinal count \(rmsFinal.count) must equal dim \(ModelConfig.dim)")
        }

        let dim = ModelConfig.dim
        let projColsPerGroup = dim / groups
        let expColsPerGroup = bottleneck / groups

        // classifierProjection: full weight or grouped slice
        let projExpectedCount = bottleneck * projColsPerGroup
        guard classifierProjection.count >= projExpectedCount else {
            throw .invalidArguments(
                "classifierProjection count \(classifierProjection.count) too small for bottleneck \(bottleneck) * (dim/groups) \(projColsPerGroup)"
            )
        }
        let expExpectedCount = vocabSize * expColsPerGroup
        guard classifierExpansion.count >= expExpectedCount else {
            throw .invalidArguments(
                "classifierExpansion count \(classifierExpansion.count) too small for vocabSize \(vocabSize) * (bottleneck/groups) \(expColsPerGroup)"
            )
        }

        let generator = FactoredGenerationRMSNormClassifierGenerator(
            vocabSize: vocabSize,
            bottleneck: bottleneck,
            laneSpatial: laneSpatial,
            groups: groups
        )

        let rmsBlob = rmsFinal.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: 1, cols: dim)
        }
        let projBlob = classifierProjection.withUnsafeBufferPointer { ptr in
            if ptr.count == bottleneck * projColsPerGroup {
                return WeightBlob.build(from: ptr, rows: bottleneck, cols: projColsPerGroup)
            }
            let sliced = UnsafeBufferPointer(start: ptr.baseAddress, count: bottleneck * projColsPerGroup)
            return WeightBlob.build(from: sliced, rows: bottleneck, cols: projColsPerGroup)
        }
        let expBlob = classifierExpansion.withUnsafeBufferPointer { ptr in
            if ptr.count == vocabSize * expColsPerGroup {
                return WeightBlob.build(from: ptr, rows: vocabSize, cols: expColsPerGroup)
            }
            let sliced = UnsafeBufferPointer(start: ptr.baseAddress, count: vocabSize * expColsPerGroup)
            return WeightBlob.build(from: sliced, rows: vocabSize, cols: expColsPerGroup)
        }

        self.rmsNormClassifier = try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                (path: "@model_path/weights/cls_proj.bin", data: projBlob),
                (path: "@model_path/weights/cls_expand.bin", data: expBlob),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
        self.vocabSize = vocabSize
        self.bottleneck = bottleneck
        self.laneSpatial = laneSpatial
        self.groups = groups
    }
}
