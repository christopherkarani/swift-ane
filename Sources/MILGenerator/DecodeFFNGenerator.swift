import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Decode-time FFN kernel with fused residual.
///
/// Input:
/// - `x`: `[1, dim, 1, laneSpatial]` (token packed at lane 0, remaining lanes zero)
///
/// Output:
/// - `x + ffn(x)`: `[1, dim, 1, laneSpatial]`
public struct DecodeFFNGenerator: MILProgramGenerator {
    public let laneSpatial: Int
    public let architecture: LayerWeightsArchitecture

    public init(
        laneSpatial: Int = 32,
        architecture: LayerWeightsArchitecture = .rmsNormSwiGLU
    ) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
        self.architecture = architecture
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * laneSpatial * 2] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let normalized: Int
            let y: Int

            switch architecture {
            case .rmsNormSwiGLU:
                normalized = try graph.rmsNorm(
                    "norm",
                    input: x,
                    dim: ModelConfig.dim,
                    spatial: laneSpatial,
                    eps: 0.00001,
                    weightPath: "@model_path/weights/rms2.bin"
                )
                y = try graph.swigluFFN(
                    "ffn",
                    input: normalized,
                    inDim: ModelConfig.dim,
                    hiddenDim: ModelConfig.hidden,
                    spatial: laneSpatial,
                    w1Path: "@model_path/weights/w1.bin",
                    w3Path: "@model_path/weights/w3.bin",
                    w2Path: "@model_path/weights/w2.bin"
                )
            case .gpt2:
                normalized = try graph.layerNorm(
                    "norm",
                    input: x,
                    dim: ModelConfig.dim,
                    spatial: laneSpatial,
                    eps: 0.00001,
                    gammaPath: "@model_path/weights/rms2.bin",
                    betaPath: "@model_path/weights/rms2_beta.bin"
                )
                y = try graph.ffn(
                    "ffn",
                    input: normalized,
                    inDim: ModelConfig.dim,
                    hiddenDim: ModelConfig.hidden,
                    spatial: laneSpatial,
                    w1Path: "@model_path/weights/w1.bin",
                    b1Path: "@model_path/weights/b1.bin",
                    w2Path: "@model_path/weights/w2.bin",
                    b2Path: "@model_path/weights/b2.bin",
                    activation: .gelu
                )
            }
            let out = try graph.add("out", x: x, y: y)
            try LegacyGraphSupport.setOutputs(&graph, [("out", out)])
        }
    }
}
