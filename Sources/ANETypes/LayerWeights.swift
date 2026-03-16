public enum LayerWeightsArchitecture: Sendable, Equatable {
    case rmsNormSwiGLU
    case gpt2
}

public struct LayerWeights: ~Copyable {
    public let architecture: LayerWeightsArchitecture
    public let Wq: TensorBuffer
    public let Wk: TensorBuffer
    public let Wv: TensorBuffer
    public let Wo: TensorBuffer
    public let W1: TensorBuffer
    public let W2: TensorBuffer
    public let W3: TensorBuffer
    public let rmsAtt: TensorBuffer
    public let rmsFfn: TensorBuffer
    public let attentionNormBeta: TensorBuffer
    public let ffnNormBeta: TensorBuffer
    public let bq: TensorBuffer
    public let bk: TensorBuffer
    public let bv: TensorBuffer
    public let bo: TensorBuffer
    public let b1: TensorBuffer
    public let b2: TensorBuffer

    public init() {
        self.init(architecture: .rmsNormSwiGLU)
    }

    public init(architecture: LayerWeightsArchitecture) {
        self.architecture = architecture
        self.Wq = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wk = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wv = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wo = TensorBuffer(count: ModelConfig.woSize, zeroed: false)
        self.W1 = TensorBuffer(count: ModelConfig.w1Size, zeroed: false)
        self.W2 = TensorBuffer(count: ModelConfig.w2Size, zeroed: false)
        self.W3 = TensorBuffer(count: ModelConfig.w3Size, zeroed: false)
        self.rmsAtt = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.rmsFfn = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.attentionNormBeta = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.ffnNormBeta = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.bq = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.bk = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.bv = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.bo = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.b1 = TensorBuffer(count: ModelConfig.hidden, zeroed: false)
        self.b2 = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    }
}
