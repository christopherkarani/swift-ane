import Testing
import ANETypes
@testable import ModelSupport

@Test func registryLookupReturnsExpectedConfigs() throws {
    let gpt2 = try #require(ModelRegistry.config(named: "gpt2_124m"))
    #expect(gpt2.vocab == 50_257)
    #expect(gpt2.hiddenDim == 3_072)
    #expect(gpt2.architecture == .gpt2)

    let stories = try #require(ModelRegistry.config(named: "stories110m"))
    #expect(stories.vocab == 32_000)
    #expect(stories.hiddenDim == 2_048)
    #expect(stories.architecture == .llama)
}

@Test func registryContainsAllFourModels() {
    #expect(ModelRegistry.all.count == 4)
    #expect(ModelRegistry.all["smolLM_135m"]?.nKVHead == 3)
    #expect(ModelRegistry.all["tinyLlama_1_1b"]?.nHead == 32)
}

@Test func multiModelConfigCoexistsWithANETypesModelConfig() {
    let config = MultiModelConfig(
        name: "test",
        nLayer: 1,
        nHead: 1,
        nKVHead: 1,
        dModel: 64,
        headDim: 64,
        hiddenDim: 128,
        vocab: 256,
        maxSeq: 64,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(config.dModel == 64)
    #expect(ModelConfig.dim == 768)
}
