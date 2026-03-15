import Testing
import ANEGraphIR
import ANEPasses
@testable import ModelSupport

@Test func gpt2LayerGraphUsesLayerNormGeluAndBiases() throws {
    let config = ModelRegistry.gpt2_124m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/gpt2")
    let graph = try TransformerLayerGraphBuilder.forwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    #expect(graph.graphInputs.map(\.name) == ["x"])
    #expect(graph.graphOutputs.map(\.name) == ["out"])
    #expect(graph.nodes.contains { $0.name == "layer0_ln1_gamma" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_ln1_beta" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_act_tanh" && $0.op == .tanh })
    #expect(graph.nodes.contains { $0.name == "layer0_q_bias" })
    #expect(graph.nodes.contains { $0.name == "layer0_proj_bias" })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_up_bias" })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_down_bias" })
    #expect(allWeightOffsets(in: graph).allSatisfy { $0 == 128 })
    #expect(!ANEOptimizationPipeline.validate(graph).contains { $0.severity == .error })
}

@Test func llamaLayerGraphUsesRmsNormSwiGLUAndNoBiases() throws {
    let config = ModelRegistry.stories110m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/llama")
    let graph = try TransformerLayerGraphBuilder.forwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    #expect(graph.nodes.contains { $0.name == "layer0_rms1_weight" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_gate_act_sigmoid" && $0.op == .sigmoid })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_gated" && $0.op == .mul })
    #expect(!graph.nodes.contains { $0.name.contains("bias") })
    #expect(allWeightOffsets(in: graph).allSatisfy { $0 == 128 })
    #expect(!ANEOptimizationPipeline.validate(graph).contains { $0.severity == .error })
}

private func allWeightOffsets(in graph: ANEGraph) -> [UInt64] {
    graph.nodes.compactMap { node in
        guard case let .weight(_, offset) = node.attrs else {
            return nil
        }
        return offset
    }
}
