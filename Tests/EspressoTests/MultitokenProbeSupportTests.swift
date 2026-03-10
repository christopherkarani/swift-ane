import XCTest
import CoreML
import ANETypes
@testable import Espresso

final class MultitokenProbeSupportTests: XCTestCase {
    func test_recurrent_generation_weight_store_round_trips_weights() throws {
        let weights = makeTestRecurrentWeights(layerCount: 2, sharedClassifier: false)
        let path = temporaryFilePath(named: "recurrent-weights.bin")

        try RecurrentGenerationWeightStore.save(weights, to: path)
        let loaded = try RecurrentGenerationWeightStore.load(from: path)

        XCTAssertEqual(loaded.layers.count, 2)
        XCTAssertFalse(loaded.sharedClassifier)
        XCTAssertEqual(loaded.vocabSize, ModelConfig.vocab)
        XCTAssertEqual(floats(loaded.rmsFinal, prefix: 8), floats(weights.rmsFinal, prefix: 8))
        XCTAssertEqual(floats(loaded.embedding, prefix: 16), floats(weights.embedding, prefix: 16))
        XCTAssertEqual(floats(loaded.classifier, prefix: 16), floats(weights.classifier, prefix: 16))
        XCTAssertEqual(floats(loaded.layers[0].rms, prefix: 8), floats(weights.layers[0].rms, prefix: 8))
        XCTAssertEqual(floats(loaded.layers[1].Wo, prefix: 16), floats(weights.layers[1].Wo, prefix: 16))
    }

    func test_probe_plan_requires_generation_model_for_coreml_when_recurrent_checkpoint_is_selected() throws {
        let configuration = MultitokenProbeConfiguration(
            input: .recurrentCheckpoint(path: "/tmp/recurrent.bin"),
            compareCoreML: true,
            coreMLModelPath: "benchmarks/models/transformer_6layer.mlpackage",
            generationModelPath: nil
        )

        XCTAssertThrowsError(try configuration.validated()) { error in
            XCTAssertEqual(
                error as? MultitokenProbeConfigurationError,
                .missingGenerationModelPathForCoreML
            )
        }
    }

    func test_probe_plan_uses_cpu_and_neural_engine_for_matched_coreml_runs() throws {
        let configuration = MultitokenProbeConfiguration(
            input: .echo,
            compareCoreML: true,
            coreMLModelPath: "benchmarks/models/transformer_6layer.mlpackage",
            generationModelPath: nil
        )

        let plan = try configuration.validated()

        XCTAssertEqual(plan.coreMLRequest?.computeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(plan.coreMLRequest?.headWeightsSource, .echo)
    }

    func test_probe_plan_requires_explicit_input_mode() throws {
        let configuration = MultitokenProbeConfiguration(
            input: nil,
            compareCoreML: false,
            coreMLModelPath: nil,
            generationModelPath: nil
        )

        XCTAssertThrowsError(try configuration.validated()) { error in
            XCTAssertEqual(error as? MultitokenProbeConfigurationError, .missingInput)
        }
    }

    private func makeTestRecurrentWeights(
        layerCount: Int,
        sharedClassifier: Bool
    ) -> RecurrentGenerationWeights {
        let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { layerIndex in
            let weights = RWKVStyleRecurrentWeights()
            fill(weights.rms, seed: 10_000 + layerIndex * 100)
            fill(weights.Wx, seed: 20_000 + layerIndex * 100)
            fill(weights.Ws, seed: 30_000 + layerIndex * 100)
            fill(weights.Wd, seed: 40_000 + layerIndex * 100)
            fill(weights.Wo, seed: 50_000 + layerIndex * 100)
            return weights
        }

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        fill(rmsFinal, seed: 60_000)

        let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: false)
        fill(embedding, seed: 70_000)

        let classifier = TensorBuffer(
            count: sharedClassifier ? 0 : ModelConfig.vocab * ModelConfig.dim,
            zeroed: false
        )
        if !sharedClassifier {
            fill(classifier, seed: 80_000)
        }

        return RecurrentGenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            sharedClassifier: sharedClassifier
        )
    }

    private func fill(_ buffer: borrowing TensorBuffer, seed: Int) {
        buffer.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = Float(seed + idx) * 0.001
            }
        }
    }

    private func floats(_ buffer: borrowing TensorBuffer, prefix: Int) -> [Float] {
        buffer.withUnsafeBufferPointer { ptr in
            Array(ptr.prefix(prefix))
        }
    }

    private func temporaryFilePath(named name: String) -> String {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("espresso-multitoken-tests", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent(UUID().uuidString + "-" + name).path
    }
}
