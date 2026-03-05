import Foundation
import CoreML
import ANETypes

enum CoreMLBench {
    struct Result {
        let results: [(label: String, result: BenchmarkResult)]
        let modelLoadTimeMs: Double
    }

    static func run(runner: BenchmarkRunner, modelPath: String) throws -> Result {
        printStderr("=== Core ML Baseline Benchmark ===")

        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: modelPath) else {
            printStderr("  ERROR: Core ML model not found at \(modelPath)")
            printStderr("  Run: python3 scripts/generate_coreml_model.py")
            throw BenchError.coreMLModelNotFound(modelPath)
        }

        // Compile .mlpackage -> .mlmodelc (required before loading)
        printStderr("  Compiling .mlpackage...")
        let compileStart = ContinuousClock.now
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let compileTimeMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compiled in %.1f ms", compileTimeMs))

        // Measure model load time
        let loadStart = ContinuousClock.now
        let configAll = MLModelConfiguration()
        configAll.computeUnits = .all
        let modelAll = try MLModel(contentsOf: compiledURL, configuration: configAll)
        let loadTimeMs = durationMs(ContinuousClock.now - loadStart)
        printStderr(String(format: "  Model loaded in %.1f ms (compute units: .all)", loadTimeMs))

        // Create input matching tensor dimensions: [1, dim, 1, seqLen]
        let inputArray = try MLMultiArray(
            shape: [1, NSNumber(value: ModelConfig.dim), 1, NSNumber(value: ModelConfig.seqLen)],
            dataType: .float16
        )
        let count = ModelConfig.dim * ModelConfig.seqLen
        for i in 0..<count {
            inputArray[i] = NSNumber(value: Float.random(in: -0.1...0.1))
        }

        // Input key "x" matches coremltools function parameter name
        let featureProvider = try MLDictionaryFeatureProvider(
            dictionary: ["x": MLFeatureValue(multiArray: inputArray)]
        )

        var allResults: [(label: String, result: BenchmarkResult)] = []

        // Run with .all
        printStderr("  Running with .all compute units...")
        let resultAll = try runner.run(label: "CoreML (.all)") {
            let _ = try modelAll.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.all)", resultAll))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultAll.mean, resultAll.median))

        // Run with .cpuAndNeuralEngine
        printStderr("  Running with .cpuAndNeuralEngine...")
        let configANE = MLModelConfiguration()
        configANE.computeUnits = .cpuAndNeuralEngine
        let modelANE = try MLModel(contentsOf: compiledURL, configuration: configANE)
        let resultANE = try runner.run(label: "CoreML (.cpuAndNeuralEngine)") {
            let _ = try modelANE.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.cpuAndNeuralEngine)", resultANE))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultANE.mean, resultANE.median))

        // Run with .cpuAndGPU
        printStderr("  Running with .cpuAndGPU...")
        let configGPU = MLModelConfiguration()
        configGPU.computeUnits = .cpuAndGPU
        let modelGPU = try MLModel(contentsOf: compiledURL, configuration: configGPU)
        let resultGPU = try runner.run(label: "CoreML (.cpuAndGPU)") {
            let _ = try modelGPU.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.cpuAndGPU)", resultGPU))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultGPU.mean, resultGPU.median))

        return Result(results: allResults, modelLoadTimeMs: loadTimeMs)
    }
}

enum BenchError: Error, CustomStringConvertible {
    case coreMLModelNotFound(String)

    var description: String {
        switch self {
        case .coreMLModelNotFound(let path):
            return "Core ML model not found at \(path). Run: python3 scripts/generate_coreml_model.py"
        }
    }
}
