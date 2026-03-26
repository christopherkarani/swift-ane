import Testing
import Foundation
@testable import ESPBundle
@testable import ESPRuntime

@Test func runtimePrefersANEWhenSupported() throws {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.1b",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "spm-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .prefill2048, .decode1],
        maxContext: 2048,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "int8", weightBits: 8, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 2,
        optimization: .init(
            recipe: "stories-ctx1024",
            qualityGate: "exact",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: nil
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    let selection = try ESPRuntimeResolver.selectBackend(
        capabilities: .init(supportsANEPrivate: true),
        manifest: manifest
    )

    #expect(selection.backend == .anePrivate)
    #expect(selection.profile == .prefill2048)
    #expect(selection.contextTargetTokens == 1024)
}

@Test func runtimeFallsBackToCPUWhenANEUnavailable() throws {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.gpt2.124m",
        modelFamily: .gpt2,
        architectureVersion: "decoder-v1",
        tokenizerContract: "gpt2-bpe-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 1024,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .compat,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "baseline",
            qualityGate: "exact",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: nil
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    let selection = try ESPRuntimeResolver.selectBackend(
        capabilities: .init(supportsANEPrivate: false),
        manifest: manifest
    )

    #expect(selection.backend == .cpuSafe)
    #expect(selection.profile == .prefill256)
}

@Test func runtimeRejectsBundlesWithoutCompatibleBackends() {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.qwen.0_6b",
        modelFamily: .qwen,
        architectureVersion: "decoder-v1",
        tokenizerContract: "qwen-bpe-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 2048,
        contextTargetTokens: 512,
        compressionPolicy: .init(name: "int4", weightBits: 4, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .nearExact,
        adapterSlots: 4,
        optimization: .init(
            recipe: "qwen-gqa",
            qualityGate: "short-long-prompt-parity",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: nil
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    do {
        _ = try ESPRuntimeResolver.selectBackend(
            capabilities: .init(supportsANEPrivate: false),
            manifest: manifest
        )
        #expect(Bool(false), "Expected backend selection failure")
    } catch let error as ESPRuntimeSelectionError {
        #expect(error == .noCompatibleBackend)
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func runtimeRejectsRequestedContextBeyondTarget() {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.stories.ctx256",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "native-ane-fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "stories-ctx256",
            qualityGate: "exact",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: nil
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    do {
        _ = try ESPRuntimeResolver.selectBackend(
            capabilities: .init(supportsANEPrivate: true),
            manifest: manifest,
            requestedContextTokens: 300
        )
        Issue.record("Expected context target rejection")
    } catch let error as ESPRuntimeSelectionError {
        #expect(error == .requestedContextExceedsTarget(requested: 300, supported: 256))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test func runtimeOpensBundleAndResolvesANE() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weights = root.appendingPathComponent("weights-src", isDirectory: true)
    let tokenizer = root.appendingPathComponent("tokenizer-src", isDirectory: true)
    let bundleURL = root.appendingPathComponent("model.esp", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    try """
    {
      "name": "qwen3",
      "nLayer": 28,
      "nHead": 16,
      "nKVHead": 8,
      "dModel": 1024,
      "headDim": 128,
      "hiddenDim": 3072,
      "vocab": 151936,
      "maxSeq": 4096,
      "normEps": 0.000001,
      "ropeTheta": 10000,
      "eosToken": 151643,
      "architecture": "llama"
    }
    """.write(to: weights.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
    try Data("weights".utf8).write(to: weights.appendingPathComponent("lm_head.bin"))
    try Data("tokenizer".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.model"))

    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.qwen.test",
        modelFamily: .qwen,
        architectureVersion: "decoder-v1",
        tokenizerContract: "qwen-bpe-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .prefill2048, .decode1],
        maxContext: 4096,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "qwen-ctx1024",
            qualityGate: "exact",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: nil
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )
    _ = try ESPBundleArchive.create(
        at: bundleURL,
        manifest: manifest,
        weightsDirectory: weights,
        tokenizerDirectory: tokenizer
    )

    let bundle = try ESPRuntimeBundle.open(at: bundleURL)
    let selection = try ESPRuntimeRunner.resolve(bundle: bundle)
    #expect(bundle.config.name == "qwen3")
    #expect(selection.backend == .anePrivate)
    #expect(selection.contextTargetTokens == 1024)
}
