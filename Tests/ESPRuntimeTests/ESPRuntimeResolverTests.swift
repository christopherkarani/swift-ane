import Testing
import Foundation
@testable import ESPBundle
@testable import ESPRuntime

@Test func runtimePrefersANEWhenSupported() throws {
    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "espresso.llama.1b",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "spm-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.decode1],
        maxContext: 2048,
        compressionPolicy: .init(name: "int8", weightBits: 8, activationBits: nil),
        adapterSlots: 2,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    let selection = try ESPRuntimeResolver.selectBackend(
        capabilities: .init(supportsANEPrivate: true),
        manifest: manifest
    )

    #expect(selection.backend == .anePrivate)
}

@Test func runtimeFallsBackToCPUWhenANEUnavailable() throws {
    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "espresso.gpt2.124m",
        modelFamily: .gpt2,
        architectureVersion: "decoder-v1",
        tokenizerContract: "gpt2-bpe-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.decode1],
        maxContext: 1024,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    let selection = try ESPRuntimeResolver.selectBackend(
        capabilities: .init(supportsANEPrivate: false),
        manifest: manifest
    )

    #expect(selection.backend == .cpuSafe)
}

@Test func runtimeRejectsBundlesWithoutCompatibleBackends() {
    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "espresso.qwen.0_6b",
        modelFamily: .qwen,
        architectureVersion: "decoder-v1",
        tokenizerContract: "qwen-bpe-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.decode1],
        maxContext: 2048,
        compressionPolicy: .init(name: "int4", weightBits: 4, activationBits: nil),
        adapterSlots: 4,
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
        formatVersion: "1.0.0",
        modelID: "espresso.qwen.test",
        modelFamily: .qwen,
        architectureVersion: "decoder-v1",
        tokenizerContract: "qwen-bpe-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.decode1],
        maxContext: 4096,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
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
}
