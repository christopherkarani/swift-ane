import Testing
import Foundation
@testable import ESPBundle

@Test func manifestRenderingIsDeterministic() throws {
    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "espresso.qwen.0_6b",
        modelFamily: .qwen,
        architectureVersion: "decoder-v1",
        tokenizerContract: "qwen-bpe-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .prefill2048, .decode1],
        maxContext: 2048,
        compressionPolicy: .init(name: "int4-palettized", weightBits: 4, activationBits: nil),
        adapterSlots: 4,
        accuracyBaselineRef: "benchmarks/qwen-0.6b/accuracy.json",
        performanceBaselineRef: "benchmarks/qwen-0.6b/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    let renderedA = manifest.renderTOML()
    let renderedB = manifest.renderTOML()
    #expect(renderedA == renderedB)
    #expect(renderedA.contains("model_family = \"qwen\""))
    #expect(renderedA.contains("supported_backends = [\"ane-private\", \"cpu-safe\"]"))
}

@Test func manifestValidationRejectsEmptyModelID() {
    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "spm-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.decode1],
        maxContext: 2048,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    do {
        try manifest.validate()
        #expect(Bool(false), "Expected validation failure for an empty model id")
    } catch let error as ESPBundleValidationError {
        #expect(error == .emptyField("model_id"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func bundleLayoutIncludesCanonicalTopLevelEntries() {
    #expect(ESPBundleLayout.manifestFileName == "manifest.toml")
    #expect(ESPBundleLayout.requiredTopLevelEntries == [
        "arch",
        "tokenizer",
        "weights",
        "graphs",
        "states",
        "adapters",
        "compiled",
        "benchmarks",
        "licenses",
        "signatures",
    ])
}

@Test func manifestRoundTripsThroughTOMLParser() throws {
    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "espresso.gpt2.124m",
        modelFamily: .gpt2,
        architectureVersion: "decoder-v1",
        tokenizerContract: "gpt2-bpe-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 1024,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    let parsed = try ESPManifest.parseTOML(manifest.renderTOML())
    #expect(parsed == manifest)
}

@Test func bundleCreateAndOpenRoundTrip() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weights = root.appendingPathComponent("weights-src", isDirectory: true)
    let tokenizer = root.appendingPathComponent("tokenizer-src", isDirectory: true)
    let bundle = root.appendingPathComponent("model.esp", isDirectory: true)

    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: weights.appendingPathComponent("metadata.json"))
    try Data("weights".utf8).write(to: weights.appendingPathComponent("lm_head.bin"))
    try Data("tokenizer".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))

    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "espresso.llama.test",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 2048,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    _ = try ESPBundleArchive.create(
        at: bundle,
        manifest: manifest,
        weightsDirectory: weights,
        tokenizerDirectory: tokenizer
    )

    let opened = try ESPBundleArchive.open(at: bundle)
    #expect(opened.manifest == manifest)
    #expect(FileManager.default.fileExists(atPath: opened.signatureCatalogURL.path))
}

@Test func bundleOpenRejectsTamperedFileContent() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weights = root.appendingPathComponent("weights-src", isDirectory: true)
    let tokenizer = root.appendingPathComponent("tokenizer-src", isDirectory: true)
    let bundle = root.appendingPathComponent("model.esp", isDirectory: true)

    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: weights.appendingPathComponent("metadata.json"))
    try Data("weights".utf8).write(to: weights.appendingPathComponent("lm_head.bin"))
    try Data("tokenizer".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))

    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "espresso.llama.test",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 2048,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    let archive = try ESPBundleArchive.create(
        at: bundle,
        manifest: manifest,
        weightsDirectory: weights,
        tokenizerDirectory: tokenizer
    )
    try Data("tampered".utf8).write(to: archive.weightsURL.appendingPathComponent("lm_head.bin"))

    do {
        _ = try ESPBundleArchive.open(at: bundle)
        #expect(Bool(false), "Expected signature verification to fail")
    } catch let error as ESPBundleValidationError {
        #expect(error == .signatureMismatch(path: "weights/lm_head.bin"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}
