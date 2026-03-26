import Testing
import Foundation
@testable import ESPBundle

@Test func manifestRenderingIsDeterministic() throws {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.qwen.0_6b",
        modelFamily: .qwen,
        architectureVersion: "decoder-v1",
        tokenizerContract: "qwen-bpe-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .prefill2048, .decode1],
        maxContext: 2048,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "int4-palettized", weightBits: 4, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .nearExact,
        adapterSlots: 4,
        optimization: .init(
            recipe: "stories-gqa4-distilled",
            qualityGate: "short-long-prompt-parity",
            teacherModel: "teacher://qwen3-0.6b",
            draftModel: nil,
            performanceTarget: "110 tok/s"
        ),
        accuracyBaselineRef: "benchmarks/qwen-0.6b/accuracy.json",
        performanceBaselineRef: "benchmarks/qwen-0.6b/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    let renderedA = manifest.renderTOML()
    let renderedB = manifest.renderTOML()
    #expect(renderedA == renderedB)
    #expect(renderedA.contains("model_family = \"qwen\""))
    #expect(renderedA.contains("model_tier = \"optimized\""))
    #expect(renderedA.contains("behavior_class = \"near_exact\""))
    #expect(renderedA.contains("context_target_tokens = 1024"))
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
        formatVersion: "1.1.0",
        modelID: "espresso.gpt2.124m",
        modelFamily: .gpt2,
        architectureVersion: "decoder-v1",
        tokenizerContract: "gpt2-bpe-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 1024,
        contextTargetTokens: 512,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .compat,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "native-baseline",
            qualityGate: "exact",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: nil
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    let parsed = try ESPManifest.parseTOML(manifest.renderTOML())
    #expect(parsed == manifest)
}

@Test func manifestParserBackfillsDefaultsForLegacyV1Bundles() throws {
    let text = """
    format_version = "1.0.0"
    model_id = "legacy.stories"
    model_family = "llama"
    architecture_version = "decoder-v1"
    tokenizer_contract = "sentencepiece-v1"
    supported_backends = ["ane-private", "cpu-safe"]
    supported_profiles = ["prefill_256", "decode_1"]
    max_context = 256
    adapter_slots = 0
    accuracy_baseline_ref = "benchmarks/accuracy.json"
    performance_baseline_ref = "benchmarks/perf.json"
    signature_ref = "signatures/content-hashes.json"
    [compression_policy]
    name = "native-ane-fp16"
    weight_bits = 16
    """

    let manifest = try ESPManifest.parseTOML(text)

    #expect(manifest.modelTier == .compat)
    #expect(manifest.behaviorClass == .exact)
    #expect(manifest.contextTargetTokens == 256)
    #expect(manifest.optimization.recipe == "legacy")
    #expect(manifest.optimization.qualityGate == "legacy-compatible")
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
        formatVersion: "1.1.0",
        modelID: "espresso.llama.test",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 2048,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "stories-ctx1024",
            qualityGate: "short-long-prompt-parity",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: "105 tok/s"
        ),
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
        formatVersion: "1.1.0",
        modelID: "espresso.llama.test",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 2048,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "stories-ctx1024",
            qualityGate: "short-long-prompt-parity",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: "105 tok/s"
        ),
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
