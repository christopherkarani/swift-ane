import Darwin
import ESPBundle
import ESPCompiler
import Foundation
import ModelSupport
import RealModelInference

public struct ESPRuntimeBundle: Sendable, Equatable {
    public let archive: ESPBundleArchive
    public let config: MultiModelConfig

    public init(archive: ESPBundleArchive, config: MultiModelConfig) {
        self.archive = archive
        self.config = config
    }

    public static func open(at bundleURL: URL) throws -> ESPRuntimeBundle {
        let archive = try ESPBundleArchive.open(at: bundleURL)
        let config = try ESPModelConfigIO.load(
            fromMetadataFile: archive.weightsURL.appendingPathComponent("metadata.json")
        )
        return ESPRuntimeBundle(archive: archive, config: config)
    }
}

public struct ESPRuntimeHost {
    public static func currentCapabilities(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ESPDeviceCapabilities {
        ESPDeviceCapabilities(
            supportsANEPrivate: environment["ESPRESSO_DISABLE_ANE_PRIVATE"] != "1"
        )
    }
}

public enum ESPRuntimeRunner {
    public static func resolve(bundle: ESPRuntimeBundle) throws -> ESPRuntimeSelection {
        try ESPRuntimeResolver.selectBackend(
            capabilities: ESPRuntimeHost.currentCapabilities(),
            manifest: bundle.archive.manifest
        )
    }

    public static func generate(
        bundle: ESPRuntimeBundle,
        prompt: String,
        maxTokens: Int,
        temperature: Float = 0
    ) throws -> GenerationResult {
        let selection = try resolve(bundle: bundle)
        switch selection.backend {
        case .anePrivate:
            var engine = try RealModelInferenceEngine.build(
                config: bundle.config,
                weightDir: bundle.archive.weightsURL.path,
                tokenizerDir: bundle.archive.tokenizerURL.path
            )
            return try engine.generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature)
        case .cpuSafe:
            return try withTemporaryEnvironment(["ESPRESSO_USE_CPU_EXACT_DECODE": "1"]) {
                var engine = try RealModelInferenceEngine.build(
                    config: bundle.config,
                    weightDir: bundle.archive.weightsURL.path,
                    tokenizerDir: bundle.archive.tokenizerURL.path
                )
                return try engine.generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature)
            }
        }
    }
}

private func withTemporaryEnvironment<T>(
    _ overrides: [String: String],
    operation: () throws -> T
) throws -> T {
    var original: [String: String?] = [:]
    for key in overrides.keys {
        if let pointer = getenv(key) {
            original[key] = String(cString: pointer)
        } else {
            original[key] = nil
        }
    }

    for (key, value) in overrides {
        setenv(key, value, 1)
    }

    defer {
        for (key, originalValue) in original {
            if let originalValue {
                setenv(key, originalValue, 1)
            } else {
                unsetenv(key)
            }
        }
    }

    return try operation()
}
