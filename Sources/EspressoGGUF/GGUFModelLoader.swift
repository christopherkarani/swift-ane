import CryptoKit
import Foundation
import EspressoEdgeRunner
import EdgeRunnerIO
import ModelSupport
import ANETypes
import Metal

public enum GGUFModelLoaderError: Error, Sendable {
    case metalDeviceUnavailable
    case configMappingFailed(String)
    case unsupportedArchitecture(String)
    case conversionFailed(String)
    case noTensorsConverted
}

/// Loads a GGUF model file, converts weights to Espresso's BLOBFILE format,
/// and produces a weight directory + MultiModelConfig ready for RealModelInferenceEngine.build().
public enum GGUFModelLoader {
    static let prepareCacheEnvKey = "ESPRESSO_GGUF_PREPARE_CACHE"
    static let forceFreshPrepareEnvKey = "ESPRESSO_GGUF_FORCE_FRESH_PREPARE"
    static let prepareCacheRootEnvKey = "ESPRESSO_GGUF_PREPARE_CACHE_ROOT"
    static let sidecarPolicyEnvKey = "ESPRESSO_GGUF_SIDECAR_POLICY"
    static let selectedSidecarsEnvKey = "ESPRESSO_GGUF_SELECTED_SIDECARS"

    private static let preparedArtifactCacheVersion = 1
    private static let preparedArtifactManifestFileName = "prepare-manifest.json"
    private static let preparedArtifactCacheMarkerFileName = ".espresso-gguf-cache-artifact"
    private static let converterCacheEnvironmentKeys = [
        "ESPRESSO_FORCE_LLAMA_MATRIX_TRANSPOSE",
        "ESPRESSO_SKIP_LLAMA_MATRIX_TRANSPOSE",
        "ESPRESSO_FORCE_QK_INVERSE_INTERLEAVE",
        "ESPRESSO_DIM_MAJOR_TO_HEAD_MAJOR_V_WEIGHT",
        "ESPRESSO_INVERSE_INTERLEAVE_V_WEIGHT",
        "ESPRESSO_FORWARD_INTERLEAVE_V_WEIGHT",
    ]

    public struct PrepareOptions: Sendable, Equatable {
        public enum ArtifactCacheMode: Sendable, Equatable {
            case enabled
            case disabled
        }

        public enum ExactFloat32SidecarMode: Sendable, Equatable {
            case automatic
            case none
            case essential
            case selected(Set<String>)
            case full
        }

        public var artifactCacheMode: ArtifactCacheMode
        public var cacheRoot: URL?
        public var exactFloat32SidecarMode: ExactFloat32SidecarMode

        public init(
            artifactCacheMode: ArtifactCacheMode = .enabled,
            cacheRoot: URL? = nil,
            exactFloat32SidecarMode: ExactFloat32SidecarMode = .automatic
        ) {
            self.artifactCacheMode = artifactCacheMode
            self.cacheRoot = cacheRoot
            self.exactFloat32SidecarMode = exactFloat32SidecarMode
        }

        public static func environment(
            _ environment: [String: String] = ProcessInfo.processInfo.environment
        ) -> PrepareOptions {
            let cacheMode: ArtifactCacheMode
            if isEnabled(environment[GGUFModelLoader.forceFreshPrepareEnvKey]) {
                cacheMode = .disabled
            } else if let rawCacheMode = environment[GGUFModelLoader.prepareCacheEnvKey] {
                cacheMode = isEnabled(rawCacheMode) ? .enabled : .disabled
            } else {
                cacheMode = .enabled
            }

            let cacheRoot: URL?
            if let rawCacheRoot = environment[GGUFModelLoader.prepareCacheRootEnvKey] {
                let trimmed = rawCacheRoot.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    cacheRoot = nil
                } else {
                    cacheRoot = URL(
                        fileURLWithPath: NSString(string: trimmed).expandingTildeInPath,
                        isDirectory: true
                    )
                }
            } else {
                cacheRoot = nil
            }

            let selectedTensorNames = Set(
                environment[GGUFModelLoader.selectedSidecarsEnvKey]?
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty } ?? []
            )

            let sidecarMode: ExactFloat32SidecarMode
            switch environment[GGUFModelLoader.sidecarPolicyEnvKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased() {
            case nil, "":
                sidecarMode = .automatic
            case "automatic", "auto":
                sidecarMode = .automatic
            case "none", "off":
                sidecarMode = .none
            case "essential":
                sidecarMode = .essential
            case "selected":
                sidecarMode = .selected(selectedTensorNames)
            case "full":
                sidecarMode = .full
            default:
                sidecarMode = .automatic
            }

            return PrepareOptions(
                artifactCacheMode: cacheMode,
                cacheRoot: cacheRoot,
                exactFloat32SidecarMode: sidecarMode
            )
        }
    }

    public struct PreparedModel: Sendable {
        public let config: MultiModelConfig
        public let weightDir: String
        public let tensorCount: Int

        public init(config: MultiModelConfig, weightDir: String, tensorCount: Int) {
            self.config = config
            self.weightDir = weightDir
            self.tensorCount = tensorCount
        }
    }

    private struct PreparedArtifactManifest: Codable {
        let cacheVersion: Int
        let sourceGGUFPath: String
        let sourceFileSize: UInt64
        let sourceModificationTime: TimeInterval
        let tensorCount: Int
        let exactFloat32SidecarMode: String
        let selectedSidecarTensorNames: [String]
        let converterEnvironment: [String: String]
    }

    private struct PreparedMetadataFile: Decodable {
        let name: String
        let nLayer: Int
        let nHead: Int
        let nKVHead: Int
        let dModel: Int
        let headDim: Int
        let hiddenDim: Int
        let vocab: Int
        let maxSeq: Int
        let normEps: Float
        let ropeTheta: Float?
        let eosToken: Int?
        let architecture: String

        func asConfig() throws -> MultiModelConfig {
            let parsedArchitecture: MultiModelConfig.Architecture
            switch architecture.lowercased() {
            case "gpt2":
                parsedArchitecture = .gpt2
            case "llama":
                parsedArchitecture = .llama
            default:
                throw GGUFModelLoaderError.configMappingFailed("Unsupported cached architecture: \(architecture)")
            }
            return MultiModelConfig(
                name: name,
                nLayer: nLayer,
                nHead: nHead,
                nKVHead: nKVHead,
                dModel: dModel,
                headDim: headDim,
                hiddenDim: hiddenDim,
                vocab: vocab,
                maxSeq: maxSeq,
                normEps: normEps,
                ropeTheta: ropeTheta ?? 10_000.0,
                eosToken: eosToken.flatMap(TokenID.init),
                architecture: parsedArchitecture
            )
        }
    }

    static func runtimeArchitectureName(for architecture: MultiModelConfig.Architecture) -> String {
        switch architecture {
        case .gpt2:
            return "gpt2"
        case .llama:
            return "llama"
        }
    }

    static func metadataDictionary(for config: MultiModelConfig) -> [String: Any] {
        var metadata: [String: Any] = [
            "name": config.name,
            "nLayer": config.nLayer,
            "nHead": config.nHead,
            "nKVHead": config.nKVHead,
            "dModel": config.dModel,
            "headDim": config.headDim,
            "hiddenDim": config.hiddenDim,
            "vocab": config.vocab,
            "maxSeq": config.maxSeq,
            "normEps": config.normEps,
            "ropeTheta": config.ropeTheta,
            "architecture": runtimeArchitectureName(for: config.architecture),
        ]
        if let eosToken = config.eosToken {
            metadata["eosToken"] = Int(eosToken)
        }
        return metadata
    }

    static func materializeTiedLMHeadIfNeeded(
        tensorNames: some Sequence<String>,
        outputDirectory: URL,
        exactTiedLMHeadValues: [Float]? = nil,
        fileManager: FileManager = .default
    ) throws {
        let tensorNameSet = Set(tensorNames)
        guard !tensorNameSet.contains("output.weight") else { return }

        let embeddingURL = outputDirectory.appendingPathComponent("embeddings/token.bin")
        let lmHeadURL = outputDirectory.appendingPathComponent("lm_head.bin")
        guard fileManager.fileExists(atPath: embeddingURL.path),
              !fileManager.fileExists(atPath: lmHeadURL.path) else {
            return
        }

        do {
            try fileManager.linkItem(at: embeddingURL, to: lmHeadURL)
        } catch {
            try fileManager.copyItem(at: embeddingURL, to: lmHeadURL)
        }

        let exactEmbeddingURL = outputDirectory.appendingPathComponent("embeddings/token.float32.bin")
        let exactLMHeadURL = outputDirectory.appendingPathComponent("lm_head.float32.bin")
        guard !fileManager.fileExists(atPath: exactLMHeadURL.path) else {
            return
        }

        if fileManager.fileExists(atPath: exactEmbeddingURL.path) {
            do {
                try fileManager.linkItem(at: exactEmbeddingURL, to: exactLMHeadURL)
            } catch {
                try fileManager.copyItem(at: exactEmbeddingURL, to: exactLMHeadURL)
            }
            return
        }

        guard let exactTiedLMHeadValues else { return }
        try writeExactFloat32Sidecar(floats: exactTiedLMHeadValues, to: exactLMHeadURL)
    }

    /// Load a GGUF file and convert all weights to Espresso BLOBFILE format.
    /// Returns a PreparedModel with the config and a temp or cached directory of weight files.
    public static func prepare(ggufURL: URL) async throws -> PreparedModel {
        try await prepare(ggufURL: ggufURL, options: .environment())
    }

    /// Load a GGUF file and convert all weights to Espresso BLOBFILE format.
    /// Returns a PreparedModel with the config and a temp or cached directory of weight files.
    public static func prepare(
        ggufURL: URL,
        options: PrepareOptions = .environment()
    ) async throws -> PreparedModel {
        let standardizedGGUFURL = ggufURL.standardizedFileURL
        let fileManager = FileManager.default

        if options.artifactCacheMode == .enabled,
           let cachedPreparedModel = try loadCachedPreparedModelIfAvailable(
               ggufURL: standardizedGGUFURL,
               options: options,
               fileManager: fileManager
           ) {
            return cachedPreparedModel
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GGUFModelLoaderError.metalDeviceUnavailable
        }

        let ggufLoader = try GGUFLoader(url: standardizedGGUFURL)
        let weightMap = try await ggufLoader.load(from: standardizedGGUFURL)
        let erConfig = try EspressoModelConfig(from: ggufLoader.modelConfig)

        let arch: MultiModelConfig.Architecture
        switch erConfig.architectureName.lowercased() {
        case "gpt2":
            arch = .gpt2
        case "llama", "llama2", "llama3", "mistral", "qwen2", "qwen3":
            arch = .llama
        default:
            throw GGUFModelLoaderError.unsupportedArchitecture(erConfig.architectureName)
        }

        let ropeTheta = ggufLoader.modelConfig.float(forKey: "\(erConfig.architectureName).rope.freq_base")
            ?? 10_000.0

        let config = MultiModelConfig(
            name: erConfig.architectureName,
            nLayer: erConfig.blockCount,
            nHead: erConfig.headCount,
            nKVHead: erConfig.kvHeadCount,
            dModel: erConfig.embeddingDim,
            headDim: erConfig.headDim,
            hiddenDim: erConfig.feedForwardLength,
            vocab: ggufLoader.modelConfig.metadata["tokenizer.ggml.tokens"]?.arrayValue?.count
                ?? ggufLoader.modelConfig.int(forKey: "\(erConfig.architectureName).vocab_size")
                ?? 0,
            maxSeq: erConfig.contextLength,
            normEps: erConfig.rmsNormEpsilon,
            ropeTheta: ropeTheta,
            eosToken: ggufLoader.modelConfig.int(forKey: "tokenizer.ggml.eos_token_id").flatMap(TokenID.init),
            architecture: arch
        )

        let exactFloat32SidecarPolicy = resolveExactFloat32SidecarPolicy(
            mode: options.exactFloat32SidecarMode
        )
        let outputDir = try preparedArtifactOutputDirectory(
            ggufURL: standardizedGGUFURL,
            options: options,
            fileManager: fileManager
        )
        var shouldCleanupOnFailure = options.artifactCacheMode == .enabled

        do {
            try fileManager.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let converter = try WeightConverter(device: device)
            let count = try await converter.convert(
                weightMap: weightMap,
                architecture: erConfig.architectureName,
                outputDirectory: outputDir,
                llamaProjectionLayout: arch == .llama ? WeightConverter.LlamaProjectionLayout(
                    qHeadCount: erConfig.headCount,
                    kvHeadCount: erConfig.kvHeadCount,
                    headDim: erConfig.headDim
                ) : nil,
                exactFloat32SidecarPolicy: exactFloat32SidecarPolicy
            )

            guard count > 0 else {
                throw GGUFModelLoaderError.noTensorsConverted
            }

            let exactEmbeddingURL = outputDir.appendingPathComponent("embeddings/token.float32.bin")
            let shouldMaterializeExactLMHeadSidecar = WeightConverter.shouldWriteExactFloat32Sidecar(
                ggufName: "output.weight",
                architecture: erConfig.architectureName,
                policy: exactFloat32SidecarPolicy
            )

            let exactTiedLMHeadValues: [Float]? = if !weightMap.tensorNames.contains("output.weight"),
                shouldMaterializeExactLMHeadSidecar,
                !fileManager.fileExists(atPath: exactEmbeddingURL.path),
                let tokenEmbeddingTensor = weightMap["token_embd.weight"] {
                try await DequantDispatcher.dequantize(tensor: tokenEmbeddingTensor, device: device)
            } else {
                nil
            }

            try materializeTiedLMHeadIfNeeded(
                tensorNames: weightMap.tensorNames,
                outputDirectory: outputDir,
                exactTiedLMHeadValues: exactTiedLMHeadValues
            )

            let metadata = metadataDictionary(for: config)
            if let jsonData = try? JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted) {
                try? jsonData.write(to: outputDir.appendingPathComponent("metadata.json"))
            }

            try writePreparedArtifactManifest(
                ggufURL: standardizedGGUFURL,
                outputDirectory: outputDir,
                options: options,
                tensorCount: count,
                fileManager: fileManager
            )

            shouldCleanupOnFailure = false
            return PreparedModel(
                config: config,
                weightDir: outputDir.path,
                tensorCount: count
            )
        } catch {
            if shouldCleanupOnFailure {
                try? fileManager.removeItem(at: outputDir)
            }
            throw error
        }
    }

    /// Clean up a temporary weight directory created by prepare().
    public static func cleanup(weightDir: String) {
        let url = URL(fileURLWithPath: weightDir, isDirectory: true)
        guard !isManagedCachedArtifactDirectory(url) else { return }
        try? FileManager.default.removeItem(atPath: weightDir)
    }

    static func defaultPreparedArtifactCacheRoot(
        fileManager: FileManager = .default
    ) -> URL {
        let cacheBase = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        return cacheBase
            .appendingPathComponent("Espresso", isDirectory: true)
            .appendingPathComponent("gguf-prepared", isDirectory: true)
    }

    static func artifactCacheKey(
        ggufURL: URL,
        options: PrepareOptions,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) throws -> String {
        let standardizedGGUFURL = ggufURL.standardizedFileURL
        let attributes = try fileManager.attributesOfItem(atPath: standardizedGGUFURL.path)
        let fileSize = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        let modificationTime = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0

        let cachePayload: [String: Any] = [
            "version": preparedArtifactCacheVersion,
            "ggufPath": standardizedGGUFURL.path,
            "fileSize": fileSize,
            "modificationTime": modificationTime,
            "sidecarMode": sidecarModeCacheDescription(options.exactFloat32SidecarMode),
            "selectedSidecars": selectedSidecarTensorNames(for: options.exactFloat32SidecarMode),
            "converterEnvironment": converterEnvironmentFingerprint(from: environment),
        ]
        let jsonData = try JSONSerialization.data(withJSONObject: cachePayload, options: [.sortedKeys])
        let digest = SHA256.hash(data: jsonData)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func isManagedCachedArtifactDirectory(_ url: URL, fileManager: FileManager = .default) -> Bool {
        fileManager.fileExists(
            atPath: url.appendingPathComponent(preparedArtifactCacheMarkerFileName).path
        )
    }

    private static func resolveExactFloat32SidecarPolicy(
        mode: PrepareOptions.ExactFloat32SidecarMode
    ) -> WeightConverter.ExactFloat32SidecarPolicy {
        switch mode {
        case .automatic:
            return .automatic
        case .none:
            return .none
        case .essential:
            return .essential
        case .selected(let tensorNames):
            return .selected(tensorNames)
        case .full:
            return .full
        }
    }

    private static func preparedArtifactOutputDirectory(
        ggufURL: URL,
        options: PrepareOptions,
        fileManager: FileManager
    ) throws -> URL {
        guard options.artifactCacheMode == .enabled else {
            return fileManager.temporaryDirectory
                .appendingPathComponent("espresso_gguf_\(UUID().uuidString)", isDirectory: true)
        }

        let cacheRoot = options.cacheRoot ?? defaultPreparedArtifactCacheRoot(fileManager: fileManager)
        let cacheKey = try artifactCacheKey(
            ggufURL: ggufURL,
            options: options,
            fileManager: fileManager
        )
        let cacheDirectory = cacheRoot.appendingPathComponent(cacheKey, isDirectory: true)
        if fileManager.fileExists(atPath: cacheDirectory.path) {
            try? fileManager.removeItem(at: cacheDirectory)
        }
        return cacheDirectory
    }

    private static func loadCachedPreparedModelIfAvailable(
        ggufURL: URL,
        options: PrepareOptions,
        fileManager: FileManager
    ) throws -> PreparedModel? {
        let cacheRoot = options.cacheRoot ?? defaultPreparedArtifactCacheRoot(fileManager: fileManager)
        let cacheKey = try artifactCacheKey(
            ggufURL: ggufURL,
            options: options,
            fileManager: fileManager
        )
        let cacheDirectory = cacheRoot.appendingPathComponent(cacheKey, isDirectory: true)
        guard fileManager.fileExists(atPath: cacheDirectory.path) else {
            return nil
        }

        let manifestURL = cacheDirectory.appendingPathComponent(preparedArtifactManifestFileName)
        let metadataURL = cacheDirectory.appendingPathComponent("metadata.json")
        guard fileManager.fileExists(atPath: manifestURL.path),
              fileManager.fileExists(atPath: metadataURL.path) else {
            try? fileManager.removeItem(at: cacheDirectory)
            return nil
        }
        do {
            let manifestData = try Data(contentsOf: manifestURL)
            let manifest = try JSONDecoder().decode(PreparedArtifactManifest.self, from: manifestData)
            let metadataData = try Data(contentsOf: metadataURL)
            let metadata = try JSONDecoder().decode(PreparedMetadataFile.self, from: metadataData)
            let config = try metadata.asConfig()
            return PreparedModel(
                config: config,
                weightDir: cacheDirectory.path,
                tensorCount: manifest.tensorCount
            )
        } catch {
            try? fileManager.removeItem(at: cacheDirectory)
            return nil
        }
    }

    private static func writePreparedArtifactManifest(
        ggufURL: URL,
        outputDirectory: URL,
        options: PrepareOptions,
        tensorCount: Int,
        fileManager: FileManager
    ) throws {
        guard options.artifactCacheMode == .enabled else { return }

        let attributes = try fileManager.attributesOfItem(atPath: ggufURL.path)
        let manifest = PreparedArtifactManifest(
            cacheVersion: preparedArtifactCacheVersion,
            sourceGGUFPath: ggufURL.path,
            sourceFileSize: (attributes[.size] as? NSNumber)?.uint64Value ?? 0,
            sourceModificationTime: (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0,
            tensorCount: tensorCount,
            exactFloat32SidecarMode: sidecarModeCacheDescription(options.exactFloat32SidecarMode),
            selectedSidecarTensorNames: selectedSidecarTensorNames(for: options.exactFloat32SidecarMode),
            converterEnvironment: converterEnvironmentFingerprint(from: ProcessInfo.processInfo.environment)
        )
        let manifestData = try JSONEncoder().encode(manifest)
        try manifestData.write(
            to: outputDirectory.appendingPathComponent(preparedArtifactManifestFileName),
            options: .atomic
        )
        try Data().write(
            to: outputDirectory.appendingPathComponent(preparedArtifactCacheMarkerFileName),
            options: .atomic
        )
    }

    private static func selectedSidecarTensorNames(
        for mode: PrepareOptions.ExactFloat32SidecarMode
    ) -> [String] {
        switch mode {
        case .selected(let tensorNames):
            return tensorNames.sorted()
        default:
            return []
        }
    }

    private static func sidecarModeCacheDescription(
        _ mode: PrepareOptions.ExactFloat32SidecarMode
    ) -> String {
        switch mode {
        case .automatic:
            return "automatic"
        case .none:
            return "none"
        case .essential:
            return "essential"
        case .selected:
            return "selected"
        case .full:
            return "full"
        }
    }

    private static func converterEnvironmentFingerprint(
        from environment: [String: String]
    ) -> [String: String] {
        var fingerprint: [String: String] = [:]
        for key in converterCacheEnvironmentKeys.sorted() {
            if let value = environment[key], !value.isEmpty {
                fingerprint[key] = value
            }
        }
        return fingerprint
    }

    private static func isEnabled(_ rawValue: String?) -> Bool {
        guard let rawValue else { return false }
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    private static func writeExactFloat32Sidecar(
        floats: [Float],
        to url: URL
    ) throws {
        var data = Data(capacity: floats.count * MemoryLayout<UInt32>.stride)
        for value in floats {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        try data.write(to: url, options: .atomic)
    }
}
