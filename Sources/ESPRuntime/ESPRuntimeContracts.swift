import ESPBundle

public typealias ESPTokenID = Int

public protocol ESPRuntimeSession: Sendable {
    mutating func prefill(tokens: [ESPTokenID]) async throws
    mutating func decodeOne() async throws -> ESPTokenID
    mutating func decode(count: Int) async throws -> [ESPTokenID]
    mutating func reset() async throws
    mutating func attachAdapter(_ descriptor: ESPAdapterDescriptor) async throws
    mutating func detachAdapter() async throws
    func metrics() -> ESPRuntimeMetricsSnapshot
}

public struct ESPAdapterDescriptor: Sendable, Equatable {
    public let identifier: String
    public let path: String

    public init(identifier: String, path: String) {
        self.identifier = identifier
        self.path = path
    }
}

public struct ESPRuntimeMetricsSnapshot: Sendable, Equatable {
    public let selectedBackend: ESPBackendKind?
    public let compileCacheHit: Bool

    public init(selectedBackend: ESPBackendKind?, compileCacheHit: Bool) {
        self.selectedBackend = selectedBackend
        self.compileCacheHit = compileCacheHit
    }
}

public struct ESPDeviceCapabilities: Sendable, Equatable {
    public let supportsANEPrivate: Bool

    public init(supportsANEPrivate: Bool) {
        self.supportsANEPrivate = supportsANEPrivate
    }
}

public struct ESPRuntimeSelection: Sendable, Equatable {
    public let backend: ESPBackendKind
    public let profile: ESPProfile
    public let contextTargetTokens: Int
    public let reason: String

    public init(
        backend: ESPBackendKind,
        profile: ESPProfile,
        contextTargetTokens: Int,
        reason: String
    ) {
        self.backend = backend
        self.profile = profile
        self.contextTargetTokens = contextTargetTokens
        self.reason = reason
    }
}

public enum ESPRuntimeSelectionError: Error, Equatable {
    case noCompatibleBackend
    case noCompatibleProfile
    case requestedContextExceedsTarget(requested: Int, supported: Int)
}

public enum ESPRuntimeResolver {
    public static func selectBackend(
        capabilities: ESPDeviceCapabilities,
        manifest: ESPManifest,
        requestedContextTokens: Int? = nil,
        preferred: [ESPBackendKind] = [.anePrivate, .cpuSafe]
    ) throws -> ESPRuntimeSelection {
        try manifest.validate()
        if let requestedContextTokens, requestedContextTokens > manifest.contextTargetTokens {
            throw ESPRuntimeSelectionError.requestedContextExceedsTarget(
                requested: requestedContextTokens,
                supported: manifest.contextTargetTokens
            )
        }
        let selectedProfile = try selectProfile(manifest: manifest)

        for backend in preferred {
            guard manifest.supportedBackends.contains(backend) else {
                continue
            }

            switch backend {
            case .anePrivate where capabilities.supportsANEPrivate:
                return ESPRuntimeSelection(
                    backend: .anePrivate,
                    profile: selectedProfile,
                    contextTargetTokens: manifest.contextTargetTokens,
                    reason: "Private ANE backend is available on this host"
                )
            case .cpuSafe:
                return ESPRuntimeSelection(
                    backend: .cpuSafe,
                    profile: selectedProfile,
                    contextTargetTokens: manifest.contextTargetTokens,
                    reason: "Fell back to CPU-safe backend"
                )
            default:
                continue
            }
        }

        throw ESPRuntimeSelectionError.noCompatibleBackend
    }

    private static func selectProfile(manifest: ESPManifest) throws -> ESPProfile {
        if manifest.contextTargetTokens > 256 {
            if manifest.supportedProfiles.contains(.prefill2048) {
                return .prefill2048
            }
            if manifest.supportedProfiles.contains(.prefill256) {
                return .prefill256
            }
        } else if manifest.supportedProfiles.contains(.prefill256) {
            return .prefill256
        }
        if let fallback = manifest.supportedProfiles.first {
            return fallback
        }
        throw ESPRuntimeSelectionError.noCompatibleProfile
    }
}
