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
    public let reason: String

    public init(backend: ESPBackendKind, reason: String) {
        self.backend = backend
        self.reason = reason
    }
}

public enum ESPRuntimeSelectionError: Error, Equatable {
    case noCompatibleBackend
}

public enum ESPRuntimeResolver {
    public static func selectBackend(
        capabilities: ESPDeviceCapabilities,
        manifest: ESPManifest,
        preferred: [ESPBackendKind] = [.anePrivate, .cpuSafe]
    ) throws -> ESPRuntimeSelection {
        try manifest.validate()

        for backend in preferred {
            guard manifest.supportedBackends.contains(backend) else {
                continue
            }

            switch backend {
            case .anePrivate where capabilities.supportsANEPrivate:
                return ESPRuntimeSelection(
                    backend: .anePrivate,
                    reason: "Private ANE backend is available on this host"
                )
            case .cpuSafe:
                return ESPRuntimeSelection(
                    backend: .cpuSafe,
                    reason: "Fell back to CPU-safe backend"
                )
            default:
                continue
            }
        }

        throw ESPRuntimeSelectionError.noCompatibleBackend
    }
}
