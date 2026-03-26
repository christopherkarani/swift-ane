import ESPBundle

public enum ESPArchitectureRejectionReason: String, Sendable, Equatable {
    case dynamicControlFlow = "dynamic_control_flow"
    case mixtureOfExperts = "mixture_of_experts"
}

public enum ESPArchitectureSupport: Sendable, Equatable {
    case supported
    case unsupported(ESPArchitectureRejectionReason)
}

public enum ESPImportSourceKind: String, Sendable, Equatable, CaseIterable {
    case nativeModelDirectory = "native-model-directory"
    case nativeModelDirectoryWithExternalTokenizer = "native-model-directory-with-external-tokenizer"
}

public enum ESPCompilerSupportMatrix {
    public static let supportedModelFamilies: [ESPModelFamily] = [.gpt2, .llama, .qwen]
    public static let defaultBackends: [ESPBackendKind] = [.anePrivate, .cpuSafe]
    public static let defaultShippingProfiles: [ESPProfile] = [.prefill256, .prefill2048, .decode1]
    public static let experimentalProfiles: [ESPProfile] = [.decode2]

    public static func classifyArchitecture(
        hasDynamicControlFlow: Bool,
        hasMixtureOfExperts: Bool
    ) -> ESPArchitectureSupport {
        if hasDynamicControlFlow {
            return .unsupported(.dynamicControlFlow)
        }

        if hasMixtureOfExperts {
            return .unsupported(.mixtureOfExperts)
        }

        return .supported
    }
}
