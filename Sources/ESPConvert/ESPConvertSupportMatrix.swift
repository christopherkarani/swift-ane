import ESPBundle
import ESPCompiler

public enum ESPConvertSupportTier: String, Sendable, Equatable {
    case tierA
    case tierB
    case unsupported
}

public enum ESPConvertSupportMatrix {
    public static func support(for family: ESPModelFamily) -> ESPConvertSupportTier {
        switch family {
        case .gpt2, .llama, .qwen:
            .tierA
        }
    }

    public static func recommendedImportSource(for family: ESPModelFamily) -> ESPImportSourceKind {
        switch family {
        case .gpt2:
            .nativeModelDirectoryWithExternalTokenizer
        case .llama, .qwen:
            .nativeModelDirectory
        }
    }
}
