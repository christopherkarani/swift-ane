import ESPBundle
import ESPConvert
import Foundation

@main
struct ESPCompilerCLI {
    static func main() {
        do {
            try run()
        } catch {
            fputs("Error: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }

    private static func run() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let command = args.first else {
            printUsage()
            return
        }

        switch command {
        case "pack-native":
            let request = try parsePackNativeRequest(arguments: args)
            let archive = try ESPNativeModelBundleExporter.exportModel(
                at: URL(fileURLWithPath: request.modelDirectory, isDirectory: true),
                tokenizerDirectory: request.tokenizerDirectory.map {
                    URL(fileURLWithPath: $0, isDirectory: true)
                },
                outputBundleURL: URL(fileURLWithPath: request.bundlePath, isDirectory: true),
                options: request.exportOptions,
                overwriteExisting: request.overwriteExisting
            )
            print(archive.bundleURL.path)
        case "inspect":
            guard args.count == 2 else {
                throw usageError("Usage: espc inspect <bundle-path>")
            }
            let archive = try ESPBundleArchive.open(at: URL(fileURLWithPath: args[1], isDirectory: true))
            print(archive.manifest.renderTOML(), terminator: "")
        default:
            throw usageError("Unknown espc command: \(command)")
        }
    }

    private struct PackNativeRequest {
        let modelDirectory: String
        let tokenizerDirectory: String?
        let bundlePath: String
        let exportOptions: ESPNativeModelBundleExportOptions
        let overwriteExisting: Bool
    }

    private static func parsePackNativeRequest(arguments: [String]) throws -> PackNativeRequest {
        guard arguments.count >= 3 else {
            throw usageError(
                "Usage: espc pack-native <model-dir> <bundle-path> [--tokenizer-dir DIR] [--overwrite]"
            )
        }

        let modelDirectory = arguments[1]
        let bundlePath = arguments[2]
        var tokenizerDirectory: String?
        var overwriteExisting = false
        var contextTargetTokens: Int?
        var modelTier = ESPModelTier.compat
        var behaviorClass = ESPBehaviorClass.exact
        var optimizationRecipe = "native-baseline"
        var qualityGate = "exact"
        var teacherModel: String?
        var draftModel: String?
        var performanceTarget: String?
        var index = 3

        while index < arguments.count {
            switch arguments[index] {
            case "--tokenizer-dir":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --tokenizer-dir")
                }
                tokenizerDirectory = arguments[index]
            case "--context-target":
                index += 1
                guard index < arguments.count, let value = Int(arguments[index]), value > 0 else {
                    throw usageError("Expected a positive integer for --context-target")
                }
                contextTargetTokens = value
            case "--model-tier":
                index += 1
                guard index < arguments.count, let value = ESPModelTier(rawValue: arguments[index]) else {
                    throw usageError("Expected --model-tier compat|optimized|native_fast")
                }
                modelTier = value
            case "--behavior-class":
                index += 1
                guard index < arguments.count, let value = ESPBehaviorClass(rawValue: arguments[index]) else {
                    throw usageError("Expected --behavior-class exact|near_exact|approximate")
                }
                behaviorClass = value
            case "--optimization-recipe":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --optimization-recipe")
                }
                optimizationRecipe = arguments[index]
            case "--quality-gate":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --quality-gate")
                }
                qualityGate = arguments[index]
            case "--teacher-model":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --teacher-model")
                }
                teacherModel = arguments[index]
            case "--draft-model":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --draft-model")
                }
                draftModel = arguments[index]
            case "--performance-target":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --performance-target")
                }
                performanceTarget = arguments[index]
            case "--overwrite":
                overwriteExisting = true
            default:
                throw usageError("Unknown espc argument: \(arguments[index])")
            }
            index += 1
        }

        return PackNativeRequest(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            bundlePath: bundlePath,
            exportOptions: .init(
                contextTargetTokens: contextTargetTokens,
                modelTier: modelTier,
                behaviorClass: behaviorClass,
                optimization: .init(
                    recipe: optimizationRecipe,
                    qualityGate: qualityGate,
                    teacherModel: teacherModel,
                    draftModel: draftModel,
                    performanceTarget: performanceTarget
                )
            ),
            overwriteExisting: overwriteExisting
        )
    }

    private static func usageError(_ message: String) -> NSError {
        NSError(domain: "ESPCompilerCLI", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func printUsage() {
        fputs(
            """
            Usage:
              espc pack-native <model-dir> <bundle-path> [--tokenizer-dir DIR] [--context-target TOKENS] [--model-tier compat|optimized|native_fast] [--behavior-class exact|near_exact|approximate] [--optimization-recipe NAME] [--quality-gate NAME] [--teacher-model MODEL] [--draft-model MODEL] [--performance-target VALUE] [--overwrite]
              espc inspect <bundle-path>

            """,
            stderr
        )
    }
}
