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

    struct PackNativeRequest {
        let modelDirectory: String
        let tokenizerDirectory: String?
        let bundlePath: String
        let exportOptions: ESPNativeModelBundleExportOptions
        let overwriteExisting: Bool
    }

    static func parsePackNativeRequest(arguments: [String]) throws -> PackNativeRequest {
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
        var outputHeadKind: ESPOutputHeadKind?
        var outputHeadBehaviorClass: ESPBehaviorClass?
        var outputHeadBottleneck: Int?
        var outputHeadGroups: Int?
        var outputHeadProjectionRef: String?
        var outputHeadExpansionRef: String?
        var draftKind: ESPDraftKind?
        var draftBehaviorClass: ESPBehaviorClass?
        var draftHorizon: Int?
        var draftVerifier: String?
        var draftRollback: String?
        var draftArtifactRef: String?
        var draftAcceptanceMetric: String?
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
            case "--output-head-kind":
                index += 1
                guard index < arguments.count,
                      let value = ESPOutputHeadKind(rawValue: arguments[index]) else {
                    throw usageError("Expected --output-head-kind dense|factored")
                }
                outputHeadKind = value
            case "--output-head-behavior-class":
                index += 1
                guard index < arguments.count,
                      let value = ESPBehaviorClass(rawValue: arguments[index]) else {
                    throw usageError("Expected --output-head-behavior-class exact|near_exact|approximate")
                }
                outputHeadBehaviorClass = value
            case "--output-head-bottleneck":
                index += 1
                guard index < arguments.count,
                      let value = Int(arguments[index]),
                      value > 0 else {
                    throw usageError("Expected a positive integer for --output-head-bottleneck")
                }
                outputHeadBottleneck = value
            case "--output-head-groups":
                index += 1
                guard index < arguments.count,
                      let value = Int(arguments[index]),
                      value > 0 else {
                    throw usageError("Expected a positive integer for --output-head-groups")
                }
                outputHeadGroups = value
            case "--output-head-projection":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --output-head-projection")
                }
                outputHeadProjectionRef = arguments[index]
            case "--output-head-expansion":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --output-head-expansion")
                }
                outputHeadExpansionRef = arguments[index]
            case "--draft-kind":
                index += 1
                guard index < arguments.count,
                      let value = ESPDraftKind(rawValue: arguments[index]) else {
                    throw usageError("Expected --draft-kind exact_two_token")
                }
                guard value == .exactTwoToken else {
                    throw usageError("Expected --draft-kind exact_two_token")
                }
                draftKind = value
            case "--draft-behavior-class":
                index += 1
                guard index < arguments.count,
                      let value = ESPBehaviorClass(rawValue: arguments[index]) else {
                    throw usageError("Expected --draft-behavior-class exact|near_exact|approximate")
                }
                draftBehaviorClass = value
            case "--draft-horizon":
                index += 1
                guard index < arguments.count,
                      let value = Int(arguments[index]),
                      value > 1 else {
                    throw usageError("Expected an integer > 1 for --draft-horizon")
                }
                draftHorizon = value
            case "--draft-verifier":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --draft-verifier")
                }
                draftVerifier = arguments[index]
            case "--draft-rollback":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --draft-rollback")
                }
                draftRollback = arguments[index]
            case "--draft-artifact":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --draft-artifact")
                }
                draftArtifactRef = arguments[index]
            case "--draft-acceptance-metric":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --draft-acceptance-metric")
                }
                draftAcceptanceMetric = arguments[index]
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
                ),
                outputHead: outputHeadKind.map {
                    ESPOutputHeadMetadata(
                        kind: $0,
                        behaviorClass: outputHeadBehaviorClass ?? behaviorClass,
                        bottleneck: outputHeadBottleneck,
                        groups: outputHeadGroups,
                        projectionRef: outputHeadProjectionRef,
                        expansionRef: outputHeadExpansionRef
                    )
                },
                draft: draftKind.map {
                    ESPDraftMetadata(
                        kind: $0,
                        behaviorClass: draftBehaviorClass ?? behaviorClass,
                        horizon: draftHorizon ?? ($0 == .exactTwoToken ? 2 : 4),
                        verifier: draftVerifier ?? "exact",
                        rollback: draftRollback ?? "exact_replay",
                        artifactRef: draftArtifactRef ?? "weights/future-sidecar.bin",
                        acceptanceMetric: draftAcceptanceMetric ?? "accepted_future_tokens"
                    )
                }
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
              espc pack-native <model-dir> <bundle-path> [--tokenizer-dir DIR] [--context-target TOKENS] [--model-tier compat|optimized|native_fast] [--behavior-class exact|near_exact|approximate] [--optimization-recipe NAME] [--quality-gate NAME] [--teacher-model MODEL] [--draft-model MODEL] [--performance-target VALUE] [--output-head-kind dense|factored] [--output-head-behavior-class exact|near_exact|approximate] [--output-head-bottleneck TOKENS] [--output-head-groups COUNT] [--output-head-projection PATH] [--output-head-expansion PATH] [--draft-kind exact_two_token] [--draft-behavior-class exact|near_exact|approximate] [--draft-horizon TOKENS] [--draft-verifier MODE] [--draft-rollback MODE] [--draft-artifact PATH] [--draft-acceptance-metric NAME] [--overwrite]
              espc inspect <bundle-path>

            """,
            stderr
        )
    }
}
