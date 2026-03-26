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
        var index = 3

        while index < arguments.count {
            switch arguments[index] {
            case "--tokenizer-dir":
                index += 1
                guard index < arguments.count else {
                    throw usageError("Missing value for --tokenizer-dir")
                }
                tokenizerDirectory = arguments[index]
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
              espc pack-native <model-dir> <bundle-path> [--tokenizer-dir DIR] [--overwrite]
              espc inspect <bundle-path>

            """,
            stderr
        )
    }
}
