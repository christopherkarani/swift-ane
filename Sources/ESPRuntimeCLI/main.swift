import ESPRuntime
import Foundation

@main
struct ESPRuntimeCLI {
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
        case "inspect":
            guard args.count == 2 else {
                throw usageError("Usage: esprun inspect <bundle-path>")
            }
            let bundle = try ESPRuntimeBundle.open(at: URL(fileURLWithPath: args[1], isDirectory: true))
            print(bundle.archive.manifest.renderTOML(), terminator: "")
        case "resolve":
            guard args.count == 2 else {
                throw usageError("Usage: esprun resolve <bundle-path>")
            }
            let bundle = try ESPRuntimeBundle.open(at: URL(fileURLWithPath: args[1], isDirectory: true))
            let selection = try ESPRuntimeRunner.resolve(bundle: bundle)
            print(selection.backend.rawValue)
        case "generate":
            guard args.count >= 3 else {
                throw usageError("Usage: esprun generate <bundle-path> <prompt> [max-tokens]")
            }
            let bundle = try ESPRuntimeBundle.open(at: URL(fileURLWithPath: args[1], isDirectory: true))
            let prompt = args[2]
            let maxTokens = args.count > 3 ? Int(args[3]) ?? 32 : 32
            let result = try ESPRuntimeRunner.generate(bundle: bundle, prompt: prompt, maxTokens: maxTokens)
            print(result.text)
        default:
            throw usageError("Unknown esprun command: \(command)")
        }
    }

    private static func usageError(_ message: String) -> NSError {
        NSError(domain: "ESPRuntimeCLI", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func printUsage() {
        fputs(
            """
            Usage:
              esprun inspect <bundle-path>
              esprun resolve <bundle-path>
              esprun generate <bundle-path> <prompt> [max-tokens]

            """,
            stderr
        )
    }
}
