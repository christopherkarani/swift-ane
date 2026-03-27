import Foundation
import Testing
@testable import ESPCompilerCLI

@Test func packNativeRejectsUnsupportedMultiTokenDraftKind() {
    do {
        _ = try ESPCompilerCLI.parsePackNativeRequest(arguments: [
            "pack-native",
            "/tmp/model",
            "/tmp/model.esp",
            "--draft-kind",
            "multi_token",
        ])
        Issue.record("Expected unsupported draft-kind rejection")
    } catch let error as NSError {
        #expect(error.localizedDescription == "Expected --draft-kind exact_two_token")
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test func packNativeAcceptsExactTwoTokenDraftKind() throws {
    let request = try ESPCompilerCLI.parsePackNativeRequest(arguments: [
        "pack-native",
        "/tmp/model",
        "/tmp/model.esp",
        "--draft-kind",
        "exact_two_token",
    ])

    #expect(request.exportOptions.draft?.kind == .exactTwoToken)
    #expect(request.exportOptions.draft?.horizon == 2)
}
