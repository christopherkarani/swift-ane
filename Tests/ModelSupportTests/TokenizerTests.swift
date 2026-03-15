import Foundation
import Testing
@testable import ModelSupport

@Test func gpt2TokenizerRoundTripsHelloWorldAndNonPrintableBytes() throws {
    let fixture = try makeGPT2Fixture()
    let tokenizer = try GPT2BPETokenizer(vocabURL: fixture.vocabURL, mergesURL: fixture.mergesURL)

    let hello = "Hello, world!"
    let encoded = tokenizer.encode(hello)
    #expect(!encoded.isEmpty)
    #expect(tokenizer.decode(encoded) == hello)

    let newline = "\n"
    let newlineEncoded = tokenizer.encode(newline)
    #expect(newlineEncoded == [4])
    #expect(tokenizer.decode(newlineEncoded) == newline)
}

@Test func sentencePieceTokenizerRoundTripsHelloWorldAndByteFallback() throws {
    let modelURL = try makeSentencePieceFixture()
    let tokenizer = try SentencePieceTokenizer(modelURL: modelURL)

    let hello = "Hello, world!"
    let encoded = tokenizer.encode(hello)
    #expect(tokenizer.decode(encoded) == hello)

    let newline = "\n"
    let newlineEncoded = tokenizer.encode(newline)
    #expect(tokenizer.decode(newlineEncoded) == newline)
}

@Test func gpt2TokenizerRejectsMalformedVocabulary() throws {
    let directory = try makeTempDirectory()
    let vocabURL = directory.appendingPathComponent("vocab.json")
    let mergesURL = directory.appendingPathComponent("merges.txt")
    try #"["not","a","map"]"#.write(to: vocabURL, atomically: true, encoding: .utf8)
    try "#version: 0.2\n".write(to: mergesURL, atomically: true, encoding: .utf8)

    #expect(throws: GPT2BPETokenizerError.malformedVocabulary) {
        _ = try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
    }
}

@Test func gpt2TokenizerRejectsNonContiguousVocabularyIDs() throws {
    let directory = try makeTempDirectory()
    let vocabURL = directory.appendingPathComponent("vocab.json")
    let mergesURL = directory.appendingPathComponent("merges.txt")
    let data = try JSONSerialization.data(withJSONObject: ["a": 0, "b": 2], options: [.sortedKeys])
    try data.write(to: vocabURL)
    try "#version: 0.2\n".write(to: mergesURL, atomically: true, encoding: .utf8)

    #expect(throws: GPT2BPETokenizerError.nonContiguousTokenIDs(expectedCount: 2, actualMaxTokenID: 2)) {
        _ = try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
    }
}

@Test func sentencePieceTokenizerRejectsTruncatedModel() throws {
    let directory = try makeTempDirectory()
    let modelURL = directory.appendingPathComponent("tokenizer.model")
    try Data([0x01, 0x02]).write(to: modelURL)

    #expect(throws: SentencePieceTokenizerError.truncatedModel) {
        _ = try SentencePieceTokenizer(modelURL: modelURL)
    }
}

@Test func sentencePieceTokenizerRejectsTruncatedEntry() throws {
    let directory = try makeTempDirectory()
    let modelURL = directory.appendingPathComponent("tokenizer.model")
    var data = Data()
    append(Int32(8), to: &data)
    append(Float(0), to: &data)
    append(Int32(4), to: &data)
    data.append(contentsOf: Array("ab".utf8))
    try data.write(to: modelURL)

    #expect(throws: SentencePieceTokenizerError.truncatedEntry(index: 0)) {
        _ = try SentencePieceTokenizer(modelURL: modelURL)
    }
}

private func makeGPT2Fixture() throws -> (vocabURL: URL, mergesURL: URL) {
    let directory = try makeTempDirectory()
    let newlinePiece = String(gpt2ByteUnicodeMap()[10]!)
    let spacePiece = String(gpt2ByteUnicodeMap()[32]!)
    let vocab: [String: Int] = [
        "Hello": 0,
        ",": 1,
        "\(spacePiece)world": 2,
        "!": 3,
        newlinePiece: 4,
    ]
    let vocabData = try JSONSerialization.data(withJSONObject: vocab, options: [.sortedKeys])
    let vocabURL = directory.appendingPathComponent("vocab.json")
    try vocabData.write(to: vocabURL)

    let merges = [
        "#version: 0.2",
        "H e",
        "He l",
        "Hel l",
        "Hell o",
        "\(spacePiece) w",
        "\(spacePiece)w o",
        "\(spacePiece)wo r",
        "\(spacePiece)wor l",
        "\(spacePiece)worl d",
    ].joined(separator: "\n")
    let mergesURL = directory.appendingPathComponent("merges.txt")
    try merges.write(to: mergesURL, atomically: true, encoding: .utf8)
    return (vocabURL, mergesURL)
}

private func makeSentencePieceFixture() throws -> URL {
    let directory = try makeTempDirectory()
    let modelURL = directory.appendingPathComponent("tokenizer.model")
    let pieces: [(String, Float)] = [
        ("▁", 0.0),
        ("H", 0.0),
        ("e", 0.0),
        ("l", 0.0),
        ("o", 0.0),
        (",", 0.0),
        ("w", 0.0),
        ("r", 0.0),
        ("d", 0.0),
        ("!", 0.0),
        ("▁H", 1.0),
        ("▁He", 2.0),
        ("▁Hel", 3.0),
        ("▁Hell", 4.0),
        ("▁Hello", 5.0),
        ("▁w", 1.0),
        ("▁wo", 2.0),
        ("▁wor", 3.0),
        ("▁worl", 4.0),
        ("▁world", 5.0),
        ("<0x0A>", 0.0),
    ]

    var data = Data()
    append(Int32(16), to: &data)
    for (piece, score) in pieces {
        append(score, to: &data)
        let bytes = Array(piece.utf8)
        append(Int32(bytes.count), to: &data)
        data.append(contentsOf: bytes)
    }
    try data.write(to: modelURL)
    return modelURL
}

private func append(_ value: Int32, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}

private func append(_ value: Float, to data: inout Data) {
    var bits = value.bitPattern.littleEndian
    withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
}

private func makeTempDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func gpt2ByteUnicodeMap() -> [UInt8: UnicodeScalar] {
    var printable = Array(33...126) + Array(161...172) + Array(174...255)
    var mapped = printable
    let printableSet = Set(printable)
    var extra = 0
    for value in 0..<256 where !printableSet.contains(value) {
        printable.append(value)
        mapped.append(256 + extra)
        extra += 1
    }
    var result: [UInt8: UnicodeScalar] = [:]
    for (index, byte) in printable.enumerated() {
        result[UInt8(byte)] = UnicodeScalar(mapped[index])!
    }
    return result
}
