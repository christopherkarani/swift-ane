import Foundation
import Testing
import ANETypes
@testable import ModelSupport

@Test func blobWeightLoaderReadsHeaderAndDecodesFP16Payload() throws {
    let weights: [Float] = [1.0, -2.5, 3.25, 0.5]
    let blob = WeightBlob.build(from: weights, rows: 2, cols: 2)
    let path = try temporaryFile(named: "weights.bin", contents: blob)

    let header = try BlobWeightLoader.readHeader(from: path.path)
    #expect(header.magic == 0xDEADBEEF)
    #expect(header.dataOffset == 128)
    #expect(header.dataSize == UInt32(weights.count * 2))

    let decoded = try BlobWeightLoader.load(from: path.path)
    #expect(decoded.count == weights.count)
    for (lhs, rhs) in zip(decoded, weights) {
        #expect(abs(lhs - rhs) < 0.01)
    }
}

@Test func blobWeightLoaderRejectsInvalidMagic() throws {
    var bytes = [UInt8](repeating: 0, count: 128)
    bytes[72] = 2
    bytes[80] = 128
    let path = try temporaryFile(named: "invalid.bin", contents: Data(bytes))

    #expect(throws: BlobWeightLoaderError.invalidMagic(found: 0)) {
        try BlobWeightLoader.readHeader(from: path.path)
    }
}

@Test func blobWeightLoaderRejectsTruncatedPayload() throws {
    let blob = WeightBlob.build(from: [1.0, 2.0, 3.0, 4.0], rows: 2, cols: 2)
    let truncated = blob.prefix(blob.count - 2)
    let path = try temporaryFile(named: "truncated.bin", contents: Data(truncated))

    #expect(throws: BlobWeightLoaderError.truncatedPayload(expectedByteCount: 8, actualByteCount: 6)) {
        try BlobWeightLoader.load(from: path.path)
    }
}

private func temporaryFile(named name: String, contents: Data) throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let path = directory.appendingPathComponent(name)
    try contents.write(to: path)
    return path
}
