import Foundation

public enum BlobWeightLoaderError: Error, Sendable, Equatable {
    case truncatedHeader(actualByteCount: Int)
    case invalidMagic(found: UInt32)
    case invalidDataOffset(found: UInt32)
    case truncatedPayload(expectedByteCount: Int, actualByteCount: Int)
    case invalidPayloadSize(Int)
}

public enum BlobWeightLoader {
    public struct Header: Sendable, Equatable {
        public let magic: UInt32
        public let dataSize: UInt32
        public let dataOffset: UInt32
    }

    public static func readHeader(from path: String) throws -> Header {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        let data = try handle.read(upToCount: 128) ?? Data()
        return try parseHeader(from: data)
    }

    public static func load(from path: String) throws -> [Float] {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }

        let headerData = try handle.read(upToCount: 128) ?? Data()
        let header = try parseHeader(from: headerData)
        guard header.dataSize.isMultiple(of: 2) else {
            throw BlobWeightLoaderError.invalidPayloadSize(Int(header.dataSize))
        }

        try handle.seek(toOffset: UInt64(header.dataOffset))
        let payload = try handle.read(upToCount: Int(header.dataSize)) ?? Data()
        guard payload.count == Int(header.dataSize) else {
            throw BlobWeightLoaderError.truncatedPayload(
                expectedByteCount: Int(header.dataSize),
                actualByteCount: payload.count
            )
        }

        var values: [Float] = []
        values.reserveCapacity(payload.count / MemoryLayout<UInt16>.stride)
        for index in stride(from: 0, to: payload.count, by: MemoryLayout<UInt16>.stride) {
            let bits = payload.withUnsafeBytes { raw in
                raw.loadUnaligned(
                    fromByteOffset: index,
                    as: UInt16.self
                )
            }
            values.append(Float(Float16(bitPattern: UInt16(littleEndian: bits))))
        }
        return values
    }

    static func parseHeader(from data: Data) throws -> Header {
        guard data.count >= 128 else {
            throw BlobWeightLoaderError.truncatedHeader(actualByteCount: data.count)
        }

        let magic = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 64, as: UInt32.self)
        }
        let dataSize = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 72, as: UInt32.self)
        }
        let dataOffset = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 80, as: UInt32.self)
        }

        let normalizedMagic = UInt32(littleEndian: magic)
        guard normalizedMagic == 0xDEADBEEF else {
            throw BlobWeightLoaderError.invalidMagic(found: normalizedMagic)
        }

        let normalizedOffset = UInt32(littleEndian: dataOffset)
        guard normalizedOffset == 128 else {
            throw BlobWeightLoaderError.invalidDataOffset(found: normalizedOffset)
        }

        return Header(
            magic: normalizedMagic,
            dataSize: UInt32(littleEndian: dataSize),
            dataOffset: normalizedOffset
        )
    }
}
