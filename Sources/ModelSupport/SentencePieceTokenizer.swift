import Foundation

public enum SentencePieceTokenizerError: Error, Sendable, Equatable {
    case truncatedModel
    case invalidMaxTokenLength(Int)
    case truncatedEntry(index: Int)
    case invalidUTF8Entry(index: Int)
    case emptyVocabulary
}

public struct SentencePieceTokenizer: Tokenizer {
    public let vocabSize: Int

    private let pieces: [String]
    private let scores: [Float]
    private let pieceToID: [String: Int]
    private let maxTokenLength: Int

    public init(modelURL: URL) throws {
        let data = try Data(contentsOf: modelURL)
        guard data.count >= 4 else {
            throw SentencePieceTokenizerError.truncatedModel
        }

        var cursor = 0
        self.maxTokenLength = Int(data.readLittleEndian(Int32.self, at: &cursor))
        guard maxTokenLength > 0 else {
            throw SentencePieceTokenizerError.invalidMaxTokenLength(maxTokenLength)
        }

        var pieces: [String] = []
        var scores: [Float] = []
        var pieceToID: [String: Int] = [:]
        var entryIndex = 0

        while cursor < data.count {
            guard cursor + 8 <= data.count else {
                throw SentencePieceTokenizerError.truncatedEntry(index: entryIndex)
            }
            let score = data.readLittleEndian(Float.self, at: &cursor)
            let length = Int(data.readLittleEndian(Int32.self, at: &cursor))
            guard length >= 0, cursor + length <= data.count else {
                throw SentencePieceTokenizerError.truncatedEntry(index: entryIndex)
            }
            let pieceData = data.subdata(in: cursor..<(cursor + length))
            cursor += length
            guard let piece = String(data: pieceData, encoding: .utf8) else {
                throw SentencePieceTokenizerError.invalidUTF8Entry(index: entryIndex)
            }

            let id = pieces.count
            pieces.append(piece)
            scores.append(score)
            pieceToID[piece] = id
            entryIndex += 1
        }

        guard !pieces.isEmpty else {
            throw SentencePieceTokenizerError.emptyVocabulary
        }

        self.pieces = pieces
        self.scores = scores
        self.pieceToID = pieceToID
        self.vocabSize = pieces.count
    }

    public func encode(_ text: String) -> [Int] {
        let normalized = "▁" + text.replacingOccurrences(of: " ", with: "▁")
        var tokens: [Int] = []

        for character in normalized {
            let piece = String(character)
            if let id = pieceToID[piece] {
                tokens.append(id)
                continue
            }

            for byte in piece.utf8 {
                let fallback = String(format: "<0x%02X>", byte)
                if let id = pieceToID[fallback] {
                    tokens.append(id)
                }
            }
        }

        while tokens.count >= 2 {
            var bestScore = -Float.greatestFiniteMagnitude
            var bestIndex: Int?
            var bestID: Int?

            for index in 0..<(tokens.count - 1) {
                let merged = pieces[tokens[index]] + pieces[tokens[index + 1]]
                guard merged.utf8.count <= maxTokenLength, let id = pieceToID[merged] else {
                    continue
                }

                let score = scores[id]
                if score > bestScore {
                    bestScore = score
                    bestIndex = index
                    bestID = id
                }
            }

            guard let bestIndex, let bestID else { break }
            tokens[bestIndex] = bestID
            tokens.remove(at: bestIndex + 1)
        }

        return tokens
    }

    public func decode(_ tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        var text = ""

        for token in tokens where pieces.indices.contains(token) {
            let piece = pieces[token]
            if let byte = Self.byteToken(piece) {
                bytes.append(byte)
            } else {
                if !bytes.isEmpty {
                    text += String(decoding: bytes, as: UTF8.self)
                    bytes.removeAll(keepingCapacity: true)
                }
                text += piece
            }
        }

        if !bytes.isEmpty {
            text += String(decoding: bytes, as: UTF8.self)
        }

        let restored = text.replacingOccurrences(of: "▁", with: " ")
        if restored.hasPrefix(" ") {
            return String(restored.dropFirst())
        }
        return restored
    }

    private static func byteToken(_ piece: String) -> UInt8? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">") else {
            return nil
        }
        let start = piece.index(piece.startIndex, offsetBy: 3)
        let end = piece.index(before: piece.endIndex)
        return UInt8(piece[start..<end], radix: 16)
    }
}

private extension Data {
    func readLittleEndian<T>(_ type: T.Type, at cursor: inout Int) -> T where T: FixedWidthInteger {
        let value = withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: cursor, as: T.self)
        }
        cursor += MemoryLayout<T>.size
        return T(littleEndian: value)
    }

    func readLittleEndian(_ type: Float.Type, at cursor: inout Int) -> Float {
        let value = withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: cursor, as: UInt32.self)
        }
        cursor += MemoryLayout<UInt32>.size
        return Float(bitPattern: UInt32(littleEndian: value))
    }
}
