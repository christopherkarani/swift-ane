import Foundation

public enum GPT2BPETokenizerError: Error, Sendable, Equatable {
    case malformedVocabulary
    case emptyVocabulary
    case nonContiguousTokenIDs(expectedCount: Int, actualMaxTokenID: Int)
}

public final class GPT2BPETokenizer: Tokenizer, @unchecked Sendable {
    public let vocabSize: Int

    private let encoder: [String: Int]
    private let decoder: [String]
    private let bpeRanks: [TokenPair: Int]
    private let byteToUnicode: [UnicodeScalar]
    private let unicodeToByte: [UInt32: UInt8]
    private let preTokenizer: NSRegularExpression
    private let cacheLock = NSLock()
    private var cache: [String: [String]] = [:]

    public init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        let rawObject = try JSONSerialization.jsonObject(with: vocabData)
        guard let rawVocab = rawObject as? [String: Any] else {
            throw GPT2BPETokenizerError.malformedVocabulary
        }
        guard !rawVocab.isEmpty else {
            throw GPT2BPETokenizerError.emptyVocabulary
        }

        var encoder: [String: Int] = [:]
        encoder.reserveCapacity(rawVocab.count)
        for (token, value) in rawVocab {
            guard let number = value as? NSNumber else {
                throw GPT2BPETokenizerError.malformedVocabulary
            }
            encoder[token] = number.intValue
        }
        self.encoder = encoder

        let maxTokenID = encoder.values.max() ?? -1
        guard maxTokenID >= 0 else {
            throw GPT2BPETokenizerError.emptyVocabulary
        }
        guard maxTokenID + 1 == encoder.count else {
            throw GPT2BPETokenizerError.nonContiguousTokenIDs(
                expectedCount: encoder.count,
                actualMaxTokenID: maxTokenID
            )
        }
        var decoder = Array(repeating: "", count: maxTokenID + 1)
        for (token, id) in encoder where decoder.indices.contains(id) {
            decoder[id] = token
        }
        self.decoder = decoder
        self.vocabSize = decoder.count

        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
        var bpeRanks: [TokenPair: Int] = [:]
        var rank = 0
        for line in merges.split(separator: "\n").map(String.init) {
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }
            let pieces = line.split(separator: " ", omittingEmptySubsequences: true)
            guard pieces.count == 2 else { continue }
            bpeRanks[TokenPair(String(pieces[0]), String(pieces[1]))] = rank
            rank += 1
        }
        self.bpeRanks = bpeRanks

        let mapping = Self.makeByteMapping()
        self.byteToUnicode = mapping.byteToUnicode
        self.unicodeToByte = mapping.unicodeToByte
        self.preTokenizer = try NSRegularExpression(
            pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+"
        )
    }

    public func encode(_ text: String) -> [Int] {
        let nsText = text as NSString
        let matches = preTokenizer.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        var tokens: [Int] = []

        for match in matches {
            let chunk = nsText.substring(with: match.range)
            let encoded = encodeBytes(of: chunk)
            for piece in bpe(encoded) {
                guard let token = encoder[piece] else { continue }
                tokens.append(token)
            }
        }

        return tokens
    }

    public func decode(_ tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        for token in tokens where decoder.indices.contains(token) {
            let piece = decoder[token]
            for scalar in piece.unicodeScalars {
                if let byte = unicodeToByte[scalar.value] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: String(scalar).utf8)
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodeBytes(of text: String) -> String {
        var result = String.UnicodeScalarView()
        result.reserveCapacity(text.utf8.count)
        for byte in text.utf8 {
            result.append(byteToUnicode[Int(byte)])
        }
        return String(result)
    }

    private func bpe(_ token: String) -> [String] {
        cacheLock.lock()
        if let cached = cache[token] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        var word = token.unicodeScalars.map { String($0) }
        guard word.count > 1 else {
            cacheLock.lock()
            cache[token] = word
            cacheLock.unlock()
            return word
        }

        while true {
            let pairs = Self.pairs(in: word)
            guard !pairs.isEmpty else { break }

            var bestPair: TokenPair?
            var bestRank = Int.max
            for pair in pairs {
                guard let rank = bpeRanks[pair], rank < bestRank else { continue }
                bestPair = pair
                bestRank = rank
            }

            guard let bestPair else { break }

            var merged: [String] = []
            var index = 0
            while index < word.count {
                if index < word.count - 1 &&
                    word[index] == bestPair.left &&
                    word[index + 1] == bestPair.right {
                    merged.append(bestPair.left + bestPair.right)
                    index += 2
                } else {
                    merged.append(word[index])
                    index += 1
                }
            }
            word = merged
            if word.count == 1 {
                break
            }
        }

        cacheLock.lock()
        cache[token] = word
        cacheLock.unlock()
        return word
    }

    private static func pairs(in word: [String]) -> [TokenPair] {
        guard word.count > 1 else { return [] }
        return zip(word, word.dropFirst()).map(TokenPair.init)
    }

    private static func makeByteMapping() -> (byteToUnicode: [UnicodeScalar], unicodeToByte: [UInt32: UInt8]) {
        var direct: [UInt8] = []
        direct.append(contentsOf: 33...126)
        direct.append(contentsOf: 161...172)
        direct.append(contentsOf: 174...255)

        var byteToUnicode = Array(repeating: UnicodeScalar(0)!, count: 256)
        for byte in direct {
            byteToUnicode[Int(byte)] = UnicodeScalar(Int(byte))!
        }

        var nextScalar = 256
        for byte in 0..<256 where byteToUnicode[byte].value == 0 {
            byteToUnicode[byte] = UnicodeScalar(nextScalar)!
            nextScalar += 1
        }

        var unicodeToByte: [UInt32: UInt8] = [:]
        for (byte, scalar) in byteToUnicode.enumerated() {
            unicodeToByte[scalar.value] = UInt8(byte)
        }

        return (byteToUnicode, unicodeToByte)
    }
}

private struct TokenPair: Hashable {
    let left: String
    let right: String

    init(_ left: String, _ right: String) {
        self.left = left
        self.right = right
    }

    init(_ values: (String, String)) {
        self.init(values.0, values.1)
    }
}
