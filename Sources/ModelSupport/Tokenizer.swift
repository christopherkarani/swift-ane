public protocol Tokenizer: Sendable {
    func encode(_ text: String) -> [Int]
    func decode(_ tokens: [Int]) -> String
    var vocabSize: Int { get }
}
