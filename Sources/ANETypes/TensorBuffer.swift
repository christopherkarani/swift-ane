import Foundation

public struct TensorBuffer: ~Copyable {
    public static let allocationAlignment: Int = 64

    public let count: Int
    private let rawStorage: UnsafeMutableRawPointer
    private let baseAddress: UnsafeMutablePointer<Float>
    private let ownsMemory: Bool

    @inline(__always)
    private static func byteCount(for elementCount: Int) -> Int {
        let bytes = elementCount.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        precondition(!bytes.overflow)
        return bytes.partialValue
    }

    public init(count: Int, zeroed: Bool) {
        precondition(count >= 0)
        self.count = count
        self.ownsMemory = true

        let requestedBytes = Self.byteCount(for: count)
        let allocatedBytes = max(requestedBytes, 1)

        self.rawStorage = UnsafeMutableRawPointer.allocate(
            byteCount: allocatedBytes,
            alignment: Self.allocationAlignment
        )
        self.baseAddress = rawStorage.bindMemory(to: Float.self, capacity: count)
        if zeroed {
            rawStorage.initializeMemory(as: UInt8.self, repeating: 0, count: requestedBytes)
        }
    }

    /// Creates a non-owning view into an existing buffer's memory.
    /// The caller must guarantee the source buffer outlives this view.
    @inline(__always)
    public init(nonOwningViewOf source: borrowing TensorBuffer) {
        self.count = source.count
        self.rawStorage = source.rawStorage
        self.baseAddress = source.baseAddress
        self.ownsMemory = false
    }

    deinit {
        if ownsMemory {
            rawStorage.deallocate()
        }
    }

    @inline(__always)
    public func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R {
        try body(baseAddress)
    }

    @inline(__always)
    public func withUnsafePointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafePointer(baseAddress))
    }

    @inline(__always)
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafeMutablePointer { ptr in
            try body(UnsafeMutableBufferPointer(start: ptr, count: count))
        }
    }

    @inline(__always)
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafePointer { ptr in
            try body(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    public func zero() {
        rawStorage.initializeMemory(as: UInt8.self, repeating: 0, count: Self.byteCount(for: count))
    }
}
