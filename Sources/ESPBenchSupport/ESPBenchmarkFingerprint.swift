public enum ESPBenchmarkMetric: String, Sendable, Hashable, CaseIterable {
    case ttftMilliseconds = "ttft_ms"
    case tokensPerSecond = "tokens_per_second"
    case coldLoadMilliseconds = "cold_load_ms"
    case warmLoadMilliseconds = "warm_load_ms"
    case compileCacheHitRate = "compile_cache_hit_rate"
    case peakResidentMemoryBytes = "peak_resident_memory_bytes"

    public static let requiredCoreMetrics: Set<ESPBenchmarkMetric> = [
        .ttftMilliseconds,
        .tokensPerSecond,
        .coldLoadMilliseconds,
        .warmLoadMilliseconds,
        .compileCacheHitRate,
        .peakResidentMemoryBytes,
    ]
}

public struct ESPBenchmarkFingerprint: Sendable, Equatable {
    public let metrics: [ESPBenchmarkMetric: Double]

    public init(metrics: [ESPBenchmarkMetric: Double]) {
        self.metrics = metrics
    }

    public func missingRequiredMetrics() -> Set<ESPBenchmarkMetric> {
        ESPBenchmarkMetric.requiredCoreMetrics.subtracting(metrics.keys)
    }
}
