import Testing
@testable import ESPBenchSupport

@Test func benchmarkFingerprintDetectsMissingRequiredMetrics() {
    let fingerprint = ESPBenchmarkFingerprint(
        metrics: [
            .ttftMilliseconds: 12.5,
            .tokensPerSecond: 84.0,
        ]
    )

    let missing = fingerprint.missingRequiredMetrics()
    #expect(missing.contains(.coldLoadMilliseconds))
    #expect(missing.contains(.warmLoadMilliseconds))
    #expect(missing.contains(.compileCacheHitRate))
}

@Test func benchmarkFingerprintIsValidWhenAllCoreMetricsExist() {
    let fingerprint = ESPBenchmarkFingerprint(
        metrics: [
            .ttftMilliseconds: 12.5,
            .tokensPerSecond: 84.0,
            .coldLoadMilliseconds: 180.0,
            .warmLoadMilliseconds: 24.0,
            .compileCacheHitRate: 1.0,
            .peakResidentMemoryBytes: 1_024,
        ]
    )

    #expect(fingerprint.missingRequiredMetrics().isEmpty)
}
