/// Vocabulary index type for token IDs throughout the Espresso pipeline.
///
/// `UInt32` accommodates large-vocabulary models (>65 535 tokens) while remaining
/// compact and compatible with `Int` conversion via `exactly:`.
public typealias TokenID = UInt32
