import Foundation

public typealias GraphHandle = OpaquePointer
public typealias EngineHandle = OpaquePointer

public enum DataType: Int {
    case float32 = 0
    case float16 = 1
    case int32 = 2
    case int8 = 3
}

public enum KernelPolicy {
    case reference
    case simd
}

public struct TraceEvent {
    public let type: Int
    public let timestamp: UInt64
    public let nodeId: Int64
    public let details: String
}

public class VectoriaGraph {
    private var handle: GraphHandle?
    private var frozen: Bool = false
    
    public init() {
        print("VectoriaGraph initialized (stub)")
    }
    
    deinit {}
    
    public func addInput(name: String, shape: [Int], dtype: DataType) throws -> Int {
        return 0 
    }
    
    public func compile() {
        frozen = true
    }
}

public class VectoriaEngine {
    private var handle: EngineHandle?
    
    public init(graph: VectoriaGraph, policy: KernelPolicy = .reference) {
        print("VectoriaEngine initialized (stub) with policy \(policy)")
    }
    
    public func execute() {}
    
    public func getTrace() -> [TraceEvent] {
        return [] // Stub
    }
}

public enum VectoriaError: Error {
    case graphFrozen
    case invalidShape
}
