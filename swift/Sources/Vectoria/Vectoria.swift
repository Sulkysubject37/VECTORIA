import Foundation

/// A handle to the C++ Graph object (opaque pointer)
public typealias GraphHandle = OpaquePointer

/// Supported data types mirroring the C++ enum
public enum DataType: Int {
    case float32 = 0
    case float16 = 1
    case int32 = 2
    case int8 = 3
}

/// Main entry point for Vectoria Swift bindings
public class VectoriaGraph {
    private var handle: GraphHandle?
    private var frozen: Bool = false
    
    public init() {
        // In a real implementation, this would call vectoria_graph_create()
        // handle = vectoria_graph_create()
        print("VectoriaGraph initialized (stub)")
    }
    
    deinit {
        // vectoria_graph_destroy(handle)
    }
    
    public func addInput(name: String, shape: [Int], dtype: DataType) throws -> Int {
        guard !frozen else {
            throw VectoriaError.graphFrozen
        }
        // vectoria_graph_add_input(...)
        return 0 // Stub ID
    }
    
    public func compile() {
        frozen = true
        // vectoria_graph_compile(handle)
        print("VectoriaGraph compiled (stub)")
    }
}

public enum VectoriaError: Error {
    case graphFrozen
    case invalidShape
}