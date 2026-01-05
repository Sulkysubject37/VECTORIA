import Foundation

public typealias GraphHandle = OpaquePointer
public typealias EngineHandle = OpaquePointer

public enum DataType: Int {
    case float32 = 0
    case float16 = 1
    case int32 = 2
    case int8 = 3
}

public enum KernelPolicy: Int {
    case reference = 0
    case simd = 1
}

public struct TraceEvent {
    public let type: Int
    public let timestamp: UInt64
    public let nodeId: Int64
    public let details: String
}

// Function Pointer Typedefs
typealias GraphCreateFn = @convention(c) () -> GraphHandle?
typealias GraphDestroyFn = @convention(c) (GraphHandle?) -> Void
typealias GraphAddInputFn = @convention(c) (GraphHandle?, UnsafePointer<Int8>, UnsafePointer<Int64>, Int32, Int32) -> Int32
typealias GraphAddOpMatMulFn = @convention(c) (GraphHandle?, Int32, Int32) -> Int32
typealias GraphSetOutputFn = @convention(c) (GraphHandle?, Int32) -> Void
typealias EngineCreateWithPolicyFn = @convention(c) (GraphHandle?, Int32) -> EngineHandle?
typealias EngineDestroyFn = @convention(c) (EngineHandle?) -> Void
typealias EngineCompileFn = @convention(c) (EngineHandle?) -> Void
typealias EngineExecuteFn = @convention(c) (EngineHandle?) -> Void
typealias EngineGetBufferFn = @convention(c) (EngineHandle?, Int32) -> UnsafeMutableRawPointer?
typealias EngineGetTraceSizeFn = @convention(c) (EngineHandle?) -> Int
typealias EngineGetTraceEventFn = @convention(c) (EngineHandle?, Int, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<UInt64>, UnsafeMutablePointer<Int64>, UnsafeMutablePointer<Int8>, Int) -> Void

public class VectoriaRuntime {
    private let library: UnsafeMutableRawPointer
    
    private let graphCreate: GraphCreateFn
    private let graphDestroy: GraphDestroyFn
    private let graphAddInput: GraphAddInputFn
    private let graphAddOpMatMul: GraphAddOpMatMulFn
    private let graphSetOutput: GraphSetOutputFn
    private let engineCreateWithPolicy: EngineCreateWithPolicyFn
    private let engineDestroy: EngineDestroyFn
    private let engineCompile: EngineCompileFn
    private let engineExecute: EngineExecuteFn
    private let engineGetBuffer: EngineGetBufferFn
    private let engineGetTraceSize: EngineGetTraceSizeFn
    private let engineGetTraceEvent: EngineGetTraceEventFn

    public init(libraryPath: String = "libvectoria.dylib") throws {
        guard let lib = dlopen(libraryPath, RTLD_NOW) else {
            throw VectoriaError.libraryLoadFailed(String(cString: dlerror()))
        }
        self.library = lib
        
        func load<T>(_ name: String) -> T {
            let sym = dlsym(lib, name)
            return unsafeBitCast(sym, to: T.self)
        }
        
        graphCreate = load("vectoria_graph_create")
        graphDestroy = load("vectoria_graph_destroy")
        graphAddInput = load("vectoria_graph_add_input")
        graphAddOpMatMul = load("vectoria_graph_add_op_matmul")
        graphSetOutput = load("vectoria_graph_set_output")
        engineCreateWithPolicy = load("vectoria_engine_create_with_policy")
        engineDestroy = load("vectoria_engine_destroy")
        engineCompile = load("vectoria_engine_compile")
        engineExecute = load("vectoria_engine_execute")
        engineGetBuffer = load("vectoria_engine_get_buffer")
        engineGetTraceSize = load("vectoria_engine_get_trace_size")
        engineGetTraceEvent = load("vectoria_engine_get_trace_event")
    }
    
    public func createGraph() -> VectoriaGraph {
        return VectoriaGraph(handle: graphCreate(), runtime: self)
    }
    
    internal func destroyGraph(_ handle: GraphHandle?) {
        graphDestroy(handle)
    }
    
    internal func addInput(_ handle: GraphHandle?, name: String, shape: [Int64], dtype: DataType) -> Int32 {
        var cShape = shape
        return graphAddInput(handle, name, &cShape, Int32(shape.count), Int32(dtype.rawValue))
    }
    
    internal func addOpMatMul(_ handle: GraphHandle?, inputA: Int32, inputB: Int32) -> Int32 {
        return graphAddOpMatMul(handle, inputA, inputB)
    }
    
    internal func setOutput(_ handle: GraphHandle?, nodeId: Int32) {
        graphSetOutput(handle, nodeId)
    }
    
    public func createEngine(graph: VectoriaGraph, policy: KernelPolicy = .reference) -> VectoriaEngine {
        let handle = engineCreateWithPolicy(graph.handle, Int32(policy.rawValue))
        return VectoriaEngine(handle: handle, runtime: self)
    }
    
    internal func destroyEngine(_ handle: EngineHandle?) {
        engineDestroy(handle)
    }
    
    internal func compileEngine(_ handle: EngineHandle?) {
        engineCompile(handle)
    }
    
    internal func executeEngine(_ handle: EngineHandle?) {
        engineExecute(handle)
    }
    
    internal func getBuffer(_ handle: EngineHandle?, nodeId: Int32) -> UnsafeMutableRawPointer? {
        return engineGetBuffer(handle, nodeId)
    }
    
    internal func getTrace(_ handle: EngineHandle?) -> [TraceEvent] {
        let count = engineGetTraceSize(handle)
        var events: [TraceEvent] = []
        
        var type: Int32 = 0
        var ts: UInt64 = 0
        var nid: Int64 = 0
        let bufLen = 256
        let buf = UnsafeMutablePointer<Int8>.allocate(capacity: bufLen)
        defer { buf.deallocate() }
        
        for i in 0..<count {
            engineGetTraceEvent(handle, i, &type, &ts, &nid, buf, bufLen)
            let details = String(cString: buf)
            events.append(TraceEvent(type: Int(type), timestamp: ts, nodeId: nid, details: details))
        }
        
        return events
    }
}

public class VectoriaGraph {
    internal let handle: GraphHandle?
    private let runtime: VectoriaRuntime
    
    internal init(handle: GraphHandle?, runtime: VectoriaRuntime) {
        self.handle = handle
        self.runtime = runtime
    }
    
    deinit {
        runtime.destroyGraph(handle)
    }
    
    public func addInput(name: String, shape: [Int64], dtype: DataType) -> Int32 {
        return runtime.addInput(handle, name: name, shape: shape, dtype: dtype)
    }
    
    public func addOpMatMul(inputA: Int32, inputB: Int32) -> Int32 {
        return runtime.addOpMatMul(handle, inputA: inputA, inputB: inputB)
    }
    
    public func setOutput(nodeId: Int32) {
        runtime.setOutput(handle, nodeId: nodeId)
    }
}

public class VectoriaEngine {
    private let handle: EngineHandle?
    private let runtime: VectoriaRuntime
    
    internal init(handle: EngineHandle?, runtime: VectoriaRuntime) {
        self.handle = handle
        self.runtime = runtime
    }
    
    deinit {
        runtime.destroyEngine(handle)
    }
    
    public func compile() {
        runtime.compileEngine(handle)
    }
    
    public func execute() {
        runtime.executeEngine(handle)
    }
    
    public func getBuffer(nodeId: Int32) -> UnsafeMutableRawPointer? {
        return runtime.getBuffer(handle, nodeId: nodeId)
    }
    
    public func getTrace() -> [TraceEvent] {
        return runtime.getTrace(handle)
    }
}

public enum VectoriaError: Error {
    case libraryLoadFailed(String)
    case graphFrozen
    case invalidShape
}
