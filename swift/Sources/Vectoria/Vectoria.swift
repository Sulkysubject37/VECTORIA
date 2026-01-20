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
typealias GraphAddOpBiasAddFn = @convention(c) (GraphHandle?, Int32, Int32) -> Int32
typealias GraphAddOpReluFn = @convention(c) (GraphHandle?, Int32) -> Int32
typealias GraphAddOpAddFn = @convention(c) (GraphHandle?, Int32, Int32) -> Int32
typealias GraphAddOpMulFn = @convention(c) (GraphHandle?, Int32, Int32) -> Int32
typealias GraphAddOpReduceSumFn = @convention(c) (GraphHandle?, Int32) -> Int32
typealias GraphAddSoftmaxFn = @convention(c) (GraphHandle?, Int32) -> Int32
typealias GraphAddSoftmaxStableFn = @convention(c) (GraphHandle?, Int32) -> Int32
typealias GraphAddLogSoftmaxFn = @convention(c) (GraphHandle?, Int32) -> Int32
typealias GraphAddCrossEntropyFn = @convention(c) (GraphHandle?, Int32, Int32) -> Int32
typealias GraphAddLayerNormFn = @convention(c) (GraphHandle?, Int32, Int32, Int32) -> Int32
typealias GraphSetOutputFn = @convention(c) (GraphHandle?, Int32) -> Void
typealias GraphExportCoreMLFn = @convention(c) (GraphHandle?, UnsafePointer<Int8>) -> Int32
typealias EngineCreateWithPolicyFn = @convention(c) (GraphHandle?, Int32) -> EngineHandle?
typealias EngineDestroyFn = @convention(c) (EngineHandle?) -> Void
typealias EngineCompileFn = @convention(c) (EngineHandle?) -> Void
typealias EngineExecuteFn = @convention(c) (EngineHandle?) -> Void
typealias EngineGetBufferFn = @convention(c) (EngineHandle?, Int32) -> UnsafeMutableRawPointer?
typealias EngineGetTraceSizeFn = @convention(c) (EngineHandle?) -> Int
typealias EngineGetTraceEventFn = @convention(c) (EngineHandle?, Int, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<UInt64>, UnsafeMutablePointer<Int64>, UnsafeMutablePointer<Int8>, Int) -> Void

public class VectoriaRuntime {
    internal let library: UnsafeMutableRawPointer
    
    private let graphCreate: GraphCreateFn
    private let graphDestroy: GraphDestroyFn
    private let graphAddInput: GraphAddInputFn
    private let graphAddOpMatMul: GraphAddOpMatMulFn
    private let graphAddOpBiasAdd: GraphAddOpBiasAddFn
    private let graphAddOpRelu: GraphAddOpReluFn
    private let graphAddOpAdd: GraphAddOpAddFn
    private let graphAddOpMul: GraphAddOpMulFn
    private let graphAddOpReduceSum: GraphAddOpReduceSumFn
    private let graphAddSoftmax: GraphAddSoftmaxFn
    private let graphAddSoftmaxStable: GraphAddSoftmaxStableFn
    private let graphAddLogSoftmax: GraphAddLogSoftmaxFn
    private let graphAddCrossEntropy: GraphAddCrossEntropyFn
    private let graphAddLayerNorm: GraphAddLayerNormFn
    private let graphSetOutput: GraphSetOutputFn
    private let graphExportCoreML: GraphExportCoreMLFn
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
        graphAddOpBiasAdd = load("vectoria_graph_add_op_bias_add")
        graphAddOpRelu = load("vectoria_graph_add_op_relu")
        graphAddOpAdd = load("vectoria_graph_add_op_add")
        graphAddOpMul = load("vectoria_graph_add_op_mul")
        graphAddOpReduceSum = load("vectoria_graph_add_op_reduce_sum")
        graphAddSoftmax = load("vectoria_graph_add_softmax")
        graphAddSoftmaxStable = load("vectoria_graph_add_softmax_stable")
        graphAddLogSoftmax = load("vectoria_graph_add_logsoftmax")
        graphAddCrossEntropy = load("vectoria_graph_add_crossentropy")
        graphAddLayerNorm = load("vectoria_graph_add_layernorm")
        graphSetOutput = load("vectoria_graph_set_output")
        graphExportCoreML = load("vectoria_export_coreml")
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

    internal func addOpBiasAdd(_ handle: GraphHandle?, input: Int32, bias: Int32) -> Int32 {
        return graphAddOpBiasAdd(handle, input, bias)
    }

    internal func addOpRelu(_ handle: GraphHandle?, input: Int32) -> Int32 {
        return graphAddOpRelu(handle, input)
    }

    internal func addOpAdd(_ handle: GraphHandle?, inputA: Int32, inputB: Int32) -> Int32 {
        return graphAddOpAdd(handle, inputA, inputB)
    }

    internal func addOpMul(_ handle: GraphHandle?, inputA: Int32, inputB: Int32) -> Int32 {
        return graphAddOpMul(handle, inputA, inputB)
    }

    internal func addOpReduceSum(_ handle: GraphHandle?, input: Int32) -> Int32 {
        return graphAddOpReduceSum(handle, input)
    }

    internal func addSoftmax(_ handle: GraphHandle?, input: Int32) -> Int32 {
        return graphAddSoftmax(handle, input)
    }

    internal func addSoftmaxStable(_ handle: GraphHandle?, input: Int32) -> Int32 {
        return graphAddSoftmaxStable(handle, input)
    }

    internal func addLogSoftmax(_ handle: GraphHandle?, input: Int32) -> Int32 {
        return graphAddLogSoftmax(handle, input)
    }

    internal func addCrossEntropy(_ handle: GraphHandle?, logits: Int32, target: Int32) -> Int32 {
        return graphAddCrossEntropy(handle, logits, target)
    }

    internal func addLayerNorm(_ handle: GraphHandle?, input: Int32, gamma: Int32, beta: Int32) -> Int32 {
        return graphAddLayerNorm(handle, input, gamma, beta)
    }
    
    internal func setOutput(_ handle: GraphHandle?, nodeId: Int32) {
        graphSetOutput(handle, nodeId)
    }

    internal func exportToCoreML(_ handle: GraphHandle?, path: String) throws {
        let result = graphExportCoreML(handle, path)
        if result != 0 {
            throw VectoriaError.libraryLoadFailed("Export failed") // Reuse error or add new one
        }
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

    public func addOpBiasAdd(input: Int32, bias: Int32) -> Int32 {
        return runtime.addOpBiasAdd(handle, input: input, bias: bias)
    }

    public func addOpRelu(input: Int32) -> Int32 {
        return runtime.addOpRelu(handle, input: input)
    }

    public func addOpAdd(inputA: Int32, inputB: Int32) -> Int32 {
        return runtime.addOpAdd(handle, inputA: inputA, inputB: inputB)
    }

    public func addOpMul(inputA: Int32, inputB: Int32) -> Int32 {
        return runtime.addOpMul(handle, inputA: inputA, inputB: inputB)
    }

    public func addOpReduceSum(input: Int32) -> Int32 {
        return runtime.addOpReduceSum(handle, input: input)
    }

    public func addSoftmax(input: Int32) -> Int32 {
        return runtime.addSoftmax(handle, input: input)
    }

    public func addSoftmaxStable(input: Int32) -> Int32 {
        return runtime.addSoftmaxStable(handle, input: input)
    }

    public func addLogSoftmax(input: Int32) -> Int32 {
        return runtime.addLogSoftmax(handle, input: input)
    }

    public func addCrossEntropy(logits: Int32, target: Int32) -> Int32 {
        return runtime.addCrossEntropy(handle, logits: logits, target: target)
    }

    public func addLayerNorm(input: Int32, gamma: Int32, beta: Int32) -> Int32 {
        return runtime.addLayerNorm(handle, input: input, gamma: gamma, beta: beta)
    }
    
    public func setOutput(nodeId: Int32) {
        runtime.setOutput(handle, nodeId: nodeId)
    }

    public func exportToCoreML(path: String) throws {
        try runtime.exportToCoreML(handle, path: path)
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
