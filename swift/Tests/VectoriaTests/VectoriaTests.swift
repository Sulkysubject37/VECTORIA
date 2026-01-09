import XCTest
@testable import Vectoria

final class VectoriaTests: XCTestCase {
    var runtime: VectoriaRuntime!
    
    override func setUpWithError() throws {
        // Assume libvectoria.dylib is in the current working directory or provided path
        // For local tests, we'll try to find it.
        runtime = try VectoriaRuntime(libraryPath: "libvectoria.dylib")
    }
    
    func testMatMul() throws {
        let graph = runtime.createGraph()
        let x = graph.addInput(name: "X", shape: [2, 2], dtype: .float32)
        let w = graph.addInput(name: "W", shape: [2, 2], dtype: .float32)
        let op = graph.addOpMatMul(inputA: x, inputB: w)
        graph.setOutput(nodeId: op)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let xPtr = engine.getBuffer(nodeId: x)!.assumingMemoryBound(to: Float.self)
        let wPtr = engine.getBuffer(nodeId: w)!.assumingMemoryBound(to: Float.self)
        
        for i in 0..<4 { xPtr[i] = 1.0 } // [1, 1, 1, 1]
        // Identity
        for i in 0..<4 { wPtr[i] = 0.0 }
        wPtr[0] = 1.0; wPtr[3] = 1.0
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: op)!.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(outPtr[0], 1.0)
        XCTAssertEqual(outPtr[1], 1.0)
        XCTAssertEqual(outPtr[2], 1.0)
        XCTAssertEqual(outPtr[3], 1.0)
    }
    
    func testBiasAdd() throws {
        let graph = runtime.createGraph()
        let x = graph.addInput(name: "X", shape: [2, 2], dtype: .float32)
        let b = graph.addInput(name: "B", shape: [1, 2], dtype: .float32)
        let op = graph.addOpBiasAdd(input: x, bias: b)
        graph.setOutput(nodeId: op)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let xPtr = engine.getBuffer(nodeId: x)!.assumingMemoryBound(to: Float.self)
        let bPtr = engine.getBuffer(nodeId: b)!.assumingMemoryBound(to: Float.self)
        
        xPtr[0] = 1.0; xPtr[1] = 2.0; xPtr[2] = 3.0; xPtr[3] = 4.0
        bPtr[0] = 0.5; bPtr[1] = 1.0
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: op)!.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(outPtr[0], 1.5, accuracy: 1e-5)
        XCTAssertEqual(outPtr[1], 3.0, accuracy: 1e-5)
        XCTAssertEqual(outPtr[2], 3.5, accuracy: 1e-5)
        XCTAssertEqual(outPtr[3], 5.0, accuracy: 1e-5)
    }
    
    func testRelu() throws {
        let graph = runtime.createGraph()
        let x = graph.addInput(name: "X", shape: [2, 2], dtype: .float32)
        let op = graph.addOpRelu(input: x)
        graph.setOutput(nodeId: op)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let xPtr = engine.getBuffer(nodeId: x)!.assumingMemoryBound(to: Float.self)
        xPtr[0] = -1.0; xPtr[1] = 0.0; xPtr[2] = 1.0; xPtr[3] = 2.0
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: op)!.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(outPtr[0], 0.0)
        XCTAssertEqual(outPtr[1], 0.0)
        XCTAssertEqual(outPtr[2], 1.0)
        XCTAssertEqual(outPtr[3], 2.0)
    }

    func testAdd() throws {
        let graph = runtime.createGraph()
        let a = graph.addInput(name: "A", shape: [4], dtype: .float32)
        let b = graph.addInput(name: "B", shape: [4], dtype: .float32)
        let op = graph.addOpAdd(inputA: a, inputB: b)
        graph.setOutput(nodeId: op)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let aPtr = engine.getBuffer(nodeId: a)!.assumingMemoryBound(to: Float.self)
        let bPtr = engine.getBuffer(nodeId: b)!.assumingMemoryBound(to: Float.self)
        
        for i in 0..<4 { 
            aPtr[i] = Float(i+1) // 1, 2, 3, 4
            bPtr[i] = Float((i+1)*10) // 10, 20, 30, 40
        }
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: op)!.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(outPtr[0], 11.0)
        XCTAssertEqual(outPtr[1], 22.0)
        XCTAssertEqual(outPtr[2], 33.0)
        XCTAssertEqual(outPtr[3], 44.0)
    }

    func testMul() throws {
        let graph = runtime.createGraph()
        let a = graph.addInput(name: "A", shape: [4], dtype: .float32)
        let b = graph.addInput(name: "B", shape: [4], dtype: .float32)
        let op = graph.addOpMul(inputA: a, inputB: b)
        graph.setOutput(nodeId: op)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let aPtr = engine.getBuffer(nodeId: a)!.assumingMemoryBound(to: Float.self)
        let bPtr = engine.getBuffer(nodeId: b)!.assumingMemoryBound(to: Float.self)
        
        aPtr[0] = 2.0; aPtr[1] = 3.0; aPtr[2] = 4.0; aPtr[3] = 5.0
        bPtr[0] = 0.5; bPtr[1] = 2.0; bPtr[2] = -1.0; bPtr[3] = 0.0
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: op)!.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(outPtr[0], 1.0)
        XCTAssertEqual(outPtr[1], 6.0)
        XCTAssertEqual(outPtr[2], -4.0)
        XCTAssertEqual(outPtr[3], 0.0)
    }

    func testReduceSum() throws {
        let graph = runtime.createGraph()
        let x = graph.addInput(name: "X", shape: [2, 3], dtype: .float32)
        let op = graph.addOpReduceSum(input: x)
        graph.setOutput(nodeId: op)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let xPtr = engine.getBuffer(nodeId: x)!.assumingMemoryBound(to: Float.self)
        // [1, 2, 3], [4, 5, 6]
        for i in 0..<6 { xPtr[i] = Float(i+1) }
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: op)!.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(outPtr[0], 6.0, accuracy: 1e-5)
        XCTAssertEqual(outPtr[1], 15.0, accuracy: 1e-5)
    }
    
    func testSoftmax() throws {
        let graph = runtime.createGraph()
        let x = graph.addInput(name: "X", shape: [3], dtype: .float32)
        let op = graph.addSoftmax(input: x)
        graph.setOutput(nodeId: op)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let xPtr = engine.getBuffer(nodeId: x)!.assumingMemoryBound(to: Float.self)
        xPtr[0] = 1.0; xPtr[1] = 2.0; xPtr[2] = 3.0
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: op)!.assumingMemoryBound(to: Float.self)
        
        let e1 = exp(1.0)
        let e2 = exp(2.0)
        let e3 = exp(3.0)
        let sum = e1 + e2 + e3
        
        XCTAssertEqual(outPtr[0], Float(e1/sum), accuracy: 1e-5)
        XCTAssertEqual(outPtr[1], Float(e2/sum), accuracy: 1e-5)
        XCTAssertEqual(outPtr[2], Float(e3/sum), accuracy: 1e-5)
    }

    func testIntegrationChain() throws {
        let graph = runtime.createGraph()
        let x = graph.addInput(name: "X", shape: [1, 2], dtype: .float32)
        let w = graph.addInput(name: "W", shape: [2, 2], dtype: .float32)
        let b = graph.addInput(name: "B", shape: [1, 2], dtype: .float32)
        
        let mm = graph.addOpMatMul(inputA: x, inputB: w)
        let ba = graph.addOpBiasAdd(input: mm, bias: b)
        let relu = graph.addOpRelu(input: ba)
        
        graph.setOutput(nodeId: relu)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        let xPtr = engine.getBuffer(nodeId: x)!.assumingMemoryBound(to: Float.self)
        let wPtr = engine.getBuffer(nodeId: w)!.assumingMemoryBound(to: Float.self)
        let bPtr = engine.getBuffer(nodeId: b)!.assumingMemoryBound(to: Float.self)
        
        xPtr[0] = 1.0; xPtr[1] = -1.0
        wPtr[0] = 1.0; wPtr[1] = 2.0; wPtr[2] = 3.0; wPtr[3] = 4.0
        bPtr[0] = 1.0; bPtr[1] = 3.0
        
        engine.execute()
        
        let outPtr = engine.getBuffer(nodeId: relu)!.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(outPtr[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(outPtr[1], 1.0, accuracy: 1e-5)
    }
}
