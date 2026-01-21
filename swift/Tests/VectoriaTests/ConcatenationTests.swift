import XCTest
@testable import Vectoria

final class ConcatenationTests: XCTestCase {
    func testConcatenationTrace() throws {
        let runtime = try VectoriaRuntime()
        let graph = runtime.createGraph()
        
        let in1 = graph.addInput(name: "X1", shape: [2, 2], dtype: .float32)
        let in2 = graph.addInput(name: "X2", shape: [2, 3], dtype: .float32)
        
        let concat = graph.addConcat(inputs: [in1, in2], axis: 1)
        graph.setOutput(nodeId: concat)
        
        let engine = runtime.createEngine(graph: graph)
        engine.compile()
        engine.execute()
        
        let trace = engine.getTrace()
        XCTAssertFalse(trace.isEmpty)
        
        let details = trace.map { $0.details }
        XCTAssertTrue(details.contains(where: { $0.contains("Reference | Axis: 1") }))
    }
}
