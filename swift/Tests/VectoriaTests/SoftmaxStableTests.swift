import XCTest
@testable import Vectoria

final class SoftmaxStableTests: XCTestCase {
    func testSoftmaxStableTrace() throws {
        let runtime = try VectoriaRuntime()
        let graph = runtime.createGraph()
        
        let inId = graph.addInput(name: "Input", shape: [2, 3], dtype: .float32)
        let sId = graph.addSoftmaxStable(input: inId)
        
        graph.setOutput(nodeId: sId)
        
        let engine = runtime.createEngine(graph: graph)
        engine.compile()
        engine.execute()
        
        let trace = engine.getTrace()
        XCTAssertFalse(trace.isEmpty)
        
        let details = trace.map { $0.details }
        XCTAssertTrue(details.contains(where: { $0.contains("Start | Mode") }))
    }
}
