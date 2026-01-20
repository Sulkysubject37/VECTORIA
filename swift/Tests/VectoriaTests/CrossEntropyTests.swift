import XCTest
@testable import Vectoria

final class CrossEntropyTests: XCTestCase {
    func testCrossEntropyTrace() throws {
        let runtime = try VectoriaRuntime()
        let graph = runtime.createGraph()
        
        let logitsId = graph.addInput(name: "Logits", shape: [2, 3], dtype: .float32)
        let targetId = graph.addInput(name: "Target", shape: [2, 3], dtype: .float32)
        
        let ceId = graph.addCrossEntropy(logits: logitsId, target: targetId)
        
        graph.setOutput(nodeId: ceId)
        
        let engine = runtime.createEngine(graph: graph)
        engine.compile()
        engine.execute()
        
        let trace = engine.getTrace()
        XCTAssertFalse(trace.isEmpty)
        
        let details = trace.map { $0.details }
        XCTAssertTrue(details.contains(where: { $0.contains("Start | Mode") }))
    }
}
