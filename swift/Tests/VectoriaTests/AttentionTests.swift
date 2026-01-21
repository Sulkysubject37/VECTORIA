import XCTest
@testable import Vectoria

final class AttentionTests: XCTestCase {
    func testAttentionTrace() throws {
        let runtime = try VectoriaRuntime()
        let graph = runtime.createGraph()
        
        let q = graph.addInput(name: "Q", shape: [2, 4], dtype: .float32)
        let k = graph.addInput(name: "K", shape: [2, 4], dtype: .float32)
        let v = graph.addInput(name: "V", shape: [2, 2], dtype: .float32)
        
        let attn = graph.addAttention(q: q, k: k, v: v)
        graph.setOutput(nodeId: attn)
        
        let engine = runtime.createEngine(graph: graph)
        engine.compile()
        engine.execute()
        
        let trace = engine.getTrace()
        XCTAssertFalse(trace.isEmpty)
        
        let details = trace.map { $0.details }
        XCTAssertTrue(details.contains(where: { $0.contains("Start | Mode") }))
    }
}
