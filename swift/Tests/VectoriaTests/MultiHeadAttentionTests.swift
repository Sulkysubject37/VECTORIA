import XCTest
@testable import Vectoria

final class MultiHeadAttentionTests: XCTestCase {
    func testMHATrace() throws {
        let runtime = try VectoriaRuntime()
        let graph = runtime.createGraph()
        
        let dModel: Int64 = 4
        let heads: Int32 = 2
        let seqLen: Int64 = 3
        
        let x = graph.addInput(name: "X", shape: [seqLen, dModel], dtype: .float32)
        let wq = graph.addInput(name: "WQ", shape: [dModel, dModel], dtype: .float32)
        let wk = graph.addInput(name: "WK", shape: [dModel, dModel], dtype: .float32)
        let wv = graph.addInput(name: "WV", shape: [dModel, dModel], dtype: .float32)
        let wo = graph.addInput(name: "WO", shape: [dModel, dModel], dtype: .float32)
        
        let mha = graph.addMultiHeadAttention(x: x, wq: wq, wk: wk, wv: wv, wo: wo, numHeads: heads)
        graph.setOutput(nodeId: mha)
        
        let engine = runtime.createEngine(graph: graph)
        engine.compile()
        engine.execute()
        
        let trace = engine.getTrace()
        XCTAssertFalse(trace.isEmpty)
        
        let details = trace.map { $0.details }
        XCTAssertTrue(details.contains(where: { $0.contains("Start | Mode") }))
    }
}
