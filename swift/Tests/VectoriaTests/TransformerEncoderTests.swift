import XCTest
@testable import Vectoria

final class TransformerEncoderTests: XCTestCase {
    func testEncoderTrace() throws {
        let runtime = try VectoriaRuntime()
        let graph = runtime.createGraph()
        
        let T: Int64 = 2, dModel: Int64 = 4, heads: Int32 = 2, dff: Int64 = 8
        
        let x = graph.addInput(name: "X", shape: [T, dModel], dtype: .float32)
        let wq = graph.addInput(name: "WQ", shape: [dModel, dModel], dtype: .float32)
        let wk = graph.addInput(name: "WK", shape: [dModel, dModel], dtype: .float32)
        let wv = graph.addInput(name: "WV", shape: [dModel, dModel], dtype: .float32)
        let wo = graph.addInput(name: "WO", shape: [dModel, dModel], dtype: .float32)
        let g1 = graph.addInput(name: "G1", shape: [dModel], dtype: .float32)
        let b1 = graph.addInput(name: "B1", shape: [dModel], dtype: .float32)
        let wf1 = graph.addInput(name: "WF1", shape: [dModel, dff], dtype: .float32)
        let bf1 = graph.addInput(name: "BF1", shape: [dff], dtype: .float32)
        let wf2 = graph.addInput(name: "WF2", shape: [dff, dModel], dtype: .float32)
        let bf2 = graph.addInput(name: "BF2", shape: [dModel], dtype: .float32)
        let g2 = graph.addInput(name: "G2", shape: [dModel], dtype: .float32)
        let b2 = graph.addInput(name: "B2", shape: [dModel], dtype: .float32)
        
        let enc = graph.addTransformerEncoder(x: x, wq: wq, wk: wk, wv: wv, wo: wo, numHeads: heads,
                                              gamma1: g1, beta1: b1, w1: wf1, b1: bf1, w2: wf2, b2: bf2,
                                              gamma2: g2, beta2: b2)
        graph.setOutput(nodeId: enc)
        
        let engine = runtime.createEngine(graph: graph)
        engine.compile()
        engine.execute()
        
        let trace = engine.getTrace()
        XCTAssertFalse(trace.isEmpty)
        XCTAssertTrue(trace.map { $0.details }.contains(where: { $0.contains("Start | Mode") }))
    }
}
