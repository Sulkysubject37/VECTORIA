import XCTest
@testable import Vectoria

final class StructuralOpsTests: XCTestCase {
    func testStructuralTrace() throws {
        let runtime = try VectoriaRuntime()
        let graph = runtime.createGraph()
        
        let inId = graph.addInput(name: "Input", shape: [2, 3], dtype: .float32)
        let tId = graph.addTranspose(input: inId, perm: [1, 0])
        let rId = graph.addReshape(input: tId, shape: [6])
        
        graph.setOutput(nodeId: rId)
        
        let engine = runtime.createEngine(graph: graph)
        engine.compile()
        engine.execute()
        
        let trace = engine.getTrace()
        XCTAssertFalse(trace.isEmpty)
        
        let details = trace.map { $0.details }
        // We expect "Reference | Inputs: [...]" for Transpose
        // "Reference (Copy) | Inputs: [...]" for Reshape
        XCTAssertTrue(details.contains(where: { $0.contains("Reference") }))
    }
}
