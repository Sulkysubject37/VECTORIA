import XCTest
@testable import Vectoria

final class CoreMLEquivalenceTests: XCTestCase {
    var runtime: VectoriaRuntime!
    
    override func setUpWithError() throws {
        runtime = try VectoriaRuntime(libraryPath: "libvectoria.dylib")
    }
    
    func testExportStructure() throws {
        // Build graph: Add(A, B)
        let graph = runtime.createGraph()
        let a = graph.addInput(name: "A", shape: [2, 2], dtype: .float32)
        let b = graph.addInput(name: "B", shape: [2, 2], dtype: .float32)
        let op = graph.addOpAdd(inputA: a, inputB: b)
        graph.setOutput(nodeId: op)
        
        let path = "test_swift_export.mlpackage"
        try graph.exportToCoreML(path: path)
        
        // Structural check via FileManager
        let fileManager = FileManager.default
        let milPath = path + "/Data/com.apple.CoreML/model.mil"
        XCTAssertTrue(fileManager.fileExists(atPath: milPath))
        
        // Cleanup
        try? fileManager.removeItem(atPath: path)
    }
}
