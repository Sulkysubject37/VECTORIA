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

    func testStructuralExport() throws {
        let graph = runtime.createGraph()
        let inNode = graph.addInput(name: "X", shape: [2, 3], dtype: .float32)
        let t = graph.addTranspose(input: inNode, perm: [1, 0])
        let r = graph.addReshape(input: t, shape: [6])
        graph.setOutput(nodeId: r)
        
        let path = "test_swift_structural.mlpackage"
        try graph.exportToCoreML(path: path)
        
        let fileManager = FileManager.default
        let milPath = path + "/Data/com.apple.CoreML/model.mil"
        XCTAssertTrue(fileManager.fileExists(atPath: milPath))
        
        if let content = try? String(contentsOfFile: milPath) {
            XCTAssertTrue(content.contains("transpose"))
            XCTAssertTrue(content.contains("reshape"))
        }
        
        try? fileManager.removeItem(atPath: path)
    }
}
