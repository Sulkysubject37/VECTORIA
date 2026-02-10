import Foundation
import Vectoria

func runExample() {
    print("VECTORIA Swift Example: Transformer Encoder")
    
    do {
        // Try to find the library in the current directory or parent
        let runtime = try VectoriaRuntime(libraryPath: "./libvectoria.dylib")
        let graph = runtime.createGraph()
        
        // Define Transformer Encoder Inputs (T=2, D=4)
        let x = graph.addInput(name: "X", shape: [2, 4], dtype: .float32)
        
        // Parameters (Simplified)
        let wq = graph.addInput(name: "WQ", shape: [4, 4], dtype: .float32)
        let wk = graph.addInput(name: "WK", shape: [4, 4], dtype: .float32)
        let wv = graph.addInput(name: "WV", shape: [4, 4], dtype: .float32)
        let wo = graph.addInput(name: "WO", shape: [4, 4], dtype: .float32)
        
        let gamma1 = graph.addInput(name: "G1", shape: [4], dtype: .float32)
        let beta1 = graph.addInput(name: "B1", shape: [4], dtype: .float32)
        
        let w1 = graph.addInput(name: "W1", shape: [4, 8], dtype: .float32)
        let b1 = graph.addInput(name: "B1_FFN", shape: [8], dtype: .float32)
        let w2 = graph.addInput(name: "W2", shape: [8, 4], dtype: .float32)
        let b2 = graph.addInput(name: "B2_FFN", shape: [4], dtype: .float32)
        
        let gamma2 = graph.addInput(name: "G2", shape: [4], dtype: .float32)
        let beta2 = graph.addInput(name: "B2", shape: [4], dtype: .float32)
        
        let encoder = graph.addTransformerEncoder(
            x: x, wq: wq, wk: wk, wv: wv, wo: wo, numHeads: 1,
            gamma1: gamma1, beta1: beta1,
            w1: w1, b1: b1, w2: w2, b2: b2,
            gamma2: gamma2, beta2: beta2
        )
        
        graph.setOutput(nodeId: encoder)
        
        let engine = runtime.createEngine(graph: graph, policy: .reference)
        engine.compile()
        
        print("Executing Engine...")
        engine.execute()
        
        print("Execution Trace:")
        let trace = engine.getTrace()
        for event in trace {
            print("  [\(event.timestamp)] Type: \(event.type), Node: \(event.nodeId), Details: \(event.details)")
        }
        
    } catch {
        print("Error: \(error)")
    }
}

runExample()
