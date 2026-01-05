import Foundation

public enum Architecture: Int {
    case unknown = 0
    case x86_64 = 1
    case arm64 = 2
}

public struct SystemCapabilities {
    public let arch: Architecture
    public let archName: String
    public let simdCompiled: Bool
    public let simdSupported: Bool
}

extension VectoriaRuntime {
    public func getSystemCapabilities() -> SystemCapabilities {
        typealias GetCapsFn = @convention(c) (UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int8>, Int) -> Void
        
        let sym = dlsym(library, "vectoria_get_capabilities")
        let fn = unsafeBitCast(sym, to: GetCapsFn.self)
        
        var arch: Int32 = 0
        var compiled: Int32 = 0
        var supported: Int32 = 0
        let bufLen = 64
        let buf = UnsafeMutablePointer<Int8>.allocate(capacity: bufLen)
        defer { buf.deallocate() }
        
        fn(&arch, &compiled, &supported, buf, bufLen)
        
        return SystemCapabilities(
            arch: Architecture(rawValue: Int(arch)) ?? .unknown,
            archName: String(cString: buf),
            simdCompiled: compiled == 1,
            simdSupported: supported == 1
        )
    }
}
