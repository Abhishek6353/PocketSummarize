//
//  MLHelpers.swift
//  PocketSummarize
//
//  Created by Apple on 05/12/25.
//

// Utility helpers for creating MLMultiArray inputs and converting outputs.
// Referenced by ContentView and SummaryEngine. :contentReference[oaicite:0]{index=0}
import Foundation
import CoreML

/// Build MLMultiArray of Int32 with shape [1, seqLen]
/// - Parameters:
///   - values: array of Int (will be cast to Int32)
///   - seqLen: expected length (will assert/crop/pad to this length)
/// - Returns: MLMultiArray with dataType .int32 and shape [1, seqLen]
public func makeInt32MLArray(from values: [Int], seqLen: Int) throws -> MLMultiArray {
    let shape: [NSNumber] = [1, NSNumber(value: seqLen)]
    let ml = try MLMultiArray(shape: shape, dataType: .int32)
    
    // Ensure we have exactly seqLen elements (crop or pad with zeros)
    var vals = values
    if vals.count > seqLen {
        vals = Array(vals.prefix(seqLen))
    } else if vals.count < seqLen {
        vals += Array(repeating: 0, count: seqLen - vals.count)
    }
    
    // Fill flat-index order
    // Use direct pointer writes for performance
    let count = seqLen
    let ptr = UnsafeMutableRawPointer(ml.dataPointer).assumingMemoryBound(to: Int32.self)
    for i in 0..<count {
        ptr[i] = Int32(vals[i])
    }
    return ml
}

/// Build MLMultiArray of Float32 with shape [1, seqLen]
/// - Useful for models that expect float inputs (legacy tests)
public func makeFloatMLArray(from floats: [Float], seqLen: Int) throws -> MLMultiArray {
    let shape: [NSNumber] = [1, NSNumber(value: seqLen)]
    let ml = try MLMultiArray(shape: shape, dataType: .float32)
    
    var vals = floats
    if vals.count > seqLen {
        vals = Array(vals.prefix(seqLen))
    } else if vals.count < seqLen {
        vals += Array(repeating: 0.0, count: seqLen - vals.count)
    }
    
    let count = seqLen
    let ptr = UnsafeMutableRawPointer(ml.dataPointer).assumingMemoryBound(to: Float.self)
    for i in 0..<count {
        ptr[i] = vals[i]
    }
    return ml
}

public extension MLMultiArray {
    /// Convert MLMultiArray (float32/double/int32) to [Float]
    /// Works for output embeddings (typical .float32) and also supports .double/.int32.
    func toFloatArray() -> [Float] {
        let count = self.count
        var result = [Float](repeating: 0, count: count)
        
        switch self.dataType {
        case .float32:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Float.self)
            for i in 0..<count { result[i] = ptr[i] }
        case .double:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Double.self)
            for i in 0..<count { result[i] = Float(ptr[i]) }
        case .int32:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Int32.self)
            for i in 0..<count { result[i] = Float(ptr[i]) }
        default:
            // Fallback using NSNumber accessor (slower)
            for i in 0..<count {
                result[i] = self[i].floatValue
            }
        }
        return result
    }
    
    /// Convert MLMultiArray (int32) to [Int]
    /// Useful for reading back input ids when needed for debugging.
    func toIntArray() -> [Int] {
        let count = self.count
        var result = [Int](repeating: 0, count: count)
        
        switch self.dataType {
        case .int32:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Int32.self)
            for i in 0..<count { result[i] = Int(ptr[i]) }
        case .float32:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Float.self)
            for i in 0..<count { result[i] = Int(ptr[i]) }
        case .double:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Double.self)
            for i in 0..<count { result[i] = Int(ptr[i]) }
        default:
            for i in 0..<count {
                result[i] = self[i].intValue
            }
        }
        return result
    }
}
