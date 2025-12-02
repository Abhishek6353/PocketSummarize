//
//  ContentView.swift
//  PocketSummarize
//
//  Created by Apple on 02/12/25.
//

import SwiftUI
import CoreML

struct ContentView: View {
    @State private var status: String = "Initializing…"
    
    var body: some View {
        VStack(spacing: 16) {
            Text("AllMiniLML6V2 — Core ML Test")
                .font(.title2).bold()
            
            Text(status)
                .font(.body)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
                .multilineTextAlignment(.leading)
            
            Spacer()
        }
        .padding()
        .task {
            await runModelTest()
        }
    }
    
    // MARK: - Run test
    func runModelTest() async {
        do {
            // 1) Load model
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try AllMiniLML6V2(configuration: config)
            
            // 2) Sample text
            let sampleText = "This is a quick test of MiniLM CoreML model"
            
            // 3) Tokenize (FAKE tokenizer for test only)
            // Produces arrays of Float with length seqLen (64).
            let seqLen = 64
            let (idsFloat, maskFloat) = fakeTestTokenizerFloats(text: sampleText, seqLen: seqLen)
            
            // 4) Convert to MLMultiArray (Float32) with shape [1, seqLen]
            let inputIdsArray = try makeFloatMLArray(from: idsFloat, seqLen: seqLen)
            let attentionMaskArray = try makeFloatMLArray(from: maskFloat, seqLen: seqLen)
            
            // 5) Call the model (convenience API)
            let output = try model.prediction(input_ids: inputIdsArray, attention_mask: attentionMaskArray)
            
            // 6) Read output (var_570)
            let mlArray = output.var_570    // MLMultiArray
            let embedding = mlArray.toFloatArray()
            
            // 7) Display some info
            let firstN = min(8, embedding.count)
            let preview = embedding.prefix(firstN).map { String(format: "%.5f", $0) }.joined(separator: ", ")
            await MainActor.run {
                status = """
                Success! Embedding length: \(embedding.count)
                First \(firstN) values: [\(preview)]
                (Note: values are from a test tokenizer; use a real tokenizer for meaningful results.)
                """
            }
        } catch {
            await MainActor.run {
                status = "Error: \(error.localizedDescription)\nCheck model is included in bundle and input shapes/types match."
            }
        }
    }
}


// MARK: - Helpers (Test tokenizer + MLMultiArray builders)

/// Fake tokenizer for quick testing ONLY:
/// Converts words to pseudo-token floats by hashing each word and producing small integers,
/// pads/truncates to seqLen, returns Float arrays for input_ids and attention_mask.
func fakeTestTokenizerFloats(text: String, seqLen: Int) -> ([Float], [Float]) {
    let words = text.split { $0 == " " || $0.isNewline || $0.isPunctuation }.map { String($0) }
    var tokens = [Float]()
    for w in words {
        // simple stable hash -> small token id (not real tokenizer)
        var h: Int = 0
        for scalar in w.unicodeScalars {
            h = (h &* 31) &+ Int(scalar.value)
        }
        // keep token id in small range
        let tokenId = abs(h) % 30000
        tokens.append(Float(tokenId))
    }
    // build attention mask (1.0 for real tokens)
    var mask = [Float](repeating: 0.0, count: seqLen)
    if tokens.count >= seqLen {
        tokens = Array(tokens.prefix(seqLen))
        for i in 0..<seqLen { mask[i] = 1.0 }
    } else {
        for i in 0..<tokens.count { mask[i] = 1.0 }
        // pad tokens with 0.0
        tokens += Array(repeating: 0.0, count: seqLen - tokens.count)
    }
    // ensure length exactly seqLen
    if tokens.count != seqLen { tokens = Array(tokens.prefix(seqLen)) }
    if mask.count != seqLen { mask = Array(mask.prefix(seqLen)) }
    return (tokens, mask)
}

/// Build MLMultiArray of Float32 with shape [1, seqLen]
func makeFloatMLArray(from floats: [Float], seqLen: Int) throws -> MLMultiArray {
    // shape must be [1, seqLen]
    let shape: [NSNumber] = [1, NSNumber(value: seqLen)]
    let ml = try MLMultiArray(shape: shape, dataType: .float32)
    // Fill sequentially
    for i in 0..<floats.count {
        let value = floats[i]
        ml[i] = NSNumber(value: value)
    }
    return ml
}

extension MLMultiArray {
    /// Convert MLMultiArray (float32 or double) to [Float]
    func toFloatArray() -> [Float] {
        let count = self.count
        var result = [Float](repeating: 0, count: count)
        switch self.dataType {
        case .float32:
            // pointer to Float
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Float.self)
            for i in 0..<count { result[i] = ptr[i] }
        case .double:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Double.self)
            for i in 0..<count { result[i] = Float(ptr[i]) }
        case .int32:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Int32.self)
            for i in 0..<count { result[i] = Float(ptr[i]) }
        default:
            // fallback: use featureValue conversion (slower)
            for i in 0..<count {
                result[i] = self[i].floatValue
            }
        }
        return result
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}

