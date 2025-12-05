//
//  SummaryEngine.swift
//  PocketSummarize
//
//  Created by Apple on 05/12/25.
//

import Foundation
import CoreML

public struct SummaryOptions {
    public var maxSentences: Int = 3
    public var seqLen: Int = 64
    public init(maxSentences: Int = 3, seqLen: Int = 64) {
        self.maxSentences = maxSentences
        self.seqLen = seqLen
    }
}

public final class SummaryEngine {
    private let tokenizer: MiniLMTokenizer
    private let model: AllMiniLML6V2
    private let seqLen: Int
    
    public init(vocabFileName: String = "vocab.txt",
                resourceSubpath: String? = "Models/llm",
                seqLen: Int = 64) throws
    {
        self.seqLen = seqLen
        self.tokenizer = try MiniLMTokenizer(vocabFileName: vocabFileName,
                                             resourceSubpath: resourceSubpath,
                                             bundle: .main,
                                             maxSequenceLength: seqLen)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        self.model = try AllMiniLML6V2(configuration: cfg)
    }
    
    // Public API: returns an extractive summary (joined sentences)
    public func summarize(_ text: String, options: SummaryOptions = .init()) async throws -> String {
        // 1) Split into sentences (simple rule-based)
        let sentences = Self.splitIntoSentences(text)
        guard !sentences.isEmpty else { return "" }
        
        // 2) Embed whole document (use concatenation or average of sentences)
        let docEmbedding = try await embed(text: text)
        
        // 3) Embed each sentence
        var sentEmbeddings: [[Float]] = []
        for s in sentences {
            let emb = try await embed(text: s)
            sentEmbeddings.append(emb)
        }
        
        // 4) Score sentences by cosine similarity to docEmbedding
        var scores: [Float] = sentEmbeddings.map { cosineSimilarity($0, docEmbedding) }
        
        // 5) Pick top-N indices
        let topN = min(options.maxSentences, sentences.count)
        let indexed = scores.enumerated().map { ($0.offset, $0.element) }
        let sortedByScore = indexed.sorted { $0.1 > $1.1 }
        let topIndices = sortedByScore.prefix(topN).map { $0.0 }
        
        // 6) Keep original order
        let orderedTop = topIndices.sorted()
        let summary = orderedTop.map { sentences[$0] }.joined(separator: " ")
        
        return summary
    }
    
    // MARK: - Helpers
    
    // Embed a single text (tokenize -> make Int32 MLMultiArray -> model -> float array)
    private func embed(text: String) async throws -> [Float] {
        // Tokenize (synchronous)
        let encoded = tokenizer.encode(text)
        
        // Ensure lengths match seqLen
        let ids = encoded.inputIds
        let mask = encoded.attentionMask
        
        guard ids.count == seqLen, mask.count == seqLen else {
            throw NSError(domain: "SummaryEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "Tokenizer produced invalid lengths"])
        }
        
        // Build MLMultiArray (synchronous)
        let inputIdsArray = try makeInt32MLArray(from: ids, seqLen: seqLen)
        let attentionMaskArray = try makeInt32MLArray(from: mask, seqLen: seqLen)
        
        // Call model (synchronous API; wrap in Task to avoid blocking)
        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let output = try self.model.prediction(input_ids: inputIdsArray, attention_mask: attentionMaskArray)
                    let mlArray = output.var_570
                    let emb = mlArray.toFloatArray()
                    cont.resume(returning: emb)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
    
    // Simple sentence splitter (works well enough for English short text)
    private static func splitIntoSentences(_ text: String) -> [String] {
        // Use CFStringTokenizer for better splitting or simple punctuation split:
        var results: [String] = []
        let options: NSString.EnumerationOptions = [.bySentences, .substringNotRequired]
        (text as NSString).enumerateSubstrings(in: NSRange(location: 0, length: text.utf16.count),
                                               options: [.bySentences]) { (sub, _, _, _) in
            if let s = sub?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty {
                results.append(s)
            }
        }
        return results
    }
    
    // Cosine similarity
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = sqrt(na) * sqrt(nb)
        return denom == 0 ? 0 : dot / denom
    }
}
