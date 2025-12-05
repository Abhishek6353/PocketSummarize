//
//  SummaryView.swift
//  PocketSummarize
//
//  Created by Apple on 05/12/25.
//

import SwiftUI

struct SummaryView: View {
    @State private var inputText: String = ""
       @State private var summaryText: String = ""
       @State private var isLoading: Bool = false
       @State private var showDebug: Bool = false
       @State private var lastTokens: [String] = []
       @State private var lastIds: [Int] = []
       
       private let engine = try! SummaryEngine(seqLen: 64)

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    
                    // Title
                    Text("PocketSummarize")
                        .font(.largeTitle.bold())
                        .frame(maxWidth: .infinity, alignment: .leading)
                    
                    // INPUT
                    TextEditor(text: $inputText)
                        .frame(height: 200)
                        .padding(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.secondary, lineWidth: 1)
                        )
                        .scrollContentBackground(.hidden)
                    
                    // BUTTON
                    Button(action: {
                        hideKeyboard()
                        summarize()
                    }) {
                        Text("Summarize")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    
                    if isLoading {
                        ProgressView("Processing…")
                            .padding(.top, 8)
                    }
                    
                    // SUMMARY OUTPUT
                    if !summaryText.isEmpty {
                        Text(summaryText)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(.secondarySystemBackground))
                            .cornerRadius(10)
                    }
                    
                    // DEBUG TOGGLE
                    Toggle("Show Debug Info", isOn: $showDebug)
                    
                    // DEBUG VIEW
                    if showDebug {
                        TokenDebugView(tokens: lastTokens, ids: lastIds)
                    }
                    
                    Spacer(minLength: 40)
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }

    // MARK: - Summarize
    func summarize() {
        isLoading = true
        summaryText = ""

        Task {
            do {
                // Debug: re-tokenize for display
                let tok = try MiniLMTokenizer(vocabFileName: "vocab.txt",
                                              resourceSubpath: "Models/llm",
                                              maxSequenceLength: 64)
                let encoded = tok.encode(inputText)
                lastTokens = encoded.tokens
                lastIds = encoded.inputIds

                // Run summarizer
                let result = try await engine.summarize(inputText)
                await MainActor.run {
                    summaryText = result
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    summaryText = "Error: \(error.localizedDescription)"
                    isLoading = false
                }
            }
        }
    }
}


import SwiftUI

extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                        to: nil, from: nil, for: nil)
    }
}
