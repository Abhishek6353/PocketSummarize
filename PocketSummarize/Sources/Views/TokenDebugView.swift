//
//  TokenDebugView.swift
//  PocketSummarize
//
//  Created by Apple on 05/12/25.
//

import SwiftUI

struct TokenDebugView: View {
    let tokens: [String]
    let ids: [Int]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Tokens (\(tokens.count))")
                .font(.headline)

            ScrollView(.horizontal) {
                Text(tokens.joined(separator: " "))
                    .font(.caption)
                    .padding(6)
            }
            .background(Color(.secondarySystemBackground))
            .cornerRadius(8)

            Text("Input IDs (\(ids.count))")
                .font(.headline)

            ScrollView(.horizontal) {
                Text(ids.map(String.init).joined(separator: ", "))
                    .font(.caption)
                    .padding(6)
            }
            .background(Color(.secondarySystemBackground))
            .cornerRadius(8)

            Spacer()
        }
        .padding(8)
    }
}
