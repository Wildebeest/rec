package main

import "fmt"

func (m Manifest) contextLimit() int {
	if m.MaxContext > 0 {
		return m.MaxContext
	}
	return m.Context
}

func (m Manifest) contextSummary() string {
	summary := formatTokens(m.Context) + " tokens, input + output"
	if m.Context < m.contextLimit() {
		summary += " (up to " + formatTokens(m.contextLimit()) + " with --context)"
	}
	return summary
}

func withContext(m Manifest, requested int) (Manifest, error) {
	limit := m.contextLimit()
	if requested < 256 || requested > limit {
		return m, fmt.Errorf("context must be between 256 and %d tokens for this profile (requested %d)", limit, requested)
	}
	m.MaxContext = limit
	m.Context = requested
	return m, nil
}
