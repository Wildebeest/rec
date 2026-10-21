package main

import (
	"os"

	"charm.land/lipgloss/v2"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		_, _ = lipgloss.Fprintln(os.Stderr, failureStyle.Bold(true).Render("rec:"), err)
		os.Exit(1)
	}
}
