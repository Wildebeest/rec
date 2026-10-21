package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/colorprofile"
	"github.com/charmbracelet/x/ansi"
	"github.com/charmbracelet/x/term"
)

var (
	accentStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("#3987e5"))
	boldStyle    = lipgloss.NewStyle().Bold(true)
	faintStyle   = lipgloss.NewStyle().Faint(true)
	successStyle = lipgloss.NewStyle().Foreground(lipgloss.Green)
	failureStyle = lipgloss.NewStyle().Foreground(lipgloss.Red)
	warningStyle = lipgloss.NewStyle().Foreground(lipgloss.Yellow)
	barColors    = lipgloss.Blend1D(barWidth, lipgloss.Color("#2a78d6"), lipgloss.Color("#5ec8f0"))
	panelStyle   = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("#2a78d6")).Padding(0, 2)
)

const barWidth = 30

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

type screen struct {
	out  *colorprofile.Writer
	fd   uintptr
	live bool
}

func newScreen(out io.Writer) *screen {
	s := &screen{out: colorprofile.NewWriter(out, os.Environ())}
	if f, ok := out.(*os.File); ok {
		s.fd = f.Fd()
		s.live = term.IsTerminal(s.fd) && s.out.Profile > colorprofile.NoTTY
	}
	if !s.live {
		s.out.Profile = colorprofile.NoTTY
	}
	return s
}

func (s *screen) println(text string) { _, _ = fmt.Fprintln(s.out, text) }

func (s *screen) redraw(text string) {
	if width, _, err := term.GetSize(s.fd); err == nil && width > 1 {
		text = ansi.Truncate(text, width-1, "…")
	}
	_, _ = fmt.Fprint(s.out, "\r"+text+ansi.EraseLineRight)
}

func (s *screen) step(mark, label, detail string) {
	text := "  " + mark + " " + label
	if detail != "" {
		text += "  " + faintStyle.Render(detail)
	}
	s.println(text)
}

func (s *screen) succeeded(label, detail string) {
	s.step(successStyle.Render("✓"), label, detail)
}

func (s *screen) failed(label, detail string) { s.step(failureStyle.Render("✗"), label, detail) }

func (s *screen) table(rows [][2]string) {
	width := 0
	for _, row := range rows {
		width = max(width, len(row[0]))
	}
	for _, row := range rows {
		s.println("  " + faintStyle.Render(fmt.Sprintf("%-*s", width, row[0])) + "  " + row[1])
	}
}

func displayName(model string) string {
	return strings.ToUpper(model[:1]) + model[1:]
}

func formatBytes(n int64) string {
	if n >= 1e9 {
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	}
	return fmt.Sprintf("%.1f MB", float64(n)/1e6)
}

func formatDuration(d time.Duration) string {
	d = d.Round(time.Second)
	if d >= time.Minute {
		return fmt.Sprintf("%dm %02ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%ds", int(d.Seconds()))
}

func formatTokens(n int) string {
	digits := fmt.Sprint(n)
	var b strings.Builder
	for i, digit := range digits {
		if i > 0 && (len(digits)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(digit)
	}
	return b.String()
}

func bar(fraction float64) string {
	halves := int(fraction * barWidth * 2)
	var b strings.Builder
	for i := range barWidth {
		cell := lipgloss.NewStyle().Foreground(barColors[i])
		switch {
		case 2*i+2 <= halves:
			b.WriteString(cell.Render("━"))
		case 2*i+1 == halves:
			b.WriteString(cell.Render("╸"))
		default:
			b.WriteString(faintStyle.Render("━"))
		}
	}
	return b.String()
}
