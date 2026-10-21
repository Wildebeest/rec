package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

func TestReadmeCommandsMatchHelp(t *testing.T) {
	readme, err := os.ReadFile("../README.md")
	if err != nil {
		t.Fatal(err)
	}
	row := regexp.MustCompile("(?m)^\\| `(rec [^`]+)` \\| (.+) \\|$")
	var documented []string
	for _, m := range row.FindAllStringSubmatch(string(readme), -1) {
		documented = append(documented, strings.ReplaceAll(m[1], `\|`, "|")+": "+m[2])
	}
	var help []string
	for _, c := range commands {
		if c.name != "--version" {
			name, _, _ := strings.Cut(c.usage(), " [")
			help = append(help, name+": "+c.summary)
		}
	}
	if strings.Join(documented, "\n") != strings.Join(help, "\n") {
		t.Fatalf("README commands:\n%s\n\nrec --help:\n%s", strings.Join(documented, "\n"), strings.Join(help, "\n"))
	}
}

func TestEveryCommandHasHelp(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	for _, c := range commands {
		if c.name == "--version" {
			continue
		}
		args := append(strings.Fields(c.name)[:1], "--help")
		if err := run(args); err != nil {
			t.Errorf("rec %s: %v", strings.Join(args, " "), err)
		}
	}
}
