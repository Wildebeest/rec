//go:build darwin || linux

package main

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
)

type openCodePin struct {
	Version string `json:"version"`
	Client  struct {
		Tag string `json:"tag"`
		File
	} `json:"client"`
}

func loadOpenCodePin() (openCodePin, error) {
	var pin openCodePin
	err := json.Unmarshal(openCodePins, &pin)
	return pin, err
}

func openCodeBinary(p Paths, pin openCodePin) string {
	return filepath.Join(p.Data, "clients", pin.Client.Tag, "opencode")
}

// Downloads, verifies and unpacks the pinned client on first use.
func openCodeClient(p Paths, pin openCodePin, client *http.Client, out io.Writer) (string, error) {
	binary := openCodeBinary(p, pin)
	dir := filepath.Dir(binary)
	if _, err := os.Stat(binary); err == nil {
		return binary, nil
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	l, err := lock(filepath.Join(dir, "download.lock"))
	if err != nil {
		return "", fmt.Errorf("another launch is downloading OpenCode; retry when it finishes")
	}
	defer unlock(l)
	archive := filepath.Join(dir, pin.Client.Name)
	progress := newTransfer(out, "Downloading OpenCode "+pin.Version, "Downloaded OpenCode "+pin.Version, 0, pin.Client.Size)
	if err := download(context.Background(), client, pin.Client.File, archive, progress, 0); err != nil {
		progress.fail("")
		return "", fmt.Errorf("OpenCode download failed: %w", err)
	}
	progress.succeed()
	if err := gunzipExecutable(archive, binary); err != nil {
		return "", err
	}
	return binary, os.Remove(archive)
}

func gunzipExecutable(archive, dest string) error {
	in, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer in.Close()
	z, err := gzip.NewReader(in)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(dest), ".rec-tmp-")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	_, err = io.Copy(f, z)
	if err == nil {
		err = f.Chmod(0700)
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(f.Name(), dest)
}

// Writes the logo plugin for this rec version and returns its directory.
func openCodePlugin(p Paths) (string, error) {
	root := filepath.Join(p.Data, "clients", "opencode", Version)
	err := fs.WalkDir(openCodeLogo, "clients/opencode/recursive-logo", func(name string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		dest := filepath.Join(root, name[len("clients/opencode/"):])
		if d.IsDir() {
			return os.MkdirAll(dest, 0700)
		}
		data, err := openCodeLogo.ReadFile(name)
		if err != nil {
			return err
		}
		return atomicWrite(dest, data, 0600)
	})
	return filepath.Join(root, "recursive-logo"), err
}

// Adds the logo plugin to the user's inline TUI settings, keeping their entries.
func openCodeTUIConfig(existing, plugin string) (string, error) {
	config := map[string]any{}
	if existing != "" {
		if err := json.Unmarshal([]byte(existing), &config); err != nil {
			return "", fmt.Errorf("invalid OPENCODE_CLI_CONFIG_CONTENT: %w", err)
		}
	}
	plugins, ok := config["plugins"].([]any)
	if config["plugins"] != nil && !ok {
		return "", fmt.Errorf("invalid OPENCODE_CLI_CONFIG_CONTENT: plugins must be a list")
	}
	for _, entry := range plugins {
		if entry == plugin {
			return existing, nil
		}
		if object, ok := entry.(map[string]any); ok && object["package"] == plugin {
			return existing, nil
		}
	}
	config["plugins"] = append(plugins, plugin)
	data, err := json.Marshal(config)
	return string(data), err
}
