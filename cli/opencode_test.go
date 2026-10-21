//go:build darwin || linux

package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestOpenCodePinIsComplete(t *testing.T) {
	pin, err := loadOpenCodePin()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(pin.Client.Tag, "opencode-rec-"+pin.Version) || pin.Client.Name == "" || pin.Client.URL != "https://github.com/recursive-oss/rec/releases/download/"+pin.Client.Tag+"/"+pin.Client.Name || pin.Client.Size <= 0 || len(pin.Client.SHA256) != 64 {
		t.Fatalf("incomplete OpenCode pin: %+v", pin)
	}
}

func TestOpenCodeClientDownloadsOnce(t *testing.T) {
	var archive bytes.Buffer
	z := gzip.NewWriter(&archive)
	if _, err := z.Write([]byte("#!/bin/sh\necho opencode\n")); err != nil {
		t.Fatal(err)
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Write(archive.Bytes())
	}))
	defer s.Close()
	sum := sha256.Sum256(archive.Bytes())
	var pin openCodePin
	pin.Version = "2.0.29"
	pin.Client.Tag = "opencode-rec-2.0.29"
	pin.Client.File = File{Name: "opencode-darwin-arm64.gz", URL: s.URL, Size: int64(archive.Len()), SHA256: hex.EncodeToString(sum[:])}

	p := dataPaths(t.TempDir())
	var out bytes.Buffer
	for range 2 {
		binary, err := openCodeClient(p, pin, s.Client(), &out)
		if err != nil {
			t.Fatal(err)
		}
		if got, err := os.ReadFile(binary); err != nil || !strings.Contains(string(got), "echo opencode") {
			t.Fatalf("client not unpacked: %v", err)
		}
		if st, _ := os.Stat(binary); st.Mode().Perm() != 0700 {
			t.Fatal("client is not executable")
		}
	}
	if requests.Load() != 1 {
		t.Fatalf("downloaded %d times", requests.Load())
	}

	pin.Client.Tag = "opencode-rec-2.0.29-2"
	pin.Client.SHA256 = strings.Repeat("0", 64)
	if _, err := openCodeClient(p, pin, s.Client(), &out); err == nil {
		t.Fatal("accepted an asset with the wrong checksum")
	}
	if _, err := os.Stat(filepath.Join(p.Data, "clients", pin.Client.Tag, "opencode")); !os.IsNotExist(err) {
		t.Fatal("installed an unverified client")
	}
}

func TestOpenCodePluginAndTUIConfig(t *testing.T) {
	p := dataPaths(t.TempDir())
	plugin, err := openCodePlugin(p)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"tui.tsx", "pixels.json", "assets/logo-frame-1.svg"} {
		if _, err := os.Stat(filepath.Join(plugin, name)); err != nil {
			t.Fatal(err)
		}
	}
	for _, existing := range []string{`{"plugins":["/logo"]}`, `{"plugins":[{"package":"/logo","options":{"animate":false}}]}`} {
		if got, err := openCodeTUIConfig(existing, "/logo"); err != nil || got != existing {
			t.Fatalf("duplicated or changed a registered plugin: %q", got)
		}
	}
	if got, err := openCodeTUIConfig(`{"plugins":["other"]}`, "/logo"); err != nil || got != `{"plugins":["other","/logo"]}` {
		t.Fatalf("did not append the plugin: %q", got)
	}
	if _, err := openCodeTUIConfig("{", "/logo"); err == nil {
		t.Fatal("accepted invalid settings")
	}
}
