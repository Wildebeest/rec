//go:build darwin || linux

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

func piInstallDir(p Paths) string {
	sum := sha256.Sum256(append(append([]byte{}, piPackage...), piLock...))
	return filepath.Join(p.Data, "clients", "pi", "npm-"+hex.EncodeToString(sum[:8]))
}

func piExecutable(p Paths) string {
	return filepath.Join(piInstallDir(p), "node_modules", ".bin", "pi")
}

func installPi(p Paths, out io.Writer) (string, error) {
	dir := piInstallDir(p)
	parent := filepath.Dir(dir)
	executable := piExecutable(p)
	if _, err := os.Stat(executable); err == nil {
		return executable, nil
	}
	npm, err := exec.LookPath("npm")
	if err != nil {
		return "", fmt.Errorf("rec launch pi requires Node.js/npm; install Node.js, then retry")
	}
	if err := os.MkdirAll(parent, 0700); err != nil {
		return "", err
	}
	l, err := lock(filepath.Join(p.State, "pi-install.lock"))
	if err != nil {
		return "", fmt.Errorf("another launcher is installing Pi; retry when it finishes")
	}
	defer unlock(l)
	if _, err := os.Stat(executable); err == nil {
		return executable, nil
	}
	// Remove staging left by an interrupted install. Earlier pins are kept,
	// since a running Pi may still load modules from them.
	leftovers, err := filepath.Glob(filepath.Join(parent, ".install-*"))
	if err != nil {
		return "", err
	}
	for _, path := range leftovers {
		if err := os.RemoveAll(path); err != nil {
			return "", err
		}
	}
	staging, err := os.MkdirTemp(parent, ".install-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(staging)
	if err := os.WriteFile(filepath.Join(staging, "package.json"), piPackage, 0600); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(staging, "package-lock.json"), piLock, 0600); err != nil {
		return "", err
	}
	installing := startActivity(out, "Installing Pi", "Installed Pi")
	cmd := exec.Command(npm, "ci", "--no-audit", "--no-fund", "--loglevel=error")
	cmd.Dir = staging
	if output, err := cmd.CombinedOutput(); err != nil {
		installing.fail()
		_, _ = out.Write(output)
		return "", fmt.Errorf("installing Pi with npm ci: %w", err)
	}
	installing.succeed()
	if err := os.Rename(staging, dir); err != nil {
		return "", err
	}
	return executable, nil
}

func piConfig(p Paths, s Status) (string, string, error) {
	if err := validateClientStatus(s); err != nil {
		return "", "", err
	}
	dir := filepath.Join(p.Data, "clients", "pi", Version)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", "", err
	}
	extension := filepath.Join(dir, "flint-pi.ts")
	for _, name := range []string{"flint-pi.ts", "pi-transport.mjs", "pi-logo.mjs"} {
		data, err := clientAssets.ReadFile("clients/pi/" + name)
		if err != nil {
			return "", "", err
		}
		if err := atomicWrite(filepath.Join(dir, name), data, 0600); err != nil {
			return "", "", err
		}
	}
	if err := atomicWrite(filepath.Join(dir, "logo-pixels.json"), recursiveLogo, 0600); err != nil {
		return "", "", err
	}
	config, err := json.Marshal(map[string]any{
		"baseUrl": s.Endpoint, "contextWindow": s.Context, "maxTokens": s.Context,
	})
	return extension, string(config), err
}

func piUser(user string) harness {
	return harness{
		unset: []string{"PI_CODING_AGENT_SESSION_DIR"},
		set:   [][2]string{{"PI_CODING_AGENT_DIR", user}, {"PI_SKIP_VERSION_CHECK", "1"}},
	}
}
