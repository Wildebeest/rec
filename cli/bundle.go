package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func verifyRuntime(dir string, m Manifest) error {
	for _, f := range m.RuntimeFiles {
		path := filepath.Join(dir, f.Name)
		if err := verifyFile(path, f); err != nil {
			return err
		}
		if f.Executable {
			st, err := os.Stat(path)
			if err != nil {
				return err
			}
			if st.Mode()&0100 == 0 {
				return fmt.Errorf("runtime is not executable")
			}
		}
	}
	return nil
}

func unpack(data []byte, dir string, m Manifest) error {
	if digest(data) != m.RuntimeSHA {
		return fmt.Errorf("runtime archive SHA-256 mismatch")
	}
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	allowed := map[string]File{}
	seen := map[string]bool{}
	for _, f := range m.RuntimeFiles {
		allowed[f.Name] = f
	}
	for {
		h, e := tr.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			return e
		}
		f, ok := allowed[h.Name]
		if !ok || seen[h.Name] || !safeRuntimeName(h.Name) || h.Typeflag != tar.TypeReg || h.Size != f.Size {
			return fmt.Errorf("unexpected runtime archive entry: %q", h.Name)
		}
		seen[h.Name] = true
		mode := os.FileMode(0600)
		if f.Executable {
			mode = 0700
		}
		if err := os.MkdirAll(filepath.Dir(filepath.Join(dir, h.Name)), 0700); err != nil {
			return err
		}
		dest, err := os.OpenFile(filepath.Join(dir, h.Name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
		if err != nil {
			return err
		}
		_, err = io.CopyN(dest, tr, h.Size)
		ce := dest.Close()
		if err != nil {
			return err
		}
		if ce != nil {
			return ce
		}
	}
	return verifyRuntime(dir, m)
}

func ensureRuntime(p Paths, m Manifest, override string) (string, error) {
	if override != "" {
		dir, err := filepath.Abs(override)
		if err != nil {
			return "", err
		}
		return dir, verifyRuntime(dir, m)
	}
	parent := filepath.Join(p.Data, "runtimes")
	if err := os.MkdirAll(parent, 0700); err != nil {
		return "", err
	}
	l, err := lock(filepath.Join(parent, "extract.lock"))
	if err != nil {
		return "", err
	}
	defer unlock(l)
	dir := filepath.Join(parent, m.RuntimeSHA)
	if verifyRuntime(dir, m) == nil {
		return dir, nil
	}
	if _, err := os.Lstat(dir); err == nil {
		return "", fmt.Errorf("cached runtime is corrupt; remove %s and retry", dir)
	}
	data, err := assets.ReadFile("assets/runtime.tar.gz")
	if err != nil {
		return "", fmt.Errorf("no embedded runtime; use packaging/package.py")
	}
	var size int64
	for _, f := range m.RuntimeFiles {
		size += f.Size
	}
	if err := diskSpace(parent, size+64*1024*1024); err != nil {
		return "", err
	}
	tmp, err := os.MkdirTemp(parent, ".extract-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmp)
	if err := unpack(data, tmp, m); err != nil {
		return "", err
	}
	return dir, os.Rename(tmp, dir)
}
