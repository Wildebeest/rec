//go:build darwin || linux

package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// Everything rec writes lives under Data; uninstall removes that one directory.
type Paths struct{ Data, State, Config string }

func paths() (Paths, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Paths{}, err
	}
	root := filepath.Join(home, ".local/share")
	if p := os.Getenv("XDG_DATA_HOME"); filepath.IsAbs(p) {
		root = p
	}
	p := dataPaths(filepath.Join(root, "rec"))
	for _, dir := range []string{p.Data, p.State, p.Config} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return p, err
		}
		if err := os.Chmod(dir, 0700); err != nil {
			return p, err
		}
	}
	return p, nil
}

func dataPaths(data string) Paths {
	return Paths{data, filepath.Join(data, "state"), filepath.Join(data, "config")}
}

func lock(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		return nil, fmt.Errorf("another rec operation holds %s", filepath.Base(path))
	}
	return f, nil
}
func unlock(f *os.File) { _ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN); f.Close() }

func atomicWrite(path string, data []byte, mode os.FileMode) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".rec-tmp-")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if err = f.Chmod(mode); err == nil {
		_, err = f.Write(data)
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(name, path)
}

func token(p Paths, create bool) (string, error) {
	f, err := lock(filepath.Join(p.Config, "key.lock"))
	if err != nil {
		return "", err
	}
	defer unlock(f)
	path := filepath.Join(p.Config, "api-key")
	b, err := os.ReadFile(path)
	if err == nil {
		s := strings.TrimSpace(string(b))
		raw, e := hex.DecodeString(s)
		if e != nil || len(raw) != 32 {
			return "", fmt.Errorf("invalid local key file")
		}
		return s, os.Chmod(path, 0600)
	}
	if !os.IsNotExist(err) || !create {
		return "", fmt.Errorf("no local key; run a model first: %w", err)
	}
	raw := make([]byte, 32)
	if _, err = rand.Read(raw); err != nil {
		return "", err
	}
	s := hex.EncodeToString(raw)
	return s, atomicWrite(path, []byte(s+"\n"), 0600)
}

func diskSpace(dir string, need int64) error {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return err
	}
	free := uint64(st.Bavail) * uint64(st.Bsize)
	if need < 0 || free < uint64(need) {
		return fmt.Errorf("insufficient disk: need %d bytes, available %d", need, free)
	}
	return nil
}
