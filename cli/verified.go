package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

type fileStamp struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
	Device uint64 `json:"device"`
	Inode  uint64 `json:"inode"`
	Mtime  int64  `json:"mtime_ns"`
	Ctime  int64  `json:"ctime_ns"`
}

type verifiedFiles struct {
	root   string
	stamps map[string]fileStamp
}

func statStamp(path, sha string) (fileStamp, error) {
	var st unix.Stat_t
	if err := unix.Lstat(path, &st); err != nil {
		return fileStamp{}, err
	}
	return fileStamp{SHA256: sha, Size: st.Size, Device: uint64(st.Dev), Inode: st.Ino, Mtime: st.Mtim.Nano(), Ctime: st.Ctim.Nano()}, nil
}

func loadVerified(root string) (*verifiedFiles, error) {
	v := &verifiedFiles{root: root, stamps: map[string]fileStamp{}}
	data, err := os.ReadFile(filepath.Join(root, "verified.json"))
	if errors.Is(err, fs.ErrNotExist) {
		return v, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &v.stamps); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Join(root, "verified.json"), err)
	}
	return v, nil
}

// A stamp skips hashing only while the file keeps its inode, size, mtime and ctime; userland cannot set ctime.
func (v *verifiedFiles) trusted(f File) bool {
	stamp, ok := v.stamps[f.Name]
	if !ok || stamp.SHA256 != f.SHA256 || stamp.Size != f.Size {
		return false
	}
	current, err := statStamp(filepath.Join(v.root, f.Name), f.SHA256)
	return err == nil && current == stamp
}

func (v *verifiedFiles) record(f File) error {
	stamp, err := statStamp(filepath.Join(v.root, f.Name), f.SHA256)
	if err != nil {
		return err
	}
	v.stamps[f.Name] = stamp
	return v.save()
}

// The stamp is taken before hashing so a write during the hash invalidates it.
func (v *verifiedFiles) verify(f File, progress io.Writer) error {
	path := filepath.Join(v.root, f.Name)
	stamp, err := statStamp(path, f.SHA256)
	if err != nil {
		return err
	}
	if err := verifyFileInto(path, f, progress); err != nil {
		return err
	}
	v.stamps[f.Name] = stamp
	return v.save()
}

func (v *verifiedFiles) forget(f File) error {
	delete(v.stamps, f.Name)
	return v.save()
}

func (v *verifiedFiles) save() error {
	data, err := json.Marshal(v.stamps)
	if err != nil {
		return err
	}
	return atomicWrite(filepath.Join(v.root, "verified.json"), data, 0600)
}
