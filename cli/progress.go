package main

import (
	"fmt"
	"io"
	"sync"
	"time"
)

func redrawInterval(s *screen) time.Duration {
	if s.live {
		return 100 * time.Millisecond
	}
	return 10 * time.Second
}

type transfer struct {
	screen           *screen
	label, doneLabel string
	detail           string
	n, total, start  int64
	started, last    time.Time
}

func newTransfer(out io.Writer, label, doneLabel string, offset, total int64) *transfer {
	t := &transfer{screen: newScreen(out), label: label, doneLabel: doneLabel, n: offset, total: total, start: offset, started: time.Now()}
	t.draw()
	return t
}

func (t *transfer) fraction() float64 {
	return min(1, max(0, float64(t.n)/float64(max(1, t.total))))
}

func (t *transfer) draw() {
	t.last = time.Now()
	amount := fmt.Sprintf("%3.0f%%  %s / %s", t.fraction()*100, formatBytes(t.n), formatBytes(t.total))
	if !t.screen.live {
		text := "  " + t.label + "  " + amount
		if t.detail != "" {
			text += "  " + t.detail
		}
		t.screen.println(text)
		return
	}
	text := "  " + accentStyle.Render(spinnerFrames[int(time.Since(t.started)/(100*time.Millisecond))%len(spinnerFrames)]) + " " + t.label + "  " + bar(t.fraction()) + "  " + amount
	elapsed := time.Since(t.started)
	if rate := float64(t.n-t.start) / elapsed.Seconds(); elapsed > 2*time.Second && rate > 0 {
		remaining := time.Duration(float64(t.total-t.n) / rate * float64(time.Second))
		text += faintStyle.Render(fmt.Sprintf("  %s/s  %s left", formatBytes(int64(rate)), formatDuration(remaining)))
	}
	if t.detail != "" {
		text += faintStyle.Render("  " + t.detail)
	}
	t.screen.redraw(text)
}

func (t *transfer) Write(b []byte) (int, error) {
	t.n += int64(len(b))
	if time.Since(t.last) >= redrawInterval(t.screen) {
		t.draw()
	}
	return len(b), nil
}

func (t *transfer) set(n int64, detail string) {
	t.n, t.detail = n, detail
	if t.screen.live {
		t.draw()
	}
}

func (t *transfer) note(text string) {
	t.clear()
	t.screen.step(warningStyle.Render("!"), text, "")
	t.draw()
}

func (t *transfer) clear() {
	if t.screen.live {
		t.screen.redraw("")
	}
}

func (t *transfer) succeed() {
	t.clear()
	t.screen.succeeded(t.doneLabel, formatBytes(t.total)+" in "+formatDuration(time.Since(t.started)))
}

func (t *transfer) fail(detail string) {
	t.clear()
	t.screen.failed(t.label+" failed", detail)
}

type activity struct {
	mu                       sync.Mutex
	screen                   *screen
	label, doneLabel, detail string
	started                  time.Time
	finished                 bool
	stop, done               chan struct{}
}

func startActivity(out io.Writer, label, doneLabel string) *activity {
	a := &activity{screen: newScreen(out), label: label, doneLabel: doneLabel, started: time.Now(), stop: make(chan struct{}), done: make(chan struct{})}
	a.draw()
	go func() {
		defer close(a.done)
		ticker := time.NewTicker(redrawInterval(a.screen))
		defer ticker.Stop()
		for {
			select {
			case <-a.stop:
				return
			case <-ticker.C:
				a.mu.Lock()
				a.draw()
				a.mu.Unlock()
			}
		}
	}()
	return a
}

func (a *activity) draw() {
	elapsed := time.Since(a.started)
	text := a.label
	if a.detail != "" {
		text += faintStyle.Render(" · " + a.detail)
	}
	if !a.screen.live {
		a.screen.println("  " + text + faintStyle.Render(" ("+formatDuration(elapsed)+")"))
		return
	}
	frame := spinnerFrames[int(elapsed/(100*time.Millisecond))%len(spinnerFrames)]
	a.screen.redraw("  " + accentStyle.Render(frame) + " " + text + faintStyle.Render("  "+formatDuration(elapsed)))
}

func (a *activity) phase(detail string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.finished || detail == a.detail {
		return
	}
	a.detail = detail
	a.draw()
}

func (a *activity) end(report func()) {
	close(a.stop)
	<-a.done
	a.mu.Lock()
	defer a.mu.Unlock()
	a.finished = true
	if a.screen.live {
		a.screen.redraw("")
	}
	report()
}

func (a *activity) succeed() {
	a.end(func() { a.screen.succeeded(a.doneLabel, formatDuration(time.Since(a.started))) })
}

func (a *activity) fail() { a.end(func() { a.screen.failed(a.label+" failed", "") }) }

func (a *activity) cancel() {
	a.end(func() { a.screen.step(warningStyle.Render("–"), a.label+" cancelled", "") })
}
