import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { stripVTControlCharacters as plain } from 'node:util';
import { createLogoHeader, registerLogo } from '../cli/clients/pi/pi-logo.mjs';

const shapes = JSON.parse(readFileSync(new URL('../cli/clients/opencode/recursive-logo/pixels.json', import.meta.url), 'utf8'));
const theme = {
  name: 'dark', getColorMode: () => 'truecolor', fg: (_color, text) => text, bold: text => text,
};

test('logo stays within the viewport and uses the existing artwork', () => {
  const tui = { terminal: { rows: 32 } };
  const header = createLogoHeader(tui, () => theme, shapes, false);
  for (const [width, height, columns, rows] of [
    [110, 48, 20, 12], [60, 32, 20, 12], [58, 28, 16, 10], [38, 24, 16, 10],
    [28, 24, 0, 4], [110, 20, 0, 4], [9, 12, 0, 2], [60, 11, 0, 0], [8, 24, 0, 0], [110, 32, 20, 12],
  ]) {
    tui.terminal.rows = height;
    const lines = header.render(width).map(plain);
    assert.equal(lines.length, rows);
    assert.ok(lines.every(line => line.length <= width));
    if (rows) assert.match(lines.join('\n'), /recursive/);
    if (columns) {
      const expected = columns === 20 ? shapes.large : shapes.small;
      assert.deepEqual(lines.slice(0, -2).map(line => line.match(/[\u2800-\u28ff]+/)[0]), expected.map(row => row.map(cell => cell.glyph).join('')));
    } else {
      assert.doesNotMatch(lines.join('\n'), /[\u2800-\u28ff]/);
    }
  }
});

test('nine-second loop survives resizes and stops when the welcome screen ends', t => {
  t.mock.timers.enable({ apis: ['setInterval'] });
  let now = 0, renders = 0;
  t.mock.method(performance, 'now', () => now);
  const tui = { terminal: { rows: 32 }, requestRender: () => renders++ };
  const header = createLogoHeader(tui, () => theme, shapes, true);
  const first = header.render(110);
  now = 3000; t.mock.timers.tick(50);
  assert.notDeepEqual(header.render(110), first);
  now = 9000; t.mock.timers.tick(50);
  assert.deepEqual(header.render(110), first);
  assert.equal(renders, 2);
  tui.terminal.rows = 11; header.render(110); t.mock.timers.tick(100);
  assert.equal(renders, 2);
  tui.terminal.rows = 32; header.render(110); t.mock.timers.tick(50);
  assert.equal(renders, 3);
  now = 12000; t.mock.timers.tick(50);
  const frozen = header.render(110);
  header.stop(); header.render(110); t.mock.timers.tick(100);
  assert.equal(renders, 4);
  assert.deepEqual(header.render(110), frozen);
  header.dispose();
});

test('theme changes and 256-color terminals retain readable palettes', () => {
  let current = theme;
  const header = createLogoHeader({ terminal: { rows: 32 } }, () => current, shapes, false);
  const dark = header.render(110);
  current = { ...theme, name: 'light' };
  const light = header.render(110);
  assert.notDeepEqual(light, dark);
  assert.deepEqual(light.map(plain), dark.map(plain));
  current = { ...theme, getColorMode: () => '256color' };
  const limited = header.render(110).join('\n');
  assert.match(limited, /\x1b\[38;5;/);
  assert.doesNotMatch(limited, /\x1b\[38;2;/);
});

test('headless and resumed sessions stay still; turns, shell commands and shutdown stop animation', t => {
  t.mock.timers.enable({ apis: ['setInterval'] });
  const events = {};
  registerLogo({ on: (event, callback) => { events[event] = callback; } }, shapes);
  events.session_start({}, { hasUI: false });
  events.agent_start(); events.session_shutdown();
  events.session_start({}, { hasUI: true, sessionManager: { getBranch: () => [] }, ui: { setHeader() {} } });
  events.agent_start(); events.session_shutdown();
  let header, renders = 0;
  const tui = { terminal: { rows: 32 }, requestRender: () => renders++ };
  const ctx = { hasUI: true, sessionManager: { getBranch: () => [{ type: 'message' }] },
    ui: { theme, setHeader: factory => { header = factory(tui); } } };
  events.session_start({}, ctx); header.render(110); t.mock.timers.tick(100);
  assert.equal(renders, 0);
  events.session_shutdown();
  ctx.sessionManager.getBranch = () => [];
  events.session_start({}, ctx); header.render(110); t.mock.timers.tick(50);
  assert.equal(renders, 1);
  events.agent_start(); header.render(110); t.mock.timers.tick(100);
  assert.equal(renders, 1);
  events.session_shutdown();
  events.session_start({}, ctx); header.render(110); t.mock.timers.tick(50);
  assert.equal(renders, 2);
  events.user_bash?.(); header.render(110); t.mock.timers.tick(100);
  assert.equal(renders, 2);
  events.session_shutdown();
  events.session_start({}, ctx); header.render(110); t.mock.timers.tick(50);
  assert.equal(renders, 3);
  header.dispose(); t.mock.timers.tick(100);
  assert.equal(renders, 3);
});
