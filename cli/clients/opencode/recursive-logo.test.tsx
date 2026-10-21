// Staged into the pinned OpenCode packages/tui/test by the build script.
import { expect, test } from "bun:test"
import path from "node:path"
import { createAppFixture } from "./fixture/app"

test("Recursive logo resizes without hiding the home prompt", async () => {
  await using setup = await createAppFixture({
    width: 110,
    height: 48,
    config: {
      animations: false,
      tabs: { mode: "off" },
      plugins: [{ package: path.join(import.meta.dir, "fixture/plugin/recursive-logo"), options: { animate: false } }],
    },
  })
  await setup.ready
  for (const [width, height, columns, rows] of [
    [110, 48, 20, 10], [110, 40, 20, 10], [110, 32, 20, 10],
    [58, 28, 16, 8], [38, 24, 16, 8], [28, 24, 0, 0],
    [110, 20, 0, 0], [110, 48, 20, 10],
  ]) {
    setup.resize(width, height)
    await setup.waitForFrame((frame) => {
      if (!frame.includes("recursive") || !frame.includes("Ask anything")) return false
      const logo = frame.split("\n").filter((line) => /[\u2800-\u28ff]/.test(line))
      if (!columns) return logo.length === 0
      return logo.length === rows && logo.every((line) => line.match(/[\u2800-\u28ff]+/)?.[0].length === columns)
    })
    if (width >= 60) expect(setup.captureCharFrame()).toContain("commands")
    expect(setup.captureCharFrame()).not.toContain("█▀▀█ █▀▀█ █▀▀█ █▀▀▄")
  }
}, 20_000)

test("stock OpenCode keeps its logo when the plugin is absent", async () => {
  await using setup = await createAppFixture({ config: { animations: false } })
  await setup.ready
  await setup.waitForFrame((frame) => frame.includes("█▀▀█") && frame.includes("Ask anything"))
  expect(setup.captureCharFrame()).not.toContain("recursive")
})
