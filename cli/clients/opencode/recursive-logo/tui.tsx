import { Plugin } from "@opencode/plugin/tui"
import { RGBA, TextAttributes } from "@opentui/core"
import { useTerminalDimensions } from "@opentui/solid"
import { createEffect, createMemo, createSignal, For, onCleanup, Show } from "solid-js"
import shapes from "./pixels.json"

export default Plugin.define({
  id: "recursive.flint-logo",
  setup(context) {
    context.ui.slot({
      replace: "home.logo",
      render: () => {
        const dimensions = useTerminalDimensions()
        const [phase, setPhase] = createSignal(0)
        const start = performance.now()
        const graphic = () => dimensions().width >= 38 && dimensions().height >= 24
        const large = () => dimensions().width >= 60 && dimensions().height >= 32
        const shape = () => large() ? shapes.large : shapes.small
        createEffect(() => {
          if (!graphic() || context.options.animate === false) return
          const tick = () => setPhase(((performance.now() - start) / 9000) % 1)
          tick()
          const timer = setInterval(tick, 50)
          onCleanup(() => clearInterval(timer))
        })
        const palette = createMemo(() => {
          const low = context.themeMode === "dark" ? [54, 42, 78] : [206, 196, 220]
          const high = context.themeMode === "dark" ? [227, 213, 255] : [70, 48, 108]
          return Array.from({ length: 256 }, (_, alpha) => RGBA.fromInts(
            ...low.map((value, i) => Math.round(value + (high[i] - value) * alpha / 255)) as [number, number, number],
          ))
        })
        const color = (face: number, x: number, y: number) => {
          if (face === 3) return palette()[0]
          const offset = face / 3 + (x - 0.5) * 0.06 + (y - 0.5) * 0.03
          const glow = (1 + Math.cos((phase() - offset) * Math.PI * 2)) / 2
          return palette()[Math.round(glow * 255)]
        }
        return (
          <Show when={dimensions().height >= 12}>
            <box flexDirection="row" alignItems="center" gap={2}>
              <Show when={graphic()}>
                <box>
                  <For each={shape()}>
                    {(row, y) => (
                      <text selectable={false}>
                        <For each={row}>
                          {(cell, x) => <span style={{ fg: color(cell.owner, x() / (row.length - 1), y() / (shape().length - 1)) }}>{cell.glyph}</span>}
                        </For>
                      </text>
                    )}
                  </For>
                </box>
              </Show>
              <box>
                <text fg={context.theme.text.base} attributes={TextAttributes.BOLD} selectable={false}>
                  recursive
                </text>
                <text fg={context.theme.text.muted} selectable={false}>
                  flint
                </text>
              </box>
            </box>
          </Show>
        )
      },
    })
  },
})
