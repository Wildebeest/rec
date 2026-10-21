import assert from 'node:assert/strict';
import test from 'node:test';
import { createRecStream } from '../cli/clients/pi/pi-transport.mjs';

test('Rec transport leaves sampling to the engine after user hooks and overrides stale auth', async () => {
  let observed;
  const native = { streamSimple(model, context, options) { observed = { model, options }; return 'stream'; } };
  const config = { maxTokens: 8192 };
  const stream = createRecStream(native, config, 'current-key');
  assert.throws(() => stream({ provider: 'other', id: 'flint' }, {}), /only supports rec\/flint/);
  assert.equal(observed, undefined);
  let called = false;
  const options = {
    apiKey: 'old', headers: { Authorization: 'Bearer old', 'X-Test': 'retained' },
    onPayload: async (payload) => { called = true; return { ...payload, top_p: 0.95, top_k: 20, max_completion_tokens: 100, custom: true }; },
  };
  const model = { provider: 'rec', id: 'flint', api: 'rec-flint-completions' };
  assert.equal(stream(model, {}, options), 'stream');
  assert.equal(observed.model.api, 'openai-completions');
  assert.equal(model.api, 'rec-flint-completions');
  assert.equal(observed.options.apiKey, 'current-key');
  assert.deepEqual(observed.options.headers, { Authorization: 'Bearer current-key', 'X-Test': 'retained' });
  const payload = { model: 'flint', messages: [], temperature: 0.2 };
  const result = await observed.options.onPayload(payload, observed.model);
  assert.equal(called, true);
  assert.deepEqual(result, { model: 'flint', messages: [], max_tokens: 8192, custom: true });
  assert.equal(payload.temperature, 0.2);
  stream(model, {});
  assert.equal((await observed.options.onPayload(payload, observed.model)).max_tokens, 8192);
});
