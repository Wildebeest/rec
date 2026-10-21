import { getApiProvider } from '@mariozechner/pi-ai';
import { readFileSync } from 'node:fs';
import { registerLogo } from './pi-logo.mjs';
import { createRecStream } from './pi-transport.mjs';

export default function flint(pi) {
  registerLogo(pi, JSON.parse(readFileSync(new URL('./logo-pixels.json', import.meta.url), 'utf8')));
  const config = JSON.parse(process.env.REC_PI_CONFIG);
  const key = process.env.REC_API_KEY;
  if (!key) throw new Error('Rec API key is missing; use rec launch pi.');
  const native = getApiProvider('openai-completions');
  if (!native) throw new Error('Pi OpenAI-compatible transport is unavailable.');
  pi.registerProvider('rec', {
    name: 'Recursive local', api: 'rec-flint-completions',
    baseUrl: config.baseUrl, apiKey: 'REC_API_KEY',
    streamSimple: createRecStream(native, config, key),
    models: [{
      id: 'flint', name: 'Flint', reasoning: false, input: ['text'],
      contextWindow: config.contextWindow, maxTokens: config.maxTokens,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      compat: {
        supportsStore: false, supportsDeveloperRole: false,
        supportsReasoningEffort: false, supportsStrictMode: false,
        maxTokensField: 'max_tokens',
      },
    }],
  });
}
