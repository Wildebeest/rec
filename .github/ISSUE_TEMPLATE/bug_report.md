---
name: Bug report
about: Report a problem with installing or running rec.
labels: bug
---

<!--
Remove API keys, Hugging Face tokens, and private prompts from everything you paste.
Malformed-request diagnostics can include the request body.

Do not report security vulnerabilities here. Read the security policy instead:
https://github.com/recursive-oss/rec/security/policy
-->

## What happened

<!-- Describe the problem and what you expected. If it worked in an earlier version, give that version. -->

## Steps to reproduce

<!-- The exact commands, in order. Include the client and request if the problem happens during generation. -->

## Output

<!--
Paste the full output, not only the last line.
`rec launch` writes server output to `launch-server.log` in the State directory that `rec doctor` prints.
Also paste `rec doctor`, and `rec status` if the server is running.
If installation failed, paste the installer output instead.
-->

```
```

## Environment

<!-- Chip, memory, and OS. On macOS: `sysctl -n machdep.cpu.brand_string hw.memsize` and `sw_vers`. Say if you built from source. -->

- [ ] I removed API keys, tokens, and private content from this report.
