# streamxform — streaming request body transform

A shared module (`github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform`) for plugins that read or
rewrite JSON request bodies **while the body is still arriving**, so they no longer buffer the whole
request. Memory stays independent of request size; untouched bytes are forwarded verbatim.

Used by `ai-proxy` (protocol conversion), `model-router` (find and rewrite `model` at the start of the body)
and `ai-statistics` (count user turns and pick up `model` without touching the body).

## Layers

| Layer | Files | Role |
|---|---|---|
| Scanner + writer | `engine.go`, `writer.go`, `action.go`, `jsonutil.go` | Protocol-agnostic. Walks the JSON byte stream, dispatches every key / array element of an *entered* container to the protocol, and executes the returned action. The writer builds output lazily: a container that never receives a write leaves no trace. |
| Protocols | `proto_claude.go`, `proto_gemini.go`, `proto_qwen.go`, `proto_openai.go`, `proto_openai_variants.go` | Hand-written, one per target format, derived line by line from the existing buffered transforms (`buildClaudeTextGenRequest`, `buildGeminiChatRequest`, `buildQwenTextGenerationRequest`, `defaultTransformRequestBody` …). No rule tables. |
| Guard | `guard/guard.go` | Drives a transformer from wasm-go's `ProcessStreamingRequestBodyWithAction` hook: holds the request headers (ActionPause) until a 64KB commit point, calls the plugin's `OnCommit` to apply header / context side effects, falls back to the plugin's buffered handler when a shape is unsupported before the commit point, fails the request (500) after it. Three shapes: `Transform` (whole body), `PrefixTransform` (rewrites at the start, rest forwarded without scanning), `Observe` (read-only, input forwarded as is). |

## Actions a protocol can return

| Action | Effect | Buffering |
|---|---|---|
| `Pass` | key + value forwarded verbatim (`As` renames, `Wrap` adds prefix/suffix, `Inner` strips quotes, `At` writes to an outer output level) | none |
| `Skip` | dropped | none |
| `Enter` | descend; children are dispatched too (`Lazy`: omit if empty, `Flat`: no output level, `Lenient`: scalar becomes `Pass`, `Via(hook)`: route the whole subtree to a sub-hook) | none |
| `Probe` | look at the value type first (string / object / array / null / bool / number), then decide (`OnStart`) | none |
| `Observe(cap)` | `Pass` plus a copy handed to `OnValue` | bounded |
| `Capture(cap)` | value collected and handed to `OnValue`; nothing written | bounded |
| `Defer(cap)` | key + value held; re-dispatched when the protocol calls `Release` (e.g. `content` arriving before `role`) | bounded |
| `Prefix(cap)` | string: first `cap` bytes handed to `OnPrefix`, which decides how the rest streams (e.g. `data:` image URLs) | window only |
| `Bail` | unsupported → fall back / fail | — |

Small fields (model, role, thinking config …) are captured and rewritten with the same Go structs the
buffered path uses, so `omitempty` and field shapes match byte for byte. Long strings, base64 attachments,
`tools` and `stop` arrays never enter memory: `tools` is streamed element by element through a reusable
sub-hook (`hook_tools.go`, `ToolsHook`) that the Claude and Gemini protocols mount with `Enter().Via(&hook)`;
the engine routes every callback inside that subtree to the hook and hands the closing `OnLeave` back to the
protocol.

The scanner validates JSON syntax byte by byte with the same rejection surface as `encoding/json`
(literals, number grammar, escapes, control characters, whitespace) in dispatched frames and pass-through
regions alike, using constant state and no buffering. Invalid input bails: before the commit point it falls
back to the buffered path (the client gets the usual 400), after it the request fails with 500.
Keys with escape sequences are decoded before dispatch. Only UTF-8 validity is not checked.

## Correctness

Each protocol is checked against the original implementation in the same package tests:
hand-written cases × chunk sizes 1/7/4096 plus randomized fuzzing (`ai-proxy/provider/streamxform_*_test.go`,
`STREAMXFORM_FUZZ_N` scales the fuzz size). `ai-proxy/streaming_request_test.go`, `model-router/stream_test.go`
and `ai-statistics/observer_*_test.go` drive the plugins through the wasm-go host emulator chunk by chunk
(Pause / Continue / fallback / 500 / passthrough); `KeyProbe` rewrites are compared byte for byte with `sjson`.
`bench_test.go` measures scanner throughput (long strings, base64, dense tool schemas); the string body is
scanned eight bytes at a time, so validation costs nothing on the bytes that dominate large requests.

Known deliberate differences from the buffered path: the buffered path type-checks the whole body against
its structs and returns 500 on any mismatch, the streaming path applies the same type rules only to the
fields it reads (null counts as absent, any other type mismatch bails); when `stream` appears after the
first 64KB the `Accept` header is not rewritten (providers decide streaming by the body);
`tools[].function.parameters` is forwarded verbatim instead of round-tripping through a map (the buffered
path sorts keys and reformats numbers through float64, losing precision on large integers) — same meaning,
the client's literals preserved.

## Using it from a plugin

```go
tr := streamxform.NewKeyProbe(streamxform.KeyProbeOptions{          // top-level keys only, rest verbatim
    Keys: map[string]int{"model": 4096}, ModelKey: "model",
    OnKey: func(t *streamxform.Transformer, key string, raw []byte) ([]byte, bool) { /* rewrite? */ },
})
st := guard.New(&guard.Plan{
    Tr: tr, Mode: guard.PrefixTransform,
    OnCommit: func(pre streamxform.Prelude, last bool) bool { /* set headers; false → fall back */ },
    Fallback: func(body []byte) types.Action { return onHttpRequestBody(ctx, cfg, body) }, // buffered path
    Metric: guard.NewMetric("my_plugin.stream"), Log: log.Warnf,
})
// in the hook registered with wrapper.ProcessStreamingRequestBodyWithAction:
return st.Feed(chunk, isLastChunk)
```

Register the streaming hook next to `ProcessRequestBody` (the buffered handler stays as the fallback) and
call `ctx.BufferRequestBody()` in the header phase for requests the streaming path should not handle.
`model-router/stream.go` and `ai-statistics/observer.go` are complete examples (about 100 lines each).

## Adding a protocol

1. Read the buffered transform for the provider and list every field it reads and every output field.
2. Write `proto_<name>.go`: embed `BaseProtocol`, dispatch by `t.Depth()` / `t.Last()`, capture small fields,
   stream long ones, put aggregated output in `Tail`. Mount sub-hooks (`ToolsHook`) with `Enter().Via` for
   shared OpenAI sub-structures. Anything you cannot express → `Bail`.
3. Add a differential test that calls the original builder and compares field by field.
4. Register it in `ai-proxy/provider/streamhooks.go` (`NewStreamPlan`), including any header / context side
   effects and the pre-commit requirements (`RequireModelBeforeCommit` / `RequireStreamBeforeCommit`).
