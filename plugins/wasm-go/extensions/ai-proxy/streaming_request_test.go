package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	wasmtest "github.com/higress-group/wasm-go/pkg/test"
	"github.com/stretchr/testify/require"
)

// Integration tests of streaming request body transformation: the host emulator feeds the body chunk by chunk and the
// control flow is checked: Pause before the commit point / Continue after it, fallback to the buffered path, failure after the commit point, passthrough.
// Equivalence of the converted output with the buffered implementation is covered by the differential tests in the provider package; only the integration layer is checked here.

var streamingClaudeConfig = json.RawMessage(`{"provider":{"type":"claude","apiTokens":["sk-test"],"modelMapping":{"m":"claude-3"}}}`)
var streamingGenericConfig = json.RawMessage(`{"provider":{"type":"generic","genericHost":"generic.example.com","apiTokens":["t"]}}`)

func streamingRequestHeaders(path string) [][2]string {
	return [][2]string{
		{":authority", "example.com"},
		{":path", path},
		{":method", "POST"},
		{"Content-Type", "application/json"},
		{"Content-Length", "1"},
	}
}

// feedChunks feeds the body to the plugin in chunks of the given size and returns the action per chunk and the concatenated upstream body.
func feedChunks(host wasmtest.TestHost, body []byte, chunk int) (actions []types.Action, upstream []byte) {
	for i := 0; i < len(body); i += chunk {
		j := i + chunk
		if j > len(body) {
			j = len(body)
		}
		a := host.CallOnHttpStreamingRequestBody(body[i:j], j == len(body))
		actions = append(actions, a)
		if a == types.ActionContinue {
			upstream = append(upstream, host.GetRequestBody()...)
		}
	}
	return
}

func requestHeader(host wasmtest.TestHost, name string) string {
	for _, h := range host.GetRequestHeaders() {
		if strings.EqualFold(h[0], name) {
			return h[1]
		}
	}
	return ""
}

func TestStreamingRequest_ClaudeSdkOrder(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions")))

		// field order of the official SDK: messages first, model / stream at the end; 200KB of content, far beyond the commit point
		big := strings.Repeat("y", 200000)
		body := `{"messages":[{"role":"system","content":"S"},{"role":"user","content":"` + big + `"}],"model":"m","stream":true,"max_tokens":50}`
		actions, upstream := feedChunks(host, []byte(body), 4096)

		// always Pause before the commit point (64KB), Continue after it
		pauses := 0
		for _, a := range actions {
			if a == types.ActionPause {
				pauses++
			} else {
				break
			}
		}
		// the chunk that crosses the commit point returns Continue itself: every earlier chunk Paused, and the total lands right around 64KB
		require.Less(t, pauses*4096, 64<<10, "should keep Pausing before the commit point")
		require.GreaterOrEqual(t, (pauses+1)*4096, 64<<10, "should release as soon as the commit point is passed, not accumulate further")
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		require.Greater(t, len(actions)-pauses, 1, "should Continue chunk by chunk after the commit point instead of accumulating to the end")

		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out), "the upstream body should be valid JSON: %s", truncate(upstream))
		require.Equal(t, "claude-3", out["model"])
		require.Equal(t, "S", out["system"])
		require.Equal(t, true, out["stream"])
		require.Equal(t, float64(50), out["max_tokens"])
		msgs := out["messages"].([]any)
		require.Len(t, msgs, 1)
		require.Equal(t, "user", msgs[0].(map[string]any)["role"])
		require.Len(t, msgs[0].(map[string]any)["content"].(string), 200000)
	})
}

func TestStreamingRequest_AcceptHeaderWhenStreamKnownEarly(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := `{"model":"m","stream":true,"messages":[{"role":"user","content":"` + strings.Repeat("y", 200000) + `"}]}`
		_, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, "text/event-stream", requestHeader(host, "Accept"), "Accept should be rewritten when stream is known before the commit point")
		require.True(t, json.Valid(upstream))
	})
}

func TestStreamingRequest_FallbackBeforeCommit(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		// claude_content_blocks is not reproduced by streaming and appears before the commit point → Pause throughout, the last chunk takes the buffered path
		body := `{"model":"m","messages":[{"role":"user","claude_content_blocks":[{"type":"text","text":"B"}],"content":"` + strings.Repeat("y", 200000) + `"}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for _, a := range actions[:len(actions)-1] {
			require.Equal(t, types.ActionPause, a)
		}
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		msgs := out["messages"].([]any)
		blocks := msgs[0].(map[string]any)["content"].([]any)
		require.Equal(t, "B", blocks[0].(map[string]any)["text"], "after the fallback this should be the buffered mapping of claude_content_blocks")
		require.Equal(t, "claude-3", out["model"])
	})
}

func TestStreamingRequest_FailAfterCommit(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		// the unsupported shape appears after 200KB: bytes already released, only failure is possible
		body := `{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("y", 200000) + `"},{"role":"user","content":"U","claude_content_blocks":[{"type":"text","text":"B"}]}]}`
		actions, _ := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[len(actions)-1])
		resp := host.GetLocalResponse()
		require.NotNil(t, resp, "a local 500 response should be sent")
		require.Equal(t, uint32(500), resp.StatusCode)
		require.Contains(t, string(resp.Data), "bailed after commit")
	})
}

func TestStreamingRequest_GenericPassthrough(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingGenericConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := "this is not even json " + strings.Repeat("z", 100000)
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for _, a := range actions {
			require.Equal(t, types.ActionContinue, a, "generic should release every chunk directly")
		}
		require.Equal(t, body, string(upstream))
	})
}

func TestStreamingRequest_SingleChunk(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
		require.Equal(t, types.ActionContinue, host.CallOnHttpStreamingRequestBody([]byte(body), true))
		var out map[string]any
		require.NoError(t, json.Unmarshal(host.GetRequestBody(), &out))
		require.Equal(t, "claude-3", out["model"])
		require.Equal(t, float64(4096), out["max_tokens"])
		require.Len(t, out["tools"].([]any), 1)
	})
}

func truncate(b []byte) string {
	if len(b) > 200 {
		return string(b[:200]) + "..."
	}
	return string(b)
}
