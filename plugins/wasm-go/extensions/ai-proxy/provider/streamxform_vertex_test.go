package provider

// Differential tests of the Vertex Anthropic Messages passthrough: the buffered onAnthropicMessagesRequestBody
// against the streaming OpenAI transformer with OmitModel and the VertexAnthropicVariant -- the body field by
// field, and the request path derived from the prelude against the one the buffered path wrote.

import (
	"math/rand"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func vertexAnthropicStream(v *vertexProvider) *streamxform.Transformer {
	return streamxform.NewOpenAI(streamxform.OpenAIOptions{
		MapModel:               func(m string) string { return getMappedModel(m, v.config.modelMapping) },
		DetectStream:           true,
		OmitModel:              true,
		DeveloperRoleSupported: true,
		Variant:                &streamxform.VertexAnthropicVariant{Version: vertexAnthropicVersion, DefaultMaxTokens: claudeDefaultMaxTokens},
	})
}

func vertexAnthropicCases() []string {
	big := strings.Repeat("x", 70000)
	return []string{
		`{"model":"claude-sonnet-4","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`,
		`{"messages":[{"role":"user","content":"hi"}],"model":"claude-sonnet-4"}`,
		`{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"hi"}],"system":"S"}`,
		`{"model":"claude-sonnet-4","anthropic_version":"2023-06-01","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"claude-sonnet-4","context_management":{"edits":[{"type":"clear_tool_uses_20250919"}]},"messages":[{"role":"user","content":"hi"}],"max_tokens":1}`,
		`{"model":"unmapped","messages":[{"role":"user","content":"` + big + `"}],"stream":false,"max_tokens":8}`,
		`{"model":"claude-sonnet-4-5","tools":[{"type":"web_search_20250305"},{"name":"f","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]}]}`,
		`{"model":"claude-sonnet-4","max_tokens":0,"messages":[]}`,
		`{"max_tokens":5,"messages":[{"role":"user","content":"` + big + `"}],"model":"claude-sonnet-4","stream":true}`,
		`{"model":"claude-sonnet-4","thinking":{"type":"enabled","budget_tokens":2048},"metadata":{"user_id":"u"},"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":9}`,
	}
}

func checkVertexAnthropic(t *testing.T, v *vertexProvider, in string) {
	t.Helper()
	hdr := http.Header{}
	offBody, err := v.onAnthropicMessagesRequestBody(newMapCtx(), []byte(in), hdr)
	require.NoError(t, err, in)
	off, err := decodeMap(offBody)
	require.NoError(t, err, in)
	for _, cs := range []int{1, 7, 4096} {
		tr := vertexAnthropicStream(v)
		str, ok, why := runStream(tr, in, cs)
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d body differs\n  %s", cs, in)
		pre := tr.Protocol().(streamxform.Preluder).Prelude()
		require.True(t, pre.ModelSeen, in)
		path := v.getAhthropicRequestPath(newMapCtx(), ApiNameAnthropicMessages, getMappedModel(pre.Model, v.config.modelMapping), pre.Stream)
		require.Equal(t, hdr.Get(":path"), path, "chunk=%d path differs\n  %s", cs, in)
	}
}

func TestVertexAnthropicDifferential(t *testing.T) {
	v := newAnthropicVertexProvider(false)
	for _, in := range vertexAnthropicCases() {
		checkVertexAnthropic(t, v, in)
	}
}

// Random key order and presence: the buffered path is order-blind (gjson / sjson), the streaming one must be too.
func TestVertexAnthropicFuzz(t *testing.T) {
	v := newAnthropicVertexProvider(false)
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	fields := []string{
		`"model":"claude-sonnet-4"`, `"max_tokens":77`, `"stream":true`, `"stream":false`,
		`"anthropic_version":"old"`, `"context_management":{"edits":[]}`, `"system":"S"`,
		`"messages":[{"role":"user","content":"` + strings.Repeat("m", 3000) + `"}]`,
		`"metadata":{"user_id":"u"}`, `"temperature":0.5`, `"stop_sequences":["a"]`,
	}
	for i := 0; i < fuzzN(200); i++ {
		var parts []string
		seen := map[string]bool{}
		for _, f := range rnd.Perm(len(fields)) {
			k := fields[f][:strings.Index(fields[f], ":")]
			if seen[k] || rnd.Intn(3) == 0 {
				continue
			}
			seen[k] = true
			parts = append(parts, fields[f])
		}
		if !seen[`"model"`] {
			parts = append(parts, `"model":"claude-sonnet-4-5"`)
		}
		checkVertexAnthropic(t, v, "{"+strings.Join(parts, ",")+"}")
	}
}
