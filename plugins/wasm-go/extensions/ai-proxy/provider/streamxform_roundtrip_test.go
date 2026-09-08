package provider

// Differential tests of the struct round-trip stage against the real thing: json.Unmarshal into
// chatCompletionRequest and json.Marshal back, over the OpenAI corpus, the merge and context corpora and random
// requests, at chunk sizes 1, 7 and 4096; and the same for claudeTextGenRequest over the Claude conversion's output.

import (
	"encoding/json"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func officialRoundTripChat(in string) (map[string]any, bool) {
	req := &chatCompletionRequest{}
	if err := json.Unmarshal([]byte(in), req); err != nil {
		return nil, false
	}
	b, err := json.Marshal(req)
	if err != nil {
		return nil, false
	}
	m, err := decodeMap(b)
	return m, err == nil
}

func checkRoundTrip(t *testing.T, in string) {
	t.Helper()
	off, offOK := officialRoundTripChat(in)
	for _, chunk := range []int{1, 7, 4096} {
		tr := streamxform.NewStructRoundTrip(chatRequestFieldTree, nil)
		tr.SetFieldTree(chatRequestFieldTree)
		str, ok, why := runStream(tr, in, chunk)
		if !offOK {
			require.False(t, ok, "chunk=%d: the decode fails but the stage passed\n  %s", chunk, truncateIn(in))
			continue
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", chunk, why, truncateIn(in))
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", chunk, truncateIn(in))
	}
}

func TestRoundTripDifferential(t *testing.T) {
	big := strings.Repeat("r", 70000)
	for _, in := range []string{
		`{"model":"m","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"stream":false,"temperature":0,"top_p":0,"max_tokens":0,"n":0,"seed":0,"user":""}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"stream":true,"temperature":0.5,"stream_options":{"include_usage":false}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"stream_options":{"include_usage":true,"x":1}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"stream_options":{}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"stream_options":null}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"foo":{"bar":[1,2,null]},"tools":[{"x":1}]}`,
		`{"model":"m","messages":[{"role":"user","content":"U","name":"","extra":true},{"role":"","content":null},{"content":""}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`,
		`{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"f"}},{"index":2,"type":"function","function":{"name":"g","arguments":""}}]}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"U"},{"type":"image_url","image_url":{"url":"http://x/y.png","detail":"low"}}]}]}`,
		`{"model":"m","messages":[],"metadata":{},"stop":[],"logit_bias":{}}`,
		`{"model":"m","messages":null}`,
		`{"messages":[{"role":"user","content":"U"}]}`,
		`{}`,
		`{"model":"m","messages":[{"role":"user","content":"` + big + `"}],"temperature":1e0,"max_tokens":1.0}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"response_format":{"type":"json_schema","json_schema":{"name":"a","schema":{"type":"object"}}}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"f","description":"","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tool_choice":{"type":"function","function":{"name":"f"}}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"low","max_completion_tokens":5,"parallel_tool_calls":false}`,
		`{"model":5,"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","messages":"str"}`,
		`{"model":"m","messages":[{"role":"user","content":"U","tool_calls":"x"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a\"b\\cAé<&>"}],"user":"<"}`,
	} {
		checkRoundTrip(t, in)
	}
	for _, in := range mergeCases() {
		checkRoundTrip(t, in)
	}
	for _, in := range contextCases() {
		checkRoundTrip(t, in)
	}
	for _, c := range diffCases {
		if json.Valid([]byte(c.in)) {
			checkRoundTrip(t, c.in)
		}
	}
}

func TestRoundTripFuzz(t *testing.T) {
	r := rand.New(rand.NewSource(61))
	for i := 0; i < 300; i++ {
		checkRoundTrip(t, genRequest(r))
	}
	r2 := rand.New(rand.NewSource(62))
	for i := 0; i < 200; i++ {
		checkRoundTrip(t, genMergeRequest(r2))
	}
}

// With a root key replaced: what responseJsonSchema does between the decode and the marshal.
func TestRoundTripSet(t *testing.T) {
	schema := []byte(`{"type":"json_object"}`)
	in := `{"model":"m","messages":[{"role":"user","content":"U"}],"response_format":{"type":"text"},"foo":1}`
	req := &chatCompletionRequest{}
	require.NoError(t, json.Unmarshal([]byte(in), req))
	req.ResponseFormat = map[string]interface{}{"type": "json_object"}
	b, _ := json.Marshal(req)
	off, err := decodeMap(b)
	require.NoError(t, err)
	for _, chunk := range []int{1, 7, 4096} {
		tr := streamxform.NewStructRoundTrip(chatRequestFieldTree, map[string][]byte{"response_format": schema})
		str, ok, why := runStream(tr, in, chunk)
		require.True(t, ok, why)
		require.Empty(t, diffMaps(off, str), "chunk=%d", chunk)
	}
}

// The Claude request struct: the claude inserter's round trip.
func TestRoundTripClaude(t *testing.T) {
	for _, in := range []string{
		`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"U"}],"system":"S","temperature":0,"stream":false}`,
		`{"model":"m","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"text","text":"x"},{"type":"tool_use","id":"c2","name":"g","input":{}}]}]}`, // content blocks marshal themselves: verbatim, so a null input (which the real trip drops) is not in the corpus
		`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}]}],"system":[{"type":"text","text":"S"}]}`,
		`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"U"}],"tools":[{"name":"f","description":"d","input_schema":{"type":"object"}}],"tool_choice":{"type":"auto"},"thinking":{"type":"enabled","budget_tokens":1024},"metadata":{"user_id":"u"},"stop_sequences":[]}`,
	} {
		req := &claudeTextGenRequest{}
		require.NoError(t, json.Unmarshal([]byte(in), req), in)
		b, _ := json.Marshal(req)
		off, err := decodeMap(b)
		require.NoError(t, err)
		for _, chunk := range []int{1, 7, 4096} {
			tr := streamxform.NewStructRoundTrip(claudeRequestFieldTree, nil)
			str, ok, why := runStream(tr, in, chunk)
			require.True(t, ok, "chunk=%d: %s\n  %s", chunk, why, in)
			require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", chunk, in)
		}
	}
}
