package provider

// Differential tests of the OpenAI → Bedrock Converse conversion against buildBedrockTextGenerationRequest: the
// Claude corpus plus Bedrock-specific shapes (tool calls in either order, tool results and their merging, images,
// thinking, tool_choice, output config), at chunk sizes 1, 7 and 4096.

import (
	"errors"
	"math/rand"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func newTokenBedrockProvider(pcr string) *bedrockProvider {
	return &bedrockProvider{config: ProviderConfig{
		apiTokens:               []string{"t"},
		awsRegion:               "us-east-1",
		modelMapping:            map[string]string{"m": "anthropic.claude-3-5-sonnet", "nova": "amazon.nova-pro-v1:0", "empty": ""},
		bedrockAdditionalFields: map[string]interface{}{"top_k": 5},
		promptCacheRetention:    pcr,
	}}
}

// errBedrockPanic marks a body the buffered path cannot handle at all: buildBedrockTextGenerationRequest indexes
// Content[0] of the previous message when merging a tool result, and panics when that content is empty.
var errBedrockPanic = errors.New("buffered path panics")

func officialBedrock(b *bedrockProvider, in string) (m map[string]any, err error) {
	defer func() {
		if r := recover(); r != nil {
			m, err = nil, errBedrockPanic
		}
	}()
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest([]byte(in), req); err != nil {
		return nil, err
	}
	if req.Model == "" {
		return nil, errors.New("missing model in request")
	}
	req.Model = getMappedModel(req.Model, b.config.modelMapping)
	if req.Model == "" {
		return nil, errors.New("model becomes empty after applying the configured mapping")
	}
	out, err := b.buildBedrockTextGenerationRequest(req, http.Header{})
	if err != nil {
		return nil, err
	}
	return decodeMap(out)
}

func bedrockStream(b *bedrockProvider) *streamxform.Transformer {
	return typed(streamxform.NewBedrock(streamxform.BedrockOptions{
		MapModel:             b.config.mapStrict(),
		AdditionalFields:     b.config.bedrockAdditionalFields,
		PromptCacheRetention: b.config.promptCacheRetention,
		PromptCacheSupported: isPromptCacheSupportedModel,
	}))
}

func checkBedrock(t *testing.T, b *bedrockProvider, in string) {
	t.Helper()
	off, err := officialBedrock(b, in)
	if errors.Is(err, errBedrockPanic) {
		return // streaming produces a request where the buffered path crashes; nothing to compare against
	}
	for _, cs := range []int{1, 7, 4096} {
		str, ok, why := runStream(bedrockStream(b), in, cs)
		if err != nil {
			require.False(t, ok, "chunk=%d: buffered failed (%v) but streaming passed\n  %s", cs, err, in)
			continue
		}
		if !ok && (strings.Contains(why, "prompt cache") || strings.Contains(why, "tool_calls after the content of a non-assistant message")) {
			continue // by design: cache points need the whole conversation; only assistant content waits for tool_calls
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", cs, in)
	}
}

func bedrockCases() []string {
	big := strings.Repeat("b", 70000)
	return []string{
		`{"model":"m","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","messages":[{"role":"system","content":"S"},{"role":"user","content":"` + big + `"}],"stream":true,"max_tokens":100,"temperature":0.5,"top_p":0.9}`,
		`{"model":"m","messages":[{"role":"system","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},{"role":"system"},{"role":"user","content":""}],"max_completion_tokens":7,"max_tokens":3}`,
		`{"model":"m","messages":[{"role":"user","content":"q"},{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}},{"id":"c2","type":"function","function":{"name":"g","arguments":"bad"}}]},{"role":"tool","tool_call_id":"c1","content":"r1"},{"role":"tool","tool_call_id":"c2","content":[{"type":"text","text":"r2"},{"type":"image_url","image_url":{"url":"x"}}]},{"role":"user","content":"next"}]}`,
		`{"model":"m","messages":[{"tool_calls":[{"id":"c1","function":{"name":"f","arguments":"{}"}}],"role":"assistant"},{"role":"tool","content":null,"tool_call_id":"c1"},{"role":"system","content":"between"},{"role":"tool","tool_call_id":"c3","content":"r3"},{"role":"assistant","content":"done"}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"kept","tool_calls":[]},{"role":"assistant","content":"kept2","tool_calls":null}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"image_url","image_url":{"url":"https://x/y.png"}},{"type":"image_url","image_url":{"url":"data:bad"}},{"type":"input_audio","input_audio":{"data":"d"}},"str",{"image_url":{"url":"data:image/jpeg;base64,BB"},"type":"image_url"},{"type":"text","text":""}]}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"` + big + `"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("Q", 60000) + `"}}]}]}`,
		`{"model":"m","messages":[{"role":"user","content":null},{"role":"user"},{"role":"developer","content":"d"},{"role":"user","content":{"weird":1}}]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"low"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"medium","tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],"tool_choice":"required"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"high","tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":{"type":"function","function":{"name":"f"}}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}],"tool_choice":{"type":"function","function":{"name":"f"}}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"any"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"auto"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":null}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"claude_thinking":{"type":"enabled","budget_tokens":2048},"reasoning_effort":"low"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"claude_thinking":{"type":"adaptive","display":"summarized"},"claude_output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object"}}}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"claude_thinking":{"type":"disabled"},"claude_output_config":{"effort":"weird"}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"prompt_cache_retention":"24h"}`,
		`{"model":"nova","messages":[{"role":"user","content":"U"}],"prompt_cache_retention":"in_memory"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"prompt_cache_key":"k","n":2,"stop":["x"],"user":"u"}`,
		`{"model":"m","messages":[]}`,
		`{"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"empty","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"max_tokens":"x"}`,
	}
}

func TestBedrockDifferential(t *testing.T) {
	b := newTokenBedrockProvider("")
	for _, in := range bedrockCases() {
		checkBedrock(t, b, in)
	}
	for _, c := range diffCases {
		checkBedrock(t, b, c.in)
	}
	// prompt cache configured: supported models fall back, others stream as before
	bp := newTokenBedrockProvider("in_memory")
	for _, in := range bedrockCases()[:3] {
		checkBedrock(t, bp, in)
	}
}

func TestBedrockFuzz(t *testing.T) {
	b := newTokenBedrockProvider("")
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	msgs := []string{
		`{"role":"system","content":"S"}`, `{"role":"user","content":"u"}`, `{"role":"assistant","content":"a"}`,
		`{"role":"assistant","content":"c","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{\"k\":2}"}}]}`,
		`{"tool_calls":[{"id":"2","function":{"name":"h","arguments":"{}"}}],"role":"assistant"}`,
		`{"role":"tool","content":"r","tool_call_id":"1"}`, `{"role":"tool","content":[{"type":"text","text":"tr"}],"tool_call_id":"2"}`,
		`{"role":"user","content":[{"type":"text","text":"t"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}},{"type":"image_url","image_url":{"url":"https://h/p.jpg"}}]}`,
		`{"content":"first","role":"user"}`, `{"role":"user","content":null}`, `{"role":"developer","content":"d"}`,
	}
	tops := []string{`"stream":true`, `"temperature":0.4`, `"top_p":0.5`, `"max_tokens":12`, `"max_completion_tokens":30`, `"reasoning_effort":"low"`,
		`"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]`, `"tool_choice":"required"`, `"tool_choice":{"type":"function","function":{"name":"f"}}`,
		`"claude_thinking":{"type":"enabled","budget_tokens":1024}`, `"n":3`}
	for i := 0; i < fuzzN(200); i++ {
		var sel []string
		for _, j := range rnd.Perm(len(msgs))[:1+rnd.Intn(len(msgs))] {
			sel = append(sel, msgs[j])
		}
		fields := []string{`"model":"m"`, `"messages":[` + strings.Join(sel, ",") + `]`}
		seen := map[string]bool{}
		for _, j := range rnd.Perm(len(tops)) {
			k := tops[j][:strings.Index(tops[j], ":")]
			if seen[k] || rnd.Intn(2) == 0 {
				continue
			}
			seen[k] = true
			fields = append(fields, tops[j])
		}
		var parts []string
		for _, j := range rnd.Perm(len(fields)) {
			parts = append(parts, fields[j])
		}
		checkBedrock(t, b, "{"+strings.Join(parts, ",")+"}")
	}
}
