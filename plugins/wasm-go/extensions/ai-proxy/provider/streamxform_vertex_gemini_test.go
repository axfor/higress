package provider

// Differential tests of the OpenAI → Vertex (Gemini shape) conversion against buildVertexChatRequest: the Claude
// corpus plus Vertex-specific shapes (tool calls in either order, tool messages, system messages, images, thinking,
// response_format), body and path, at chunk sizes 1, 7 and 4096.

import (
	"encoding/json"
	"errors"
	"math/rand"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func newGeminiVertexProvider() *vertexProvider {
	cfg := ProviderConfig{
		vertexProjectId:     "test-proj",
		vertexRegion:        "us-central1",
		modelMapping:        map[string]string{"m": "gemini-2.5-pro", "old": "gemini-2.0-flash", "empty": ""},
		geminiSafetySetting: map[string]string{"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH"},
	}
	return &vertexProvider{config: cfg}
}

func officialVertexGemini(v *vertexProvider, in string) (map[string]any, string, error) {
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest([]byte(in), req); err != nil {
		return nil, "", err
	}
	if req.Model == "" {
		return nil, "", errors.New("missing model in request")
	}
	req.Model = getMappedModel(req.Model, v.config.modelMapping)
	if req.Model == "" {
		return nil, "", errors.New("model becomes empty after applying the configured mapping")
	}
	vr, err := v.buildVertexChatRequest(req)
	if err != nil {
		return nil, "", err
	}
	b, err := json.Marshal(vr)
	if err != nil {
		return nil, "", err
	}
	m, err := decodeMap(b)
	if err != nil {
		return nil, "", err
	}
	// safetySettings comes from a map on the buffered path: compare it as a set
	if ss, ok := m["safetySettings"].([]any); ok {
		sort.Slice(ss, func(i, j int) bool {
			return ss[i].(map[string]any)["category"].(string) < ss[j].(map[string]any)["category"].(string)
		})
	}
	return m, v.getRequestPath(newMapCtx(), ApiNameChatCompletion, req.Model, req.Stream), nil
}

func vertexGeminiStream(v *vertexProvider) *streamxform.Transformer {
	var ss []streamxform.GeminiSafetySetting
	for k, val := range v.config.geminiSafetySetting {
		ss = append(ss, streamxform.GeminiSafetySetting{Category: k, Threshold: val})
	}
	sort.Slice(ss, func(i, j int) bool { return ss[i].Category < ss[j].Category })
	return typed(streamxform.NewVertexGemini(streamxform.VertexGeminiOptions{
		MapModel:       v.config.mapStrict(),
		SafetySettings: ss,
		ApplyResponseFormat: func(rf map[string]any, mapped string) (string, map[string]any, error) {
			var cfg vertexChatGenerationConfig
			if err := v.applyResponseFormatToGenerationConfig(rf, &cfg, mapped); err != nil {
				return "", nil, err
			}
			return cfg.ResponseMimeType, cfg.ResponseSchema, nil
		},
		DetectMime: detectMimeTypeFromURL,
	}))
}

func checkVertexGemini(t *testing.T, v *vertexProvider, in string) {
	t.Helper()
	off, wantPath, err := officialVertexGemini(v, in)
	for _, cs := range []int{1, 7, 4096} {
		tr := vertexGeminiStream(v)
		str, ok, why := runStream(tr, in, cs)
		if err != nil {
			require.False(t, ok, "chunk=%d: buffered failed (%v) but streaming passed\n  %s", cs, err, in)
			continue
		}
		if !ok && strings.Contains(why, "tool_calls after the content of a non-assistant message") {
			continue // by design: only assistant content waits for tool_calls
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", cs, in)
		pre := tr.Protocol().(streamxform.Preluder).Prelude()
		mapped := getMappedModel(pre.Model, v.config.modelMapping)
		require.Equal(t, wantPath, v.getRequestPath(newMapCtx(), ApiNameChatCompletion, mapped, pre.Stream), "chunk=%d path\n  %s", cs, in)
	}
}

func vertexGeminiCases() []string {
	big := strings.Repeat("v", 70000)
	return []string{
		`{"model":"m","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}],"stream":true}`,
		`{"model":"m","messages":[{"role":"user","content":"A"},{"role":"assistant","content":"B"},{"role":"system","content":"S"}]}`,
		`{"model":"m","messages":[{"role":"developer","content":"D"},{"role":"user","content":""},{"role":"user","content":null},{"role":"user"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"` + big + `"}],"temperature":0.7,"top_p":0.9,"max_tokens":100,"stream":false}`,
		`{"model":"m","messages":[{"role":"assistant","content":"thinking","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},{"role":"tool","tool_call_id":"c1","content":"result"}]}`,
		`{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"not json"},"thought_signature":"sig"}],"content":null},{"role":"tool","content":""},{"role":"tool","content":[{"type":"text","text":"t1"},{"type":"text","text":""}]}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"kept","tool_calls":[]},{"role":"assistant","content":"kept2","tool_calls":null}]}`,
		`{"model":"m","messages":[{"tool_calls":[{"function":{"name":"g","arguments":"{}"},"extra_content":{"google":{"thought_signature":"es"}}}],"role":"assistant","content":"x"},{"role":"tool","content":"r"}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"https://x/y.png?sz=1"}},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AAAA"}},{"type":"image_url","image_url":{"url":"data:bad"}},{"type":"image_url","image_url":{"url":"ftp://x"}},{"type":"input_audio","input_audio":{"data":"d","format":"wav"}},"str",{"image_url":{"url":"data:image/png;base64,BB"},"type":"image_url"}]}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"` + big + `"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("Q", 60000) + `"}}]}]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"low"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"medium"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"high"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"none"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"reasoning_effort":"weird"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"response_format":{"type":"json_object"}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object","properties":{"a":{"type":"string"}}}}}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"response_format":{"type":"json_schema","json_schema":{"name":"n"}}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"response_format":{"type":"text"}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"response_format":{"type":"object","properties":{}}}`,
		`{"model":"old","messages":[{"role":"user","content":"U"}],"response_format":{"type":"json_object"}}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}},{"type":"function","function":{"name":"g"}}]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"tools":null}`,
		`{"model":"m","messages":[]}`,
		`{"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"empty","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"max_tokens":"x"}`,
		`{"model":"m","messages":[{"role":"user","content":"U"}],"stop":["s"],"n":2,"seed":1,"user":"u","logit_bias":{"1":1}}`,
	}
}

func TestVertexGeminiDifferential(t *testing.T) {
	v := newGeminiVertexProvider()
	for _, in := range vertexGeminiCases() {
		checkVertexGemini(t, v, in)
	}
	for _, c := range diffCases {
		checkVertexGemini(t, v, c.in)
	}
}

func TestVertexGeminiFuzz(t *testing.T) {
	v := newGeminiVertexProvider()
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	msgs := []string{
		`{"role":"system","content":"S"}`, `{"role":"user","content":"u"}`, `{"role":"assistant","content":"a"}`,
		`{"role":"assistant","content":"c","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{\"k\":2}"}}]}`,
		`{"tool_calls":[{"function":{"name":"h","arguments":"{}"}}],"role":"assistant"}`,
		`{"role":"tool","content":"r","tool_call_id":"1"}`, `{"role":"tool","content":[{"type":"text","text":"tr"}]}`,
		`{"role":"user","content":[{"type":"text","text":"t"},{"type":"image_url","image_url":{"url":"https://h/p.jpg"}},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]}`,
		`{"content":"first","role":"user"}`, `{"role":"user","content":null}`, `{"role":"developer","content":"d"}`,
	}
	tops := []string{`"stream":true`, `"temperature":0.4`, `"top_p":0.5`, `"max_tokens":12`, `"reasoning_effort":"low"`,
		`"response_format":{"type":"json_object"}`, `"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]`, `"n":3`}
	for i := 0; i < fuzzN(200); i++ {
		var sel []string
		for _, j := range rnd.Perm(len(msgs))[:1+rnd.Intn(len(msgs))] {
			sel = append(sel, msgs[j])
		}
		fields := []string{`"model":"m"`, `"messages":[` + strings.Join(sel, ",") + `]`}
		for _, j := range rnd.Perm(len(tops)) {
			if rnd.Intn(2) == 0 {
				fields = append(fields, tops[j])
			}
		}
		var parts []string
		for _, j := range rnd.Perm(len(fields)) {
			parts = append(parts, fields[j])
		}
		checkVertexGemini(t, v, "{"+strings.Join(parts, ",")+"}")
	}
}
