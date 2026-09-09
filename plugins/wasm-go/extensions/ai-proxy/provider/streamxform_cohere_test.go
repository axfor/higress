package provider

// Differential tests of the OpenAI → Cohere chat conversion: the buffered TransformRequestBody's pure parts
// (struct decode, strict model mapping, buildCohereRequest, marshal) against the streaming transformer.

import (
	"encoding/json"
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

var errCohereNull = errors.New("buffered path sends the literal null")

func officialCohere(in string, mapping map[string]string) (map[string]any, error) {
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest([]byte(in), req); err != nil {
		return nil, err
	}
	if req.Model == "" {
		return nil, errors.New("missing model in request")
	}
	req.Model = getMappedModel(req.Model, mapping)
	if req.Model == "" {
		return nil, errors.New("model becomes empty after applying the configured mapping")
	}
	cr := (&cohereProvider{}).buildCohereRequest(req)
	if cr == nil {
		return nil, errCohereNull
	}
	b, err := json.Marshal(cr)
	if err != nil {
		return nil, err
	}
	return decodeMap(b)
}

var cohereMapping = map[string]string{"m": "command-r-plus", "gpt-*": "command-r", "empty": ""}

func cohereStream() *streamxform.Transformer {
	c := &ProviderConfig{modelMapping: cohereMapping}
	return typed(streamxform.NewCohere(streamxform.CohereOptions{MapModel: c.mapStrict()}))
}

func cohereCases() []string {
	big := strings.Repeat("c", 70000)
	return []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"gpt-4o","messages":[{"role":"system","content":"S"},{"role":"user","content":"ignored"}],"stream":true}`,
		`{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u"}},{"text":"b","type":"text"}]}],"model":"m"}`,
		`{"model":"m","messages":[{"role":"user","content":"` + big + `"}],"max_tokens":100,"temperature":0.7,"n":2,"top_p":0.9,"seed":42,"stop":["a","b"],"frequency_penalty":0.5,"presence_penalty":-0.25}`,
		`{"model":"m","messages":[{"role":"user","content":"z"}],"max_tokens":0,"temperature":0,"n":0,"top_p":0,"seed":0,"stop":[],"frequency_penalty":0,"presence_penalty":0}`,
		`{"model":"m","messages":[{"role":"user","content":null}]}`,
		`{"model":"m","messages":[{"role":"user"}]}`,
		`{"model":"m","messages":[{"content":"c<>&\"\\\n","role":"user","name":"n","tool_calls":null}],"tools":[{"type":"function","function":{"name":"f"}}],"user":"u","logit_bias":{"1":2}}`,
		`{"model":"m","messages":[]}`,
		`{"model":"m"}`,
		`{"messages":[{"role":"user","content":"x"}]}`,
		`{"model":"empty","messages":[{"role":"user","content":"x"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"1e0"}],"temperature":1e0,"max_tokens":1e2}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":1},{"type":"text"},"str",{"type":"other","text":"no"}]}]}`,
	}
}

func checkCohere(t *testing.T, in string) {
	t.Helper()
	off, err := officialCohere(in, cohereMapping)
	for _, cs := range []int{1, 7, 4096} {
		str, ok, why := runStream(cohereStream(), in, cs)
		if err != nil {
			require.False(t, ok, "chunk=%d: buffered failed (%v) but streaming passed\n  %s", cs, err, in)
			continue
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", cs, in)
	}
}

func TestCohereDifferential(t *testing.T) {
	for _, in := range cohereCases() {
		checkCohere(t, in)
	}
}

func TestCohereFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	fields := []string{
		`"model":"m"`, `"stream":true`, `"max_tokens":50`, `"temperature":0.3`, `"n":3`, `"top_p":0.8`, `"seed":7`,
		`"stop":["s"]`, `"frequency_penalty":1.5`, `"presence_penalty":0.1`, `"user":"u"`, `"tools":[]`,
		`"messages":[{"role":"user","content":"first"},{"role":"assistant","content":"second"}]`,
		`"messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("t", 2000) + `"},{"type":"text","text":"u"}]}]`,
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
		checkCohere(t, "{"+strings.Join(parts, ",")+"}")
	}
}
