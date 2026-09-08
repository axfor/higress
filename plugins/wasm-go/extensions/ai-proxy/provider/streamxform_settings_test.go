package provider

// Differential tests of the customSettings stage against ReplaceByCustomSettings: hand-written cases for the
// ordering rules, a random corpus of bodies and setting lists, and the stage in front of the Claude conversion
// over the chat corpus, at chunk sizes 1, 7 and 4096.

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func toStreamSettings(t *testing.T, cs []CustomSetting) []streamxform.Setting {
	t.Helper()
	c := &ProviderConfig{customSettings: cs}
	out, why := c.streamSettings()
	require.Empty(t, why)
	return out
}

func cs(name, value string, overwrite bool) CustomSetting {
	return CustomSetting{name: name, value: value, mode: "raw", overwrite: overwrite}
}

func checkSettings(t *testing.T, settings []CustomSetting, in string) {
	t.Helper()
	offB, err := ReplaceByCustomSettings([]byte(in), settings)
	var off map[string]any
	if err == nil {
		off, err = decodeMap(offB)
	}
	desc := func() string {
		var b strings.Builder
		for _, s := range settings {
			fmt.Fprintf(&b, " %s=%s(ow=%v)", s.name, s.value, s.overwrite)
		}
		return b.String()
	}
	for _, chunk := range []int{1, 7, 4096} {
		tr := streamxform.NewSettings(toStreamSettings(t, settings))
		str, ok, why := runStream(tr, in, chunk)
		if err != nil {
			require.False(t, ok, "chunk=%d: sjson failed (%v) but streaming passed\n  %s\n  settings:%s", chunk, err, in, desc())
			continue
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s\n  settings:%s", chunk, why, in, desc())
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s\n  settings:%s\n  official %s", chunk, in, desc(), offB)
	}
}

func TestSettingsDifferential(t *testing.T) {
	for _, c := range []struct {
		settings []CustomSetting
		bodies   []string
	}{
		{[]CustomSetting{cs("max_tokens", "100", true)}, []string{
			`{"model":"m","messages":[]}`,
			`{"max_tokens":5,"model":"m"}`,
			`{"model":"m","max_tokens":{"a":[1,2]},"x":null}`,
			`{}`,
			`{"max_tokens":null}`,
		}},
		{[]CustomSetting{cs("max_tokens", "100", false)}, []string{
			`{"model":"m"}`,
			`{"max_tokens":5,"model":"m"}`,
			`{"max_tokens":null}`,
			`{"max_tokens":"five"}`,
			`{}`,
		}},
		{[]CustomSetting{cs("parameters.max_tokens", "100", true), cs("parameters.temperature", "0.5", false)}, []string{
			`{"model":"m"}`,
			`{"parameters":{}}`,
			`{"parameters":{"temperature":1,"max_tokens":3,"top_p":0.9},"model":"m"}`,
			`{"parameters":{"temperature":null},"model":"m"}`,
			`{"parameters":"str"}`,
			`{"parameters":null}`,
			`{"parameters":7}`,
			`{"parameters":[1,2]}`,
			`{"parameters":{"nested":{"deep":true}}}`,
		}},
		{[]CustomSetting{cs("generation_config.thinking.budget", "1024", true)}, []string{
			`{"generation_config":{"thinking":{"budget":1,"other":2},"temperature":1}}`,
			`{"generation_config":{"thinking":"x"}}`,
			`{"generation_config":{"thinking":[1]}}`,
			`{"generation_config":{}}`,
			`{"a":1}`,
		}},
		// the same path twice: the last overwrite wins; an add-if-absent after an overwrite is a no-op
		{[]CustomSetting{cs("temperature", "0.1", true), cs("temperature", "0.2", true)}, []string{`{"temperature":1}`, `{}`}},
		{[]CustomSetting{cs("temperature", "0.1", true), cs("temperature", "0.2", false)}, []string{`{"temperature":1}`, `{}`}},
		{[]CustomSetting{cs("temperature", "0.1", false), cs("temperature", "0.2", true)}, []string{`{"temperature":1}`, `{}`}},
		{[]CustomSetting{cs("temperature", "0.1", false), cs("temperature", "0.2", false)}, []string{`{"temperature":1}`, `{}`}},
		// a deeper setting after an overwrite of its ancestor lands inside the ancestor's value
		{[]CustomSetting{cs("a", `{"b":1}`, true), cs("a.c", "2", true)}, []string{`{"a":{"z":9}}`, `{}`, `{"a":"s"}`}},
		{[]CustomSetting{cs("a", `{"b":1}`, true), cs("a.b", "2", false)}, []string{`{"a":{"z":9}}`, `{}`}},
		{[]CustomSetting{cs("a", `{"b":1}`, true), cs("a.c", "2", false)}, []string{`{"a":{"z":9}}`, `{}`}},
		{[]CustomSetting{cs("a", `5`, true), cs("a.c", "2", true)}, []string{`{"a":{"z":9}}`, `{}`}},
		{[]CustomSetting{cs("a", `[1]`, true), cs("a.c", "2", true)}, []string{`{"a":{"z":9}}`, `{}`}},
		// a deeper setting after an add-if-absent ancestor: inside its value when absent, on the request when present
		{[]CustomSetting{cs("a", `{"b":1}`, false), cs("a.c", "2", true)}, []string{`{"a":{"z":9}}`, `{"a":{"c":0}}`, `{}`, `{"a":null}`, `{"a":"s"}`}},
		{[]CustomSetting{cs("a", `{"b":1}`, false), cs("a.b", "2", false)}, []string{`{"a":{"z":9}}`, `{"a":{"b":0}}`, `{}`}},
		{[]CustomSetting{cs("a", `"s"`, false), cs("a.b", "2", true)}, []string{`{"a":{"z":9}}`, `{}`}},
		// an ancestor after its descendants: an overwrite replaces them, an add-if-absent never applies
		{[]CustomSetting{cs("a.c", "2", true), cs("a", `{"b":1}`, true)}, []string{`{"a":{"z":9}}`, `{}`}},
		{[]CustomSetting{cs("a.c", "2", true), cs("a", `{"b":1}`, false)}, []string{`{"a":{"z":9}}`, `{}`}},
		{[]CustomSetting{cs("a.c", "2", false), cs("a", `{"b":1}`, false)}, []string{`{"a":{"c":0}}`, `{}`}},
		// three levels with mixed modes
		{[]CustomSetting{cs("a", `{}`, false), cs("a.b", `{"x":1}`, false), cs("a.b.c", "3", true), cs("a.d", "4", false)},
			[]string{`{}`, `{"a":{}}`, `{"a":{"b":{}}}`, `{"a":{"b":{"c":0,"x":0},"d":0}}`, `{"a":{"b":"s","d":null}}`, `{"a":1}`}},
		// keys that look like path syntax on the request side pass through untouched
		{[]CustomSetting{cs("max_tokens", "1", true)}, []string{`{"a.b":1,"max_tokens":2,"c*":{"max_tokens":3}}`, `{"max_tokens":2,"messages":[{"max_tokens":9}]}`}},
		// big values around the setting
		{[]CustomSetting{cs("temperature", "0.3", true), cs("seed", "7", false)}, []string{
			`{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("x", 70000) + `"}],"temperature":1}`,
			`{"temperature":1,"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("x", 70000) + `"}]}`,
		}},
	} {
		for _, in := range c.bodies {
			checkSettings(t, c.settings, in)
		}
	}
}

// The stage reports sjson's failure and a duplicate key as unsupported, so the buffered path decides.
func TestSettingsUnsupported(t *testing.T) {
	for _, c := range []struct {
		settings []CustomSetting
		in, why  string
	}{
		{[]CustomSetting{cs("a.b", "1", true)}, `{"a":[1,2]}`, "array"},
		{[]CustomSetting{cs("a", "[1]", true), cs("a.b", "1", true)}, `{}`, "not an object"},
		{[]CustomSetting{cs("a", "1", true)}, `{"a":1,"a":2}`, "duplicate"},
	} {
		tr := streamxform.NewSettings(toStreamSettings(t, c.settings))
		_, ok, why := runStream(tr, c.in, 7)
		require.False(t, ok, c.in)
		require.Contains(t, why, c.why, c.in)
	}
	// paths the stage does not take
	for _, name := range []string{"a.1", "a.-1", `a\.b`, "a.*", "a..b", "a.#", "@x", "a?"} {
		c := &ProviderConfig{customSettings: []CustomSetting{cs(name, "1", true)}}
		_, why := c.streamSettings()
		require.NotEmpty(t, why, name)
	}
}

// genSettingsCase: a random body over a small key alphabet with nesting, and a random list of settings on
// paths over the same alphabet, so that present, absent, scalar and nested cases all occur.
func genSettingsCase(r *rand.Rand) (string, []CustomSetting) {
	keys := []string{"a", "b", "c", "max_tokens", "temperature", "parameters", "generation_config"}
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	var value func(depth int) string
	value = func(depth int) string {
		switch n := r.Intn(9); {
		case n == 0:
			return "null"
		case n == 1:
			return pick("true", "false")
		case n == 2:
			return fmt.Sprint(r.Intn(1000))
		case n == 3:
			return `"` + pick("s", "x.y", "", "é\\n") + `"`
		case n == 4 && depth < 3:
			var els []string
			for i := r.Intn(3); i > 0; i-- {
				els = append(els, value(depth+1))
			}
			return "[" + strings.Join(els, ",") + "]"
		case depth < 3:
			var kvs []string
			seen := map[string]bool{}
			for i := r.Intn(4); i > 0; i-- {
				k := keys[r.Intn(len(keys))]
				if seen[k] {
					continue
				}
				seen[k] = true
				kvs = append(kvs, `"`+k+`":`+value(depth+1))
			}
			return "{" + strings.Join(kvs, ",") + "}"
		default:
			return fmt.Sprint(r.Intn(10))
		}
	}
	body := value(0)
	if body[0] != '{' {
		body = `{"a":` + body + `}`
	}
	var settings []CustomSetting
	for i := r.Intn(4) + 1; i > 0; i-- {
		var parts []string
		for j := r.Intn(3) + 1; j > 0; j-- {
			parts = append(parts, keys[r.Intn(len(keys))])
		}
		settings = append(settings, cs(strings.Join(parts, "."), value(2), r.Intn(2) == 0))
	}
	return body, settings
}

func TestSettingsFuzz(t *testing.T) {
	r := rand.New(rand.NewSource(54))
	for i := 0; i < 500; i++ {
		body, settings := genSettingsCase(r)
		checkSettings(t, settings, body)
	}
}

// The stage in front of the Claude conversion, as the plan builds it: the settings change what the converter
// reads (max_tokens, temperature, a stream override), compared with ReplaceByCustomSettings then the buffered builder.
func TestSettingsBeforeClaude(t *testing.T) {
	settings := []CustomSetting{cs("max_tokens", "123", false), cs("temperature", "0.3", true), cs("stream", "true", true), cs("metadata.user_id", `"u1"`, false)}
	ss := toStreamSettings(t, settings)
	for _, c := range diffCases {
		if !json.Valid([]byte(c.in)) {
			continue // sjson patches over some invalid JSON (a truncated literal on a path it sets); the stage rejects it as the decoder would
		}
		offB, err := ReplaceByCustomSettings([]byte(c.in), settings)
		require.NoError(t, err, c.name)
		off, offOK := officialClaude(string(offB), false)
		for _, chunk := range []int{1, 7, 4096} {
			tr := streamxform.NewClaude(streamxform.ClaudeOptions{MapModel: func(m string) (string, error) { return m, nil }})
			tr.SetFieldTree(chatRequestFieldTree)
			p := streamxform.NewPipeline(streamxform.NewSettings(ss), tr)
			var out []byte
			p.SetSink(func(b []byte) { out = append(out, b...) })
			for i := 0; i < len(c.in); i += chunk {
				j := i + chunk
				if j > len(c.in) {
					j = len(c.in)
				}
				p.Write([]byte(c.in[i:j]))
			}
			out = append(out, p.Finish()...)
			bad, why := p.Unsupported()
			if !offOK {
				require.True(t, bad, "%s chunk=%d: buffered failed but the pipeline passed", c.name, chunk)
				continue
			}
			require.False(t, bad, "%s chunk=%d: %s", c.name, chunk, why)
			str, err := decodeMap(out)
			require.NoError(t, err, c.name)
			require.Empty(t, diffMaps(off, str), "%s chunk=%d", c.name, chunk)
			pre := p.Protocol().(streamxform.Preluder).Prelude()
			require.True(t, pre.Stream && pre.StreamSeen, "%s chunk=%d: the stream override reaches the prelude", c.name, chunk)
		}
	}
}

// responseJsonSchema (openai / longcat): the configured schema replaces the request's response_format. The buffered
// TransformRequestBody sets it on the decoded struct and re-serialises the request, so both outputs are compared
// after that round trip: the streaming output differs from it only by what the struct would have dropped.
func TestResponseJsonSchemaDifferential(t *testing.T) {
	schema := map[string]interface{}{"type": "json_schema", "json_schema": map[string]interface{}{"name": "answer", "schema": map[string]interface{}{"type": "object", "properties": map[string]interface{}{"a": map[string]interface{}{"type": "string"}}}}}
	cfg := ProviderConfig{typ: providerTypeOpenAI, responseJsonSchema: schema, modelMapping: oaiMapping}
	official := func(in string) (map[string]any, bool) {
		req := &chatCompletionRequest{}
		if err := decodeChatCompletionRequest([]byte(in), req); err != nil {
			return nil, false
		}
		req.ResponseFormat = schema
		b, _ := json.Marshal(req)
		return officialOpenAI(string(b), oaiMapping, true)
	}
	roundTrip := func(m map[string]any) map[string]any {
		b, _ := json.Marshal(m)
		req := &chatCompletionRequest{}
		require.NoError(t, json.Unmarshal(b, req))
		b, _ = json.Marshal(req)
		out, err := decodeMap(b)
		require.NoError(t, err)
		return out
	}
	rf := cfg.streamResponseFormat()
	require.NotNil(t, rf)
	for _, in := range []string{
		`{"model":"m1","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m1","response_format":{"type":"text"},"messages":[{"role":"user","content":"U"}]}`,
		`{"response_format":{"type":"json_object"},"model":"m1","stream":true,"messages":[{"role":"user","content":"U"}],"max_tokens":9}`,
		`{"model":"m1","messages":[{"role":"user","content":"U"}],"response_format":null}`,
		`{"model":"m1","messages":[{"role":"user","content":"` + strings.Repeat("r", 70000) + `"}],"response_format":{"type":"text"},"temperature":0.5}`,
		`{"model":"m1","messages":[{"role":"system","content":"D"},{"role":"user","content":"U"}],"stream":true,"stream_options":{"include_usage":true}}`,
		`{"model":"m1","messages":[{"role":"user","content":"U"}],"response_format":"bad"}`,
		`{"model":"m1","messages":[]}`,
		`{"model":"m1","messages":null}`,
		`{"model":"m1"}`,
		`{"model":"m1","messages":"str"}`,
	} {
		off, offOK := official(in)
		for _, chunk := range []int{1, 7, 4096} {
			tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{
				MapModel: func(m string) string { return getMappedModel(m, oaiMapping) }, DetectStream: true, NormalizeUsage: true,
				DeveloperRoleSupported: isDeveloperRoleSupported(providerTypeOpenAI), CheckMessages: true, ResponseFormat: rf,
			})
			tr.SetFieldTree(chatRequestFieldTree)
			str, ok, why := runStream(tr, in, chunk)
			if !offOK {
				require.False(t, ok, "chunk=%d: buffered failed but streaming passed\n  %s", chunk, in)
				continue
			}
			require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", chunk, why, in)
			require.Equal(t, schema["type"], str["response_format"].(map[string]any)["type"], in)
			require.Empty(t, diffMaps(off, roundTrip(str)), "chunk=%d\n  %s", chunk, in)
		}
	}
}

// The one visible effect of the buffered round trip: stream_options.include_usage false is dropped by omitempty and
// then re-added as true by normalizeOpenAiRequestBody. Streaming keeps the request's false. Pinned as a deviation.
func TestResponseJsonSchemaIncludeUsageDeviation(t *testing.T) {
	schema := map[string]interface{}{"type": "json_object"}
	cfg := ProviderConfig{typ: providerTypeLongcat, responseJsonSchema: schema, modelMapping: oaiMapping}
	in := `{"model":"m1","messages":[{"role":"user","content":"U"}],"stream":true,"stream_options":{"include_usage":false}}`
	req := &chatCompletionRequest{}
	require.NoError(t, decodeChatCompletionRequest([]byte(in), req))
	req.ResponseFormat = schema
	b, _ := json.Marshal(req)
	off, ok := officialOpenAI(string(b), oaiMapping, true)
	require.True(t, ok)
	require.Equal(t, map[string]any{"include_usage": true}, off["stream_options"])
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{
		MapModel: func(m string) string { return getMappedModel(m, oaiMapping) }, DetectStream: true, NormalizeUsage: true,
		DeveloperRoleSupported: isDeveloperRoleSupported(providerTypeLongcat), CheckMessages: true, ResponseFormat: cfg.streamResponseFormat(),
	})
	str, ok, why := runStream(tr, in, 7)
	require.True(t, ok, why)
	require.Equal(t, map[string]any{"include_usage": false}, str["stream_options"])
	require.Equal(t, "json_object", str["response_format"].(map[string]any)["type"])
}
