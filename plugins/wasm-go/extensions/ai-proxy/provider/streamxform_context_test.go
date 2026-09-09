package provider

// Differential tests of the setting `context` on the streaming path: the OpenAI-shaped insertion against
// defaultInsertHttpContextMessage (after the default transform, as handleRequestBody orders it), and the Claude
// conversion's system prefix against claudeProvider.insertHttpContextMessage, at chunk sizes 1, 7 and 4096.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

const ctxFileContent = "Context from the file.\nSecond line \"quoted\" \\ back"

// officialContextOpenAI: defaultTransformRequestBody then defaultInsertHttpContextMessage, both of which the buffered
// path runs on an OpenAI-shaped body; the insertion re-serialises through chatCompletionRequest.
func officialContextOpenAI(in string) (map[string]any, bool) {
	m, ok := officialOpenAI(in, oaiMapping, true)
	if !ok {
		return nil, false
	}
	b, _ := json.Marshal(m)
	out, err := defaultInsertHttpContextMessage(b, ctxFileContent)
	if err != nil {
		return nil, false
	}
	res, err := decodeMap(out)
	return res, err == nil
}

// roundTripChat normalises a streaming output the way the buffered insertion incidentally does (a struct round trip).
func roundTripChat(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	b, _ := json.Marshal(m)
	req := &chatCompletionRequest{}
	require.NoError(t, json.Unmarshal(b, req))
	b, _ = json.Marshal(req)
	out, err := decodeMap(b)
	require.NoError(t, err)
	return out
}

// newContextOpenAI is the plan's transformer: the protocol with the insertion, the struct round trip behind it.
func newContextOpenAI() streamxform.Xform {
	content := ctxFileContent
	cfg := ProviderConfig{typ: providerTypeOpenAI}
	return cfg.openAIShape(streamxform.OpenAIOptions{
		MapModel: func(m string) string { return getMappedModel(m, oaiMapping) }, DetectStream: true, NormalizeUsage: true,
		DeveloperRoleSupported: isDeveloperRoleSupported(providerTypeOpenAI), CheckMessages: true, InsertSystem: &content,
	})
}

func contextCases() []string {
	big := strings.Repeat("c", 70000)
	return []string{
		`{"model":"m1","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m1","messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}]}`,
		`{"model":"m1","messages":[{"role":"system","content":"S1"},{"role":"system","content":"S2"},{"role":"user","content":"U"},{"role":"assistant","content":"A"}]}`,
		`{"model":"m1","messages":[{"content":"U","role":"user"}]}`,
		`{"model":"m1","messages":[{"content":"S","name":"n","role":"system"},{"content":"U","role":"user","name":"x"}]}`,
		`{"model":"m1","messages":[{"role":"user","content":"` + big + `"},{"role":"assistant","content":"A"}]}`,
		`{"model":"m1","messages":[{"role":"system","content":"` + big + `"},{"role":"user","content":"U"}]}`,
		`{"model":"m1","messages":[{"content":"` + big + `","role":"user"}]}`,
		`{"messages":[{"role":"user","content":"U"}],"model":"m1","stream":true}`,
		`{"model":"m1","messages":[{"role":"user","content":[{"type":"text","text":"U"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`,
		`{"model":"m1","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"r"}]}`,
		`{"model":"m1","messages":[]}`,
		`{"model":"m1","messages":null}`,
		`{"model":"m1"}`,
		`{"model":"m1","messages":[{"content":"no role"}]}`,
		`{"model":"m1","messages":[{"role":5,"content":"U"}]}`,
		`{"model":"m1","messages":[{"role":null,"content":"U"}]}`,
		`{"model":"m1","messages":"str"}`,
		`{"model":"m1","messages":[5]}`,
		`{"model":"m1","messages":[{"role":"user","content":"U"},{"role":"user","content":"U2"}],"temperature":0.5,"max_tokens":9}`,
	}
}

func TestContextOpenAIDifferential(t *testing.T) {
	big := strings.Repeat("c", 70000)
	for _, in := range contextCases() {
		off, offOK := officialContextOpenAI(in)
		for _, chunk := range []int{1, 7, 4096} {
			str, ok, why := runXform(newContextOpenAI(), in, chunk)
			if strings.HasPrefix(in, `{"model":"m1","messages":[{"content":"`+big) {
				// content ahead of role is held until the role decides where the context message goes: bounded
				require.False(t, ok, "chunk=%d: content before role past the hold bound falls back", chunk)
				require.Contains(t, why, "limit")
				continue
			}
			if !offOK {
				require.False(t, ok, "chunk=%d: buffered failed but streaming passed\n  %s", chunk, in)
				continue
			}
			require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", chunk, why, in)
			require.Empty(t, diffMaps(off, str), "chunk=%d (the round-trip stage makes it identical)\n  %s", chunk, in)
			// the inserted message is exactly the buffered one
			msgs := str["messages"].([]any)
			found := 0
			for _, m := range msgs {
				mm := m.(map[string]any)
				if mm["role"] == "system" && mm["content"] == ctxFileContent {
					found++
				}
			}
			require.Equal(t, 1, found, "chunk=%d\n  %s", chunk, in)
		}
	}
}

// Every message is system: the buffered path prepends the context message, streaming appends it (the messages
// have already gone out). Pinned as the documented deviation.
func TestContextOpenAIAllSystemDeviation(t *testing.T) {
	in := `{"model":"m1","messages":[{"role":"system","content":"S1"},{"role":"system","content":"S2"}]}`
	off, ok := officialContextOpenAI(in)
	require.True(t, ok)
	require.Equal(t, ctxFileContent, off["messages"].([]any)[0].(map[string]any)["content"])
	str, ok, why := runXform(newContextOpenAI(), in, 7)
	require.True(t, ok, why)
	msgs := str["messages"].([]any)
	require.Len(t, msgs, 3)
	require.Equal(t, "S1", msgs[0].(map[string]any)["content"])
	require.Equal(t, ctxFileContent, msgs[2].(map[string]any)["content"])
}

// officialContextClaude: buildClaudeTextGenRequest then claudeProvider.insertHttpContextMessage. The latter calls
// String() through a nil System when the request has no system message, and panics: reported as crashed.
func officialContextClaude(in string, claudeCode bool) (m map[string]any, ok, crashed bool) {
	defer func() {
		if r := recover(); r != nil {
			m, ok, crashed = nil, false, true
		}
	}()
	m, err := officialClaudeErr(in, claudeCode)
	if err != nil {
		return nil, false, false
	}
	b, _ := json.Marshal(m)
	c := &claudeProvider{config: ProviderConfig{claudeCodeMode: claudeCode}}
	out, err := c.insertHttpContextMessage(b, ctxFileContent, false)
	if err != nil {
		return nil, false, false
	}
	res, err := decodeMap(out)
	return res, err == nil, false
}

func TestContextClaudeDifferential(t *testing.T) {
	content := ctxFileContent
	for _, claudeCode := range []bool{false, true} {
		for _, c := range diffCases {
			if !json.Valid([]byte(c.in)) {
				continue
			}
			off, offOK, crashed := officialContextClaude(c.in, claudeCode)
			for _, chunk := range []int{1, 7, 4096} {
				tr := streamxform.NewClaude(streamxform.ClaudeOptions{
					MapModel: func(m string) (string, error) { return m, nil }, ClaudeCodeMode: claudeCode, ContextPrefix: &content,
				})
				tr.SetFieldTree(chatRequestFieldTree)
				str, ok, why := runStream(tr, c.in, chunk)
				if crashed {
					// no system message: the buffered inserter panics; streaming makes the content the system prompt
					require.True(t, ok, "%s claudeCode=%v chunk=%d: %s", c.name, claudeCode, chunk, why)
					require.Equal(t, ctxFileContent, str["system"], c.name)
					continue
				}
				if !offOK {
					require.False(t, ok, "%s claudeCode=%v chunk=%d: buffered failed but streaming passed", c.name, claudeCode, chunk)
					continue
				}
				require.True(t, ok, "%s claudeCode=%v chunk=%d unexpected fallback: %s", c.name, claudeCode, chunk, why)
				require.Empty(t, diffMaps(off, str), "%s claudeCode=%v chunk=%d", c.name, claudeCode, chunk)
				require.True(t, strings.HasPrefix(str["system"].(string), ctxFileContent), c.name)
			}
		}
	}
}

// roundTripClaude normalises a streaming output the way the buffered claude inserter incidentally does (a round trip
// through claudeTextGenRequest: an absent tool_use input is written as null by the builder and dropped by the trip).
func roundTripClaude(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	b, _ := json.Marshal(m)
	req := &claudeTextGenRequest{}
	require.NoError(t, json.Unmarshal(b, req))
	b, _ = json.Marshal(req)
	out, err := decodeMap(b)
	require.NoError(t, err)
	return out
}
