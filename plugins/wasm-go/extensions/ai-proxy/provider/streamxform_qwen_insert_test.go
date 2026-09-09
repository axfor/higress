package provider

// Differential tests of the system message the native Qwen conversion inserts for qwenFileIds (qwen-long, the
// leading system messages folded into one) and for the setting context (in place), against
// buildQwenTextGenerationRequest and qwenProvider.insertHttpContextMessage, at chunk sizes 1, 7 and 4096.

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

var qwenInsertMapping = map[string]string{"long": "qwen-long", "m1": "qwen-plus"}

var errMissingModel = errors.New("missing model in request")

// officialQwenInsert: the buffered conversion with qwenFileIds configured (applied inside the builder for
// qwen-long) or with the context inserted afterwards.
func officialQwenInsert(in string, fileIds []string, content *string) (map[string]any, error) {
	body, err := convertDeveloperRoleToSystem([]byte(in))
	if err != nil {
		return nil, err
	}
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest(body, req); err != nil {
		return nil, err
	}
	if req.Model == "" {
		return nil, errMissingModel
	}
	req.Model = getMappedModel(req.Model, qwenInsertMapping)
	m := &qwenProvider{config: ProviderConfig{modelMapping: qwenInsertMapping, qwenFileIds: fileIds}}
	body, err = m.buildQwenTextGenerationRequest(qwenCtxStub{}, req, req.Stream)
	if err != nil {
		return nil, err
	}
	if content != nil {
		body, err = m.insertHttpContextMessage(body, *content, false)
		if err != nil {
			return nil, err
		}
	}
	return decodeMap(body)
}

func newQwenInsertStream(fileIds []string, content *string) *streamxform.Transformer {
	opts := streamxform.QwenNativeOptions{
		MapModel: func(m string) (string, error) {
			if m == "" {
				return "", errMissingModel
			}
			return getMappedModel(m, qwenInsertMapping), nil
		},
		SupportsPreserveThinking: qwenSupportsPreserveThinking,
		DeveloperToSystem:        true,
	}
	if len(fileIds) > 0 {
		ids := make([]string, 0, len(fileIds))
		for _, id := range fileIds {
			ids = append(ids, "fileid://"+id)
		}
		files := strings.Join(ids, ",")
		opts.InsertSystem, opts.MergeLeadingSystem, opts.InsertOnlyForModel = &files, true, qwenLongModelName
	} else {
		opts.InsertSystem = content
	}
	tr := streamxform.NewQwenNative(opts)
	tr.SetFieldTree(chatRequestFieldTree)
	return tr
}

func qwenInsertCases(model string) []string {
	big := strings.Repeat("q", 70000)
	return []string{
		`{"model":"` + model + `","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"` + model + `","messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}]}`,
		`{"model":"` + model + `","messages":[{"role":"system","content":"S1"},{"role":"system","content":"S2"},{"role":"user","content":"U"},{"role":"assistant","content":"A"}]}`,
		`{"model":"` + model + `","messages":[{"role":"system","content":"S1","name":"n","reasoning_content":"r"},{"role":"system","content":"S2"},{"role":"user","content":"U"}]}`,
		`{"model":"` + model + `","messages":[{"content":"S","role":"system"},{"content":"U","role":"user"}]}`,
		`{"model":"` + model + `","messages":[{"content":"` + big + `","role":"user"}]}`,
		`{"model":"` + model + `","messages":[{"role":"system","content":"` + big + `"},{"role":"user","content":"U"}]}`,
		`{"model":"` + model + `","messages":[{"role":"developer","content":"D"},{"role":"user","content":"U"}]}`,
		`{"model":"` + model + `","messages":[{"role":"system","content":[{"type":"text","text":"S1"},{"type":"text","text":"S2"}]},{"role":"user","content":"U"}]}`,
		`{"model":"` + model + `","messages":[{"role":"system","content":[{"type":"text","text":"S"},{"type":"image_url","image_url":{"url":"http://x/a.png"}}]},{"role":"user","content":[{"type":"text","text":"U"}]}]}`,
		`{"model":"` + model + `","messages":[{"role":"system","content":""},{"role":"user","content":"U"}]}`,
		`{"model":"` + model + `","messages":[{"role":"user","content":"U"},{"role":"system","content":"late"}]}`,
		`{"model":"` + model + `","messages":[{"role":"assistant","content":"A","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","content":"r","tool_call_id":"c1"}]}`,
		`{"model":"` + model + `","messages":[{"content":"no role"}]}`,
		`{"model":"` + model + `","messages":[]}`,
		`{"messages":[{"role":"user","content":"U"}],"model":"` + model + `"}`,
		`{"model":"` + model + `","messages":[{"role":"user","content":"U"}],"stream":true,"max_tokens":9}`,
	}
}

func checkQwenInsert(t *testing.T, name string, fileIds []string, content *string, in string) {
	t.Helper()
	off, err := officialQwenInsert(in, fileIds, content)
	for _, chunk := range []int{1, 7, 4096} {
		str, ok, why := runStream(newQwenInsertStream(fileIds, content), in, chunk)
		if err != nil {
			require.False(t, ok, "%s chunk=%d: buffered failed (%v) but streaming passed\n  %s", name, chunk, err, truncateIn(in))
			continue
		}
		if strings.HasPrefix(in, `{"messages":`) && fileIds != nil {
			require.False(t, ok, "%s chunk=%d: the insertion depends on the model, messages first falls back", name, chunk)
			continue
		}
		if strings.Contains(in, `"messages":[{"content":"qqqq`) && (fileIds == nil || strings.Contains(in, `"long"`)) {
			// content ahead of role is held until the role places the insertion: bounded
			require.False(t, ok, "%s chunk=%d: content before role past the hold bound falls back", name, chunk)
			require.Contains(t, why, "limit")
			continue
		}
		require.True(t, ok, "%s chunk=%d unexpected fallback: %s\n  %s", name, chunk, why, truncateIn(in))
		require.Empty(t, diffMaps(off, str), "%s chunk=%d\n  %s", name, chunk, truncateIn(in))
	}
}

func TestQwenFileIdsDifferential(t *testing.T) {
	ids := []string{"f1", "f2"}
	for _, in := range qwenInsertCases("long") {
		checkQwenInsert(t, "fileids", ids, nil, in)
	}
	for _, in := range qwenInsertCases("m1") { // not qwen-long: no insertion
		checkQwenInsert(t, "fileids-other-model", ids, nil, in)
	}
}

func TestQwenContextDifferential(t *testing.T) {
	content := ctxFileContent
	for _, in := range qwenInsertCases("m1") {
		checkQwenInsert(t, "context", nil, &content, in)
	}
}

// Every message is system: the buffered path puts the dummy and the inserted message first; streaming has
// released the ones written in place (context) and only the folded ones (qwenFileIds) come after the insertion.
func TestQwenInsertAllSystemDeviation(t *testing.T) {
	content := ctxFileContent
	in := `{"model":"m1","messages":[{"role":"system","content":"S1"},{"role":"system","content":"S2"}]}`
	off, err := officialQwenInsert(in, nil, &content)
	require.NoError(t, err)
	msgsOf := func(m map[string]any) []any { return m["input"].(map[string]any)["messages"].([]any) }
	require.Equal(t, qwenDummySystemMessageContent, msgsOf(off)[0].(map[string]any)["content"])
	str, ok, why := runStream(newQwenInsertStream(nil, &content), in, 7)
	require.True(t, ok, why)
	msgs := msgsOf(str)
	require.Len(t, msgs, 4)
	require.Equal(t, "S1", msgs[0].(map[string]any)["content"])
	require.Equal(t, qwenDummySystemMessageContent, msgs[2].(map[string]any)["content"])
	require.Equal(t, content, msgs[3].(map[string]any)["content"])

	// qwenFileIds: the folded system messages are written back after the insertion, as the buffered path has them
	in = `{"model":"long","messages":[{"role":"system","content":"S1","name":"n"},{"role":"system","content":"S2"}]}`
	off, err = officialQwenInsert(in, []string{"f1"}, nil)
	require.NoError(t, err)
	str, ok, why = runStream(newQwenInsertStream([]string{"f1"}, nil), in, 7)
	require.True(t, ok, why)
	require.Empty(t, diffMaps(off, str))
}
