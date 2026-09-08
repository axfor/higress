package provider

// Differential tests of the three "collect the messages' text" conversions -- MiniMax Pro, Dify, Triton -- against
// the buffered path's pure parts (struct decode, model mapping, the request builder, marshal).

import (
	"encoding/json"
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

var collectMapping = map[string]string{"m": "mapped-model", "empty": ""}

// ---- MiniMax Pro ----

func officialMiniMaxPro(in string) (map[string]any, error) {
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest([]byte(in), req); err != nil {
		return nil, err
	}
	req.Model = getMappedModel(req.Model, collectMapping)
	mp := &minimaxProvider{config: ProviderConfig{modelMapping: collectMapping}}
	b, err := json.Marshal(mp.buildMinimaxChatCompletionProRequest(req, ""))
	if err != nil {
		return nil, err
	}
	return decodeMap(b)
}

func minimaxProStream() *streamxform.Transformer {
	return typed(streamxform.NewMiniMaxPro(streamxform.MiniMaxProOptions{
		MapModel:                 func(m string) string { return getMappedModel(m, collectMapping) },
		DefaultBotName:           defaultBotName,
		DefaultSenderName:        defaultSenderName,
		DefaultBotSettingContent: defaultBotSettingContent,
		SenderTypeBot:            senderTypeBot,
		SenderTypeUser:           senderTypeUser,
	}))
}

func checkMiniMaxPro(t *testing.T, in string) {
	t.Helper()
	off, err := officialMiniMaxPro(in)
	for _, cs := range []int{1, 7, 4096} {
		str, ok, why := runStream(minimaxProStream(), in, cs)
		if err != nil {
			require.False(t, ok, "chunk=%d: buffered failed (%v) but streaming passed\n  %s", cs, err, in)
			continue
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", cs, in)
	}
}

func TestMiniMaxProDifferential(t *testing.T) {
	big := strings.Repeat("x", 70000)
	for _, in := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"system","name":"Bot","content":"S"},{"role":"user","name":"Ann","content":"hi"},{"role":"assistant","content":"a"}],"stream":true,"max_tokens":50,"temperature":0.3,"top_p":0.9}`,
		`{"model":"m","messages":[{"role":"system","content":"S1"},{"role":"system","name":"B2","content":"S2"},{"role":"user","content":"` + big + `"}]}`,
		`{"model":"m","messages":[{"role":"tool","content":"dropped"},{"role":"developer","content":"dropped"},{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}`,
		`{"model":"m","messages":[{"role":"system","content":"only system"}]}`,
		`{"model":"m","messages":[{"role":"user","content":null},{"role":"assistant"}],"max_tokens":0,"temperature":0}`,
		`{"messages":[{"role":"user","content":"no model"}]}`,
		`{"model":"empty","messages":[{"role":"user","content":"x"}]}`,
		`{"model":"m","messages":[]}`,
		`{"model":"m"}`,
		`{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":"bad"}`,
	} {
		checkMiniMaxPro(t, in)
	}
}

func TestMiniMaxProFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	msgs := []string{`{"role":"system","content":"S"}`, `{"role":"system","name":"N","content":"S2"}`, `{"role":"user","content":"u"}`,
		`{"role":"user","name":"U","content":[{"type":"text","text":"t"}]}`, `{"role":"assistant","content":"a"}`, `{"role":"tool","content":"x"}`, `{"content":"c","role":"user"}`}
	for i := 0; i < fuzzN(200); i++ {
		var sel []string
		for _, j := range rnd.Perm(len(msgs))[:rnd.Intn(len(msgs)+1)] {
			sel = append(sel, msgs[j])
		}
		fields := []string{`"model":"m"`, `"messages":[` + strings.Join(sel, ",") + `]`, `"stream":true`, `"max_tokens":9`, `"temperature":0.5`, `"top_p":0.7`, `"n":2`}
		var parts []string
		for _, j := range rnd.Perm(len(fields)) {
			if j < 2 || rnd.Intn(2) == 0 {
				parts = append(parts, fields[j])
			}
		}
		checkMiniMaxPro(t, "{"+strings.Join(parts, ",")+"}")
	}
}

// ---- Dify ----

func officialDify(in, botType, inputVar, conv string) (map[string]any, error) {
	body, err := convertDeveloperRoleToSystem([]byte(in))
	if err != nil {
		return nil, err
	}
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest(body, req); err != nil {
		return nil, err
	}
	if req.Model == "" {
		return nil, errors.New("missing model in request")
	}
	if getMappedModel(req.Model, collectMapping) == "" {
		return nil, errors.New("model becomes empty after applying the configured mapping")
	}
	content := ""
	for _, message := range req.Messages {
		if message.Role == "system" {
			content += "SYSTEM: \n" + message.StringContent() + "\n"
		} else if message.Role == "assistant" {
			content += "ASSISTANT: \n" + message.StringContent() + "\n"
		} else {
			content += "USER: \n" + message.StringContent() + "\n"
		}
	}
	mode := "blocking"
	if req.Stream {
		mode = "streaming"
	}
	user := req.User
	if user == "" {
		user = "api-user"
	}
	var r *DifyChatRequest
	switch botType {
	case BotTypeChat, BotTypeAgent:
		r = &DifyChatRequest{Inputs: map[string]interface{}{}, Query: content, ResponseMode: mode, User: user, ConversationId: conv}
	case BotTypeCompletion:
		r = &DifyChatRequest{Inputs: map[string]interface{}{"query": content}, ResponseMode: mode, User: user}
	case BotTypeWorkflow:
		r = &DifyChatRequest{Inputs: map[string]interface{}{inputVar: content}, ResponseMode: mode, User: user}
	default:
		r = &DifyChatRequest{}
	}
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	return decodeMap(b)
}

func difyStream(botType, inputVar, conv string) *streamxform.Transformer {
	c := &ProviderConfig{modelMapping: collectMapping}
	return typed(streamxform.NewDify(streamxform.DifyOptions{MapModel: c.mapStrict(), BotType: botType, InputVariable: inputVar, ConversationId: conv}))
}

func checkDify(t *testing.T, in string) {
	t.Helper()
	for _, bt := range []string{BotTypeChat, BotTypeAgent, BotTypeCompletion, BotTypeWorkflow, "Other"} {
		off, err := officialDify(in, bt, "text_in", "conv-1")
		for _, cs := range []int{1, 7, 4096} {
			str, ok, why := runStream(difyStream(bt, "text_in", "conv-1"), in, cs)
			if err != nil {
				require.False(t, ok, "%s chunk=%d: buffered failed (%v) but streaming passed\n  %s", bt, cs, err, in)
				continue
			}
			require.True(t, ok, "%s chunk=%d unexpected fallback: %s\n  %s", bt, cs, why, in)
			require.Empty(t, diffMaps(off, str), "%s chunk=%d\n  %s", bt, cs, in)
		}
	}
}

func TestDifyDifferential(t *testing.T) {
	big := strings.Repeat("y", 70000)
	for _, in := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"system","content":"S"},{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"tool","content":"t"}],"stream":true,"user":"bob"}`,
		`{"messages":[{"role":"developer","content":"D"},{"role":"user","content":"` + big + `"},{"role":"user","content":"tail"}],"model":"m"}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"a<>&\"\n"},{"type":"image_url","image_url":{"url":"u"}}]},{"role":"user","content":null},{"role":"user"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"x"}],"stream":false,"user":""}`,
		`{"messages":[{"role":"user","content":"x"}]}`,
		`{"model":"empty","messages":[{"role":"user","content":"x"}]}`,
		`{"model":"m","messages":[]}`,
		`{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":"bad"}`,
	} {
		checkDify(t, in)
	}
}

// ---- Triton ----

func officialTriton(in string) (map[string]any, string, error) {
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest([]byte(in), req); err != nil {
		return nil, "", err
	}
	if req.Model == "" {
		return nil, "", errors.New("missing model in request")
	}
	req.Model = getMappedModel(req.Model, collectMapping)
	if req.Model == "" {
		return nil, "", errors.New("model becomes empty after applying the configured mapping")
	}
	tp := &tritonProvider{config: ProviderConfig{tritonModelVersion: "2"}}
	b, err := json.Marshal(tp.BuildTritonTexGenRequest(req))
	if err != nil {
		return nil, "", err
	}
	m, err := decodeMap(b)
	if err != nil {
		return nil, "", err
	}
	path := strings.Replace(strings.Replace(tritonChatGenerationWithVersionPath, "{MODEL_VERSION}", "2", 1), "{MODEL_NAME}", req.Model, 1)
	if req.Stream {
		path += "_stream"
	}
	return m, path, nil
}

func tritonStream() *streamxform.Transformer {
	c := &ProviderConfig{modelMapping: collectMapping}
	return typed(streamxform.NewTriton(streamxform.TritonOptions{MapModel: c.mapStrict()}))
}

func checkTriton(t *testing.T, in string) {
	t.Helper()
	off, wantPath, err := officialTriton(in)
	for _, cs := range []int{1, 7, 4096} {
		tr := tritonStream()
		str, ok, why := runStream(tr, in, cs)
		if err != nil {
			require.False(t, ok, "chunk=%d: buffered failed (%v) but streaming passed\n  %s", cs, err, in)
			continue
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", cs, in)
		pre := tr.Protocol().(streamxform.Preluder).Prelude()
		path := strings.Replace(strings.Replace(tritonChatGenerationWithVersionPath, "{MODEL_VERSION}", "2", 1), "{MODEL_NAME}", getMappedModel(pre.Model, collectMapping), 1)
		if pre.Stream {
			path += "_stream"
		}
		require.Equal(t, wantPath, path, "chunk=%d path\n  %s", cs, in)
	}
}

func TestTritonDifferential(t *testing.T) {
	big := strings.Repeat("z", 70000)
	for _, in := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"id":"a","role":"user","content":"first"},{"id":"b","role":"assistant","content":"` + big + `"},{"role":"user","content":"last","id":"c"}],"stream":true,"temperature":0.4}`,
		`{"model":"m","messages":[{"role":"user","content":"x"},{"role":"user"}],"temperature":0}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"p"},{"type":"text","text":"q"}]}],"stream":false}`,
		`{"messages":[{"role":"user","content":"x"}]}`,
		`{"model":"m","messages":[]}`,
		`{"model":"m","messages":[{"role":"user","content":"x"}],"stream":"yes"}`,
	} {
		checkTriton(t, in)
	}
}

func TestTritonFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	msgs := []string{`{"id":"1","role":"user","content":"u"}`, `{"role":"assistant","content":"a","id":"2"}`, `{"role":"user","content":null}`, `{"role":"user","content":[{"type":"text","text":"t"}]}`, `{"content":"c","role":"user"}`}
	for i := 0; i < fuzzN(200); i++ {
		var sel []string
		for _, j := range rnd.Perm(len(msgs))[:1+rnd.Intn(len(msgs))] {
			sel = append(sel, msgs[j])
		}
		fields := []string{`"model":"m"`, `"messages":[` + strings.Join(sel, ",") + `]`, `"stream":true`, `"temperature":0.2`, `"max_tokens":5`}
		var parts []string
		for _, j := range rnd.Perm(len(fields)) {
			if j < 2 || rnd.Intn(2) == 0 {
				parts = append(parts, fields[j])
			}
		}
		checkTriton(t, "{"+strings.Join(parts, ",")+"}")
	}
}
