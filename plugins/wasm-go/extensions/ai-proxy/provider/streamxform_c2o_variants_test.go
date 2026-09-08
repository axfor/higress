package provider

// Differential tests of the Claude protocol conversion in front of the providers whose own transform reads what
// handleRequestBody keeps from the Claude body: zhipuai (thinking pinned to disabled unless enabled) and
// openrouter (the budget becomes reasoning.max_tokens). The pipeline is the provider's chat transformer with the
// conversion stage ahead, wired as withClaudeInput wires it; the official side runs the converter, the context
// keys, the strip and the provider's TransformRequestBody.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// zhipuTransform is zhipuAiProvider.TransformRequestBody for chat without the host: the same rewrites, then the
// default model mapping (the provider's own code calls the host for the content type).
func zhipuTransform(ctx *mapCtx, body []byte) ([]byte, error) {
	if effort := gjson.GetBytes(body, "reasoning_effort").String(); effort != "" {
		body, _ = sjson.SetBytes(body, "thinking", map[string]string{"type": "enabled"})
		body, _ = sjson.DeleteBytes(body, "reasoning_effort")
	} else if typ, ok := ctx.GetContext(ctxKeyClaudeThinkingType).(string); ok && typ != "enabled" {
		body, _ = sjson.SetBytes(body, "thinking", map[string]string{"type": "disabled"})
	}
	if requestBodyHasMessageReasoningContent(body) {
		body, _ = sjson.SetBytes(body, "thinking.clear_thinking", false)
	}
	out, ok := oaiDefaultModel(body, vMapping)
	if !ok {
		return nil, errBuildPanic
	}
	return out, nil
}

// openrouterTransform is openrouterProvider.TransformRequestBody for chat without the host.
func openrouterTransform(ctx *mapCtx, body []byte) ([]byte, error) {
	rmt := gjson.GetBytes(body, "reasoning_max_tokens")
	if !rmt.Exists() || rmt.Int() == 0 {
		if typ, _ := ctx.GetContext(ctxKeyClaudeThinkingType).(string); typ == "enabled" {
			if budget, ok := ctx.GetContext(ctxKeyClaudeBudgetTokens).(int); ok && budget > 0 {
				body, _ = sjson.DeleteBytes(body, "reasoning_effort")
				body, _ = sjson.SetBytes(body, "reasoning.max_tokens", budget)
			}
		}
	} else {
		body, _ = sjson.DeleteBytes(body, "reasoning_effort")
		body, _ = sjson.SetBytes(body, "reasoning.max_tokens", rmt.Int())
		body, _ = sjson.DeleteBytes(body, "reasoning_max_tokens")
	}
	out, ok := oaiDefaultModel(body, vMapping)
	if !ok {
		return nil, errBuildPanic
	}
	return out, nil
}

// officialClaudeInputFor: handleRequestBody's sequence for a converted Claude request up to the provider transform.
func officialClaudeInputFor(t *testing.T, in string, transform func(ctx *mapCtx, body []byte) ([]byte, error)) (map[string]any, bool) {
	t.Helper()
	ctx := newMapCtx()
	thinkingType := gjson.Get(in, "thinking.type").String()
	if thinkingType == "" {
		thinkingType = "disabled"
	}
	ctx.SetContext(ctxKeyClaudeThinkingType, thinkingType)
	if thinkingType == "enabled" {
		if b := gjson.Get(in, "thinking.budget_tokens").Int(); b > 0 {
			ctx.SetContext(ctxKeyClaudeBudgetTokens, int(b))
		}
	}
	out, err := (&ClaudeToOpenAIConverter{}).ConvertClaudeRequestToOpenAIWithOptions([]byte(in), ClaudeToOpenAIConvertOptions{})
	if err != nil {
		return nil, false
	}
	out = stripClaudeInternalMessageFields(out, false)
	out, err = transform(ctx, out)
	if err != nil {
		return nil, false
	}
	m, err := decodeMap(out)
	return m, err == nil
}

func claudeInputPipeline(v streamxform.OpenAIVariant) *streamxform.Pipeline {
	first := streamxform.NewClaudeToOpenAI(streamxform.ClaudeToOpenAIOptions{})
	first.SetFieldTree(claudeRequestFieldTree)
	second := streamxform.NewOpenAI(streamxform.OpenAIOptions{
		MapModel: func(m string) string { return getMappedModel(m, vMapping) }, DetectStream: true, NormalizeUsage: true,
		DeveloperRoleSupported: false, CheckMessages: true, Variant: v,
	})
	second.Protocol().(interface{ SetClaudeThinking(func() (string, int)) }).SetClaudeThinking(streamxform.ClaudeThinkingOf(first))
	return streamxform.NewPipeline(first, second)
}

func runPipeline(p *streamxform.Pipeline, in string, chunk int) (map[string]any, bool, string) {
	var out []byte
	p.SetSink(func(b []byte) { out = append(out, b...) })
	for i := 0; i < len(in); i += chunk {
		j := i + chunk
		if j > len(in) {
			j = len(in)
		}
		p.Write([]byte(in[i:j]))
	}
	out = append(out, p.Finish()...)
	if bad, why := p.Unsupported(); bad {
		return nil, false, why
	}
	m, err := decodeMap(out)
	if err != nil {
		return nil, false, err.Error()
	}
	return m, true, ""
}

func claudeInputVariantCases() []string {
	big := strings.Repeat("v", 70000)
	return []string{
		`{"model":"m1","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m1","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":2048}}`,
		`{"model":"m1","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":20000}}`,
		`{"model":"m1","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"}}`,
		`{"model":"m1","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}`,
		`{"thinking":{"type":"enabled","budget_tokens":5000},"model":"m1","system":"S","max_tokens":10,"messages":[{"role":"user","content":"` + big + `"}]}`,
		`{"model":"m1","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"s"},{"type":"text","text":"x"}]},{"role":"user","content":"u"}],"thinking":{"type":"enabled","budget_tokens":8000}}`,
		`{"model":"m1","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":1024}}`,
	}
}

func TestClaudeInputZhipuDifferential(t *testing.T) {
	for _, in := range claudeInputVariantCases() {
		off, ok := officialClaudeInputFor(t, in, zhipuTransform)
		require.True(t, ok, in)
		for _, chunk := range []int{1, 7, 4096} {
			str, ok, why := runPipeline(claudeInputPipeline(&streamxform.ZhipuVariant{}), in, chunk)
			require.True(t, ok, "chunk=%d: %s\n  %s", chunk, why, in)
			require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", chunk, in)
			if !strings.Contains(in, `"type":"enabled"`) {
				require.Equal(t, map[string]any{"type": "disabled"}, str["thinking"], in)
			}
		}
	}
}

func TestClaudeInputOpenRouterDifferential(t *testing.T) {
	for _, in := range claudeInputVariantCases() {
		off, ok := officialClaudeInputFor(t, in, openrouterTransform)
		require.True(t, ok, in)
		for _, chunk := range []int{1, 7, 4096} {
			str, ok, why := runPipeline(claudeInputPipeline(&streamxform.OpenRouterVariant{}), in, chunk)
			require.True(t, ok, "chunk=%d: %s\n  %s", chunk, why, in)
			require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", chunk, in)
			if strings.Contains(in, `"budget_tokens":2048`) {
				b, _ := json.Marshal(str["reasoning"])
				require.Equal(t, `{"max_tokens":2048}`, string(b), in)
				_, hasEffort := str["reasoning_effort"]
				require.False(t, hasEffort, in)
			}
		}
	}
}
