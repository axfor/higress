package provider

// Differential tests of the streaming Claude → OpenAI conversion against the buffered one followed by the
// stripping of the Claude-internal fields, which is what an OpenAI-compatible provider receives.

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/sjson"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func officialC2O(in string, preserve, disableUsage bool) (map[string]any, error) {
	out, err := (&ClaudeToOpenAIConverter{}).ConvertClaudeRequestToOpenAIWithOptions([]byte(in), ClaudeToOpenAIConvertOptions{
		PreserveMessageReasoningContent: preserve, DisableStreamUsageStats: disableUsage,
	})
	if err != nil {
		return nil, err
	}
	out = stripClaudeInternalMessageFields(out, preserve)
	for _, f := range []string{"claude_thinking", "claude_output_config", "claude_anthropic_beta"} {
		out, _ = sjson.DeleteBytes(out, f)
	}
	return decodeMap(out)
}

func c2oStream(preserve, disableUsage bool) *streamxform.Transformer {
	tr := streamxform.NewClaudeToOpenAI(streamxform.ClaudeToOpenAIOptions{PreserveReasoning: preserve, DisableStreamUsageStats: disableUsage})
	tr.SetFieldTree(claudeRequestFieldTree)
	return tr
}

func checkC2O(t *testing.T, in string) {
	t.Helper()
	for _, preserve := range []bool{false, true} {
		for _, disable := range []bool{false, true} {
			off, err := officialC2O(in, preserve, disable)
			for _, cs := range []int{1, 7, 4096} {
				str, ok, why := runStream(c2oStream(preserve, disable), in, cs)
				if err != nil {
					require.False(t, ok, "chunk=%d preserve=%v: buffered failed (%v) but streaming passed\n  %s", cs, preserve, err, in)
					continue
				}
				require.True(t, ok, "chunk=%d preserve=%v disable=%v unexpected fallback: %s\n  %s", cs, preserve, disable, why, in)
				require.Empty(t, diffMaps(off, str), "chunk=%d preserve=%v disable=%v\n  %s", cs, preserve, disable, in)
			}
		}
	}
}

func c2oCases() []string {
	big := strings.Repeat("c", 70000)
	return []string{
		`{"model":"claude-3","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"claude-3","max_tokens":10,"stream":true,"temperature":0.5,"top_p":0.9,"stop_sequences":["a","b"],"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}`,
		`{"model":"claude-3","messages":[{"role":"user","content":[{"type":"text","text":"look","cache_control":{"type":"ephemeral"}},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},{"type":"image","source":{"type":"url","url":"https://x/y.png"}},{"type":"image","source":{"type":"weird"}},{"type":"text","text":""}]}]}`,
		`{"model":"claude-3","messages":[{"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"t1","name":"f","input":{"a":1}},{"type":"tool_use","id":"t2","name":"g"},{"type":"tool_use","name":"noid"}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"r1"},{"type":"tool_result","tool_use_id":"t2","content":[{"type":"text","text":"r2a"},{"type":"text","text":"r2b"}]},{"type":"text","text":"and more"}]}]}`,
		`{"model":"claude-3","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"s"},{"type":"text","text":"answer"}]},{"role":"assistant","content":[{"type":"redacted_thinking","data":"xx"}]},{"role":"assistant","content":[{"type":"thinking","thinking":"only"}]}]}`,
		`{"model":"claude-3","system":"x-anthropic-billing-header: cc_version=1; cc_entrypoint=cli; cch=abc123; more=1","messages":[{"role":"user","content":"x-anthropic-billing-header: a; cch=zzz"}]}`,
		`{"model":"claude-3","system":[{"type":"text","text":"S1","cache_control":{"type":"ephemeral"}},{"type":"text","text":"S2"}],"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"` + big + `"}],"system":"after messages","max_tokens":5}`,
		`{"max_tokens":5,"messages":[{"role":"user","content":"u"},{"role":"assistant","content":"a"}],"model":"claude-3","stream":true,"system":"late"}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}},{"name":"g","input_schema":{}}],"tool_choice":{"type":"tool","name":"f"}}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f"}],"tool_choice":{"type":"any","disable_parallel_tool_use":true}}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"auto"}}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"none"},"tools":[]}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":1024}}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":8000}}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":20000},"output_config":{"effort":"high"},"anthropic_beta":["x"]}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}`,
		`{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"temperature":0,"top_p":0,"max_tokens":0,"stop_sequences":[],"metadata":{"user_id":"u"},"top_k":3,"anthropic_version":"v","service_tier":"auto"}`,
		`{"model":"claude-3"}`,
		`{"model":"claude-3","system":"only system"}`,
		`{"model":"claude-3","messages":[]}`,
		`{"model":"claude-3","messages":[{"role":"user","content":null},{"role":"user"}]}`,
		`{"model":"claude-3","messages":[{"role":"user","content":42}]}`,
		`{"messages":[{"role":"user","content":"no model"}]}`,
		`{"model":"claude-3","max_tokens":"x","messages":[{"role":"user","content":"hi"}]}`,
	}
}

func TestClaudeToOpenAIDifferential(t *testing.T) {
	for _, in := range c2oCases() {
		checkC2O(t, in)
	}
}

func TestClaudeToOpenAIFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	msgs := []string{
		`{"role":"user","content":"u"}`, `{"role":"assistant","content":"a"}`,
		`{"role":"user","content":[{"type":"text","text":"t"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}]}`,
		`{"role":"assistant","content":[{"type":"text","text":"c"},{"type":"tool_use","id":"1","name":"f","input":{"k":2}}]}`,
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"r"},{"type":"text","text":"more"}]}`,
		`{"role":"assistant","content":[{"type":"thinking","thinking":"th","signature":"sg"},{"type":"text","text":"x"}]}`,
		`{"role":"assistant","content":[{"type":"redacted_thinking","data":"d"}]}`,
		`{"content":"first","role":"user"}`, `{"role":"user","content":[]}`,
	}
	tops := []string{`"stream":true`, `"temperature":0.4`, `"top_p":0.5`, `"max_tokens":12`, `"stop_sequences":["s"]`, `"system":"S"`,
		`"system":[{"type":"text","text":"S1"}]`, `"tools":[{"name":"f","input_schema":{"type":"object"}}]`, `"tool_choice":{"type":"any"}`,
		`"tool_choice":{"type":"tool","name":"f","disable_parallel_tool_use":true}`, `"thinking":{"type":"enabled","budget_tokens":5000}`, `"metadata":{"user_id":"u"}`}
	for i := 0; i < fuzzN(200); i++ {
		var sel []string
		for _, j := range rnd.Perm(len(msgs))[:rnd.Intn(len(msgs)+1)] {
			sel = append(sel, msgs[j])
		}
		fields := []string{`"model":"claude-3"`, `"messages":[` + strings.Join(sel, ",") + `]`}
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
		checkC2O(t, "{"+strings.Join(parts, ",")+"}")
	}
}
