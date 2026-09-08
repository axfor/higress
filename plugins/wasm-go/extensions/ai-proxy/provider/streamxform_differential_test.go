package provider

// Differential tests of streaming request body transformation (streamxform): the same inputs go to the buffered implementation and the streaming one, compared field by field.
//
// The buffered side calls the pure functions of this package directly where possible (buildClaudeTextGenRequest, convertDeveloperRoleToSystem,
// getMappedModel ...); branches that depend on the host (request headers in defaultTransformRequestBody, the Claude context) are already
// declared non-streaming by the integration layer, so their pure-function parts are reproduced here line by line from the buffered code.
// Decoding uses UseNumber so number literals take part in the comparison (1e0 differs from 1); the buffered path reformats the number fields it reads.
//
// The random fuzz runs at a small scale by default to suit CI; STREAMXFORM_FUZZ_N scales it up.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func fuzzN(def int) int {
	if v := os.Getenv("STREAMXFORM_FUZZ_N"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// ---- source: difftest_test.go ----
// officialClaude reproduces the full pure-function path of the buffered claude provider in handleRequestBody:
// developer→system conversion → decodeChatCompletionRequest → mapModel → buildClaudeTextGenRequest.
func officialClaude(in string, claudeCode bool) (map[string]any, bool) {
	m, err := officialClaudeErr(in, claudeCode)
	return m, err == nil
}

func officialClaudeErr(in string, claudeCode bool) (map[string]any, error) {
	body, err := convertDeveloperRoleToSystem([]byte(in))
	if err != nil {
		return nil, err
	}
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest(body, req); err != nil {
		return nil, err
	}
	if req.Model == "" {
		return nil, fmt.Errorf("missing model in request") // mapModel
	}
	c := &claudeProvider{config: ProviderConfig{claudeCodeMode: claudeCode}}
	cr := c.buildClaudeTextGenRequest(req)
	b, err := json.Marshal(cr)
	if err != nil {
		return nil, err
	}
	m, err := decodeMap(b)
	return m, err
}

func officialTransform(t *testing.T, in string) (map[string]any, bool) {
	return officialClaude(in, false)
}

// typed wires the root-field type check exactly as the plugin does, so the differential and fuzz corpora run
// through it too. It goes through checkChatRequestTypes rather than calling SetFieldTypes here, so the test
// cannot drift from the decision production makes about which protocols get the check.
// fuzzSeed 允许固定随机种子（ASON_FUZZ_SEED），用于两次跑之间做受控对比；
// 缺省仍按时间取，保证每次跑覆盖不同语料。
func fuzzSeed() int64 {
	if v := os.Getenv("ASON_FUZZ_SEED"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return time.Now().UnixNano()
}

func typed(tr *streamxform.Transformer) *streamxform.Transformer {
	return checkChatRequestTypes(&StreamPlan{Tr: tr}).Tr
}

func runStream(tr *streamxform.Transformer, in string, chunk int) (map[string]any, bool, string) {
	var out []byte
	for i := 0; i < len(in); i += chunk {
		j := i + chunk
		if j > len(in) {
			j = len(in)
		}
		tr.Write([]byte(in[i:j]))
		out = append(out, tr.Out()...)
	}
	out = append(out, tr.Finish()...)
	if bad, why := tr.Unsupported(); bad {
		return nil, false, why
	}
	m, err := decodeMap(out)
	if err != nil {
		return nil, false, "output is not valid JSON: " + err.Error() + " :: " + string(out)
	}
	return m, true, ""
}

func streamTransform(in string, chunk int) (map[string]any, bool, string) {
	return runStream(streamxform.New(), in, chunk)
}

var diffCases = []struct{ name, in string }{
	{"minimal", `{"model":"m","messages":[{"role":"user","content":"U"}]}`},
	{"with system", `{"model":"m","messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}]}`},
	{"multi-turn", `{"model":"m","messages":[{"role":"user","content":"A"},{"role":"assistant","content":"B"},{"role":"user","content":"C"}]}`},
	{"all scalars", `{"model":"m","max_tokens":100,"stream":true,"temperature":0.7,"top_p":0.9,"stop":["END","X"],"messages":[{"role":"user","content":"U"}]}`},
	{"max_completion wins", `{"model":"m","max_tokens":10,"max_completion_tokens":200,"messages":[{"role":"user","content":"U"}]}`},
	{"no Claude counterpart", `{"model":"m","seed":42,"n":1,"user":"u","presence_penalty":0.5,"messages":[{"role":"user","content":"U"}]}`},
	{"shuffled", `{"messages":[{"content":"U","role":"user"}],"stop":["E"],"model":"m","max_tokens":50}`},
	{"no max_tokens", `{"model":"m","messages":[{"role":"user","content":"U"}]}`},
	{"escaped content", `{"model":"m","messages":[{"role":"user","content":"a\"b\\c\nd<>&"}]}`},
	{"zero scalars", `{"model":"m","temperature":0,"top_p":0.0,"max_tokens":0,"stream":false,"messages":[{"role":"user","content":"U"}]}`},
	{"empty system", `{"model":"m","messages":[{"role":"system","content":""},{"role":"user","content":"U"}]}`},
	{"two systems", `{"model":"m","messages":[{"role":"system","content":"S1"},{"role":"user","content":"U"},{"role":"system","content":"S2"}]}`},
	{"system array", `{"model":"m","messages":[{"role":"system","content":[{"type":"text","text":"S1"},{"type":"text","text":"S2"}]},{"role":"user","content":"U"}]}`},
	{"all system", `{"model":"m","messages":[{"role":"system","content":"S"}]}`},
	{"developer", `{"model":"m","messages":[{"role":"developer","content":"D"},{"role":"user","content":"U"}]}`},
	{"role missing", `{"model":"m","messages":[{"content":"U"}]}`},
	{"empty object message", `{"model":"m","messages":[{}]}`},
	{"content null", `{"model":"m","messages":[{"role":"assistant","content":null}]}`},
	{"content missing", `{"model":"m","messages":[{"role":"user"}]}`},
	{"content number", `{"model":"m","messages":[{"role":"user","content":42}]}`},
	{"content object", `{"model":"m","messages":[{"role":"user","content":{"a":1}}]}`},
	{"multimodal text", `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"U1"},{"type":"text","text":"U2"}]}]}`},
	{"multimodal empty array", `{"model":"m","messages":[{"role":"user","content":[]}]}`},
	{"multimodal late type", `{"model":"m","messages":[{"role":"user","content":[{"text":"U1","type":"text"}]}]}`},
	{"multimodal unknown type", `{"model":"m","messages":[{"role":"user","content":[{"type":"video","text":"x"},{"type":"text","text":"U"}]}]}`},
	{"multimodal missing type", `{"model":"m","messages":[{"role":"user","content":[{"text":"x"},{"type":"text","text":"U"}]}]}`},
	{"multimodal non-object element", `{"model":"m","messages":[{"role":"user","content":["str",1,{"type":"text","text":"U"}]}]}`},
	{"image data", `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo=","detail":"high"}}]}]}`},
	{"image data escaped", `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image\/jpeg;base64,\/9j\/4AAQ"}}]}]}`},
	{"image data not base64", `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:text/plain;charset=utf8,hello"}}]}]}`},
	{"image data empty", `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}]}]}`},
	{"image url", `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://x/y.png"}}]}]}`},
	{"image url empty", `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":""}}]}]}`},
	{"image late type", `{"model":"m","messages":[{"role":"user","content":[{"image_url":{"url":"https://x/y.png"},"type":"image_url"}]}]}`},
	{"file", `{"model":"m","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"f1","file_name":"a.pdf"}}]}]}`},
	{"file empty id", `{"model":"m","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":""}}]}]}`},
	{"audio skipped", `{"model":"m","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA","format":"wav"}},{"type":"text","text":"U"}]}]}`},
	{"cache_control", `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"U","cache_control":{"type":"ephemeral"}}]}]}`},
	{"tools", `{"model":"m","tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"a":{"type":"string"}}}}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools without parameters", `{"model":"m","tools":[{"type":"function","function":{"name":"f","parameters":{}}},{"type":"function","function":{"name":"g"}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools empty", `{"model":"m","tools":[],"messages":[{"role":"user","content":"U"}]}`},
	{"tools null", `{"model":"m","tools":null,"messages":[{"role":"user","content":"U"}]}`},
	{"tool_choice auto", `{"model":"m","tool_choice":"auto","messages":[{"role":"user","content":"U"}]}`},
	{"tool_choice none", `{"model":"m","tool_choice":"none","messages":[{"role":"user","content":"U"}]}`},
	{"tool_choice required", `{"model":"m","tool_choice":"required","parallel_tool_calls":false,"messages":[{"role":"user","content":"U"}]}`},
	{"tool_choice function", `{"model":"m","tool_choice":{"type":"function","function":{"name":"f"}},"messages":[{"role":"user","content":"U"}]}`},
	{"tool_choice function without name", `{"model":"m","tool_choice":{"type":"function","function":{}},"messages":[{"role":"user","content":"U"}]}`},
	{"tool_choice null", `{"model":"m","tool_choice":null,"messages":[{"role":"user","content":"U"}]}`},
	{"tool_choice+thinking", `{"model":"m","tool_choice":"required","reasoning_effort":"low","messages":[{"role":"user","content":"U"}]}`},
	{"tool_calls", `{"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}]}`},
	{"tool_calls empty content", `{"model":"m","messages":[{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":""}}]}]}`},
	{"tool_calls null content", `{"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"bad json"}}]}]}`},
	{"tool_calls before content", `{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}],"content":"x"}]}`},
	{"tool_calls empty array", `{"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":[]}]}`},
	{"several tool_calls", `{"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"c2","type":"function","function":{"name":"g","arguments":"null"}}]}]}`},
	{"tool_calls on user", `{"model":"m","messages":[{"role":"user","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`},
	{"tool role", `{"model":"m","messages":[{"role":"tool","tool_call_id":"c1","content":"r1"}]}`},
	{"tool role merged", `{"model":"m","messages":[{"role":"user","content":"U"},{"role":"tool","tool_call_id":"c1","content":"r1"},{"role":"tool","tool_call_id":"c2","content":"r2"},{"role":"assistant","content":"A"}]}`},
	{"tool role across system", `{"model":"m","messages":[{"role":"tool","tool_call_id":"c1","content":"r1"},{"role":"system","content":"S"},{"role":"tool","tool_call_id":"c2","content":"r2"}]}`},
	{"tool role array", `{"model":"m","messages":[{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}`},
	{"tool role without id", `{"model":"m","messages":[{"role":"tool","content":"r1"}]}`},
	{"tool role late", `{"model":"m","messages":[{"content":"r1","tool_call_id":"c1","role":"tool"}]}`},
	{"tool role last", `{"model":"m","messages":[{"role":"user","content":"U"},{"role":"tool","tool_call_id":"c1","content":"r1"}]}`},
	{"reasoning_effort", `{"model":"m","reasoning_effort":"high","max_tokens":20000,"messages":[{"role":"user","content":"U"}]}`},
	{"reasoning below max", `{"model":"m","reasoning_effort":"high","max_tokens":2000,"messages":[{"role":"user","content":"U"}]}`},
	{"reasoning too small", `{"model":"m","reasoning_effort":"low","max_tokens":500,"messages":[{"role":"user","content":"U"}]}`},
	{"reasoning_max_tokens", `{"model":"m","reasoning_max_tokens":3000,"messages":[{"role":"user","content":"U"}]}`},
	{"claude_thinking", `{"model":"m","claude_thinking":{"type":"enabled","budget_tokens":2048},"reasoning_effort":"high","messages":[{"role":"user","content":"U"}]}`},
	{"claude_thinking disabled", `{"model":"m","claude_thinking":{"type":"disabled"},"tool_choice":"required","messages":[{"role":"user","content":"U"}]}`},
	{"claude_output_config", `{"model":"m","claude_output_config":{"effort":"low","format":{"type":"json"}},"messages":[{"role":"user","content":"U"}]}`},
	{"claude_anthropic_beta", `{"model":"m","claude_anthropic_beta":["x"],"messages":[{"role":"user","content":"U"}]}`},
	{"tools reversed order", `{"model":"m","tools":[{"function":{"parameters":{"b":1,"a":{"c":[true,null]}},"description":"d","name":"f"},"type":"function"}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools name missing", `{"model":"m","tools":[{"type":"function","function":{"description":"d"}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools function missing", `{"model":"m","tools":[{"type":"function"}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools function null", `{"model":"m","tools":[{"type":"function","function":null}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools empty object element", `{"model":"m","tools":[{}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools empty description", `{"model":"m","tools":[{"type":"function","function":{"name":"f","description":"","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools extra fields", `{"model":"m","tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"},"x":[1]},"extra":{"a":1}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools parameters null", `{"model":"m","tools":[{"type":"function","function":{"name":"f","parameters":null}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools parameters with whitespace", "{\"model\":\"m\",\"tools\":[ {\n \"type\": \"function\",\n \"function\": { \"name\" : \"f\" , \"parameters\" : { \"type\" : \"object\" , \"properties\" : { } } }\n} ],\"messages\":[{\"role\":\"user\",\"content\":\"U\"}]}"},
	{"tools non-object element", `{"model":"m","tools":[5],"messages":[{"role":"user","content":"U"}]}`},
	{"tools not an array", `{"model":"m","tools":{"a":1},"messages":[{"role":"user","content":"U"}]}`},
	{"tools number", `{"model":"m","tools":5,"messages":[{"role":"user","content":"U"}]}`},
	{"tools function number", `{"model":"m","tools":[{"type":"function","function":5}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools parameters array", `{"model":"m","tools":[{"type":"function","function":{"name":"f","parameters":[]}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools parameters number", `{"model":"m","tools":[{"type":"function","function":{"name":"f","parameters":1}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools name number", `{"model":"m","tools":[{"type":"function","function":{"name":1}}],"messages":[{"role":"user","content":"U"}]}`},
	{"tools after messages", `{"model":"m","messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"f"}}],"stop":["a","b"]}`},
	{"several stop", `{"model":"m","stop":["a","\u0041","€"],"messages":[{"role":"user","content":"U"}]}`},
	{"stop string", `{"model":"m","stop":"a","messages":[{"role":"user","content":"U"}]}`},
	{"stop with number", `{"model":"m","stop":["a",1],"messages":[{"role":"user","content":"U"}]}`},
	{"stop object", `{"model":"m","stop":{"a":1},"messages":[{"role":"user","content":"U"}]}`},
	{"literal nul", `{"model":"m","stop":nul,"messages":[{"role":"user","content":"U"}]}`},
	{"literal tru", `{"model":"m","stream":tru,"messages":[{"role":"user","content":"U"}]}`},
	{"literal leading zero", `{"model":"m","max_tokens":01,"messages":[{"role":"user","content":"U"}]}`},
	{"literal 1.", `{"model":"m","temperature":1.,"messages":[{"role":"user","content":"U"}]}`},
	{"literal 1e", `{"model":"m","temperature":1e,"messages":[{"role":"user","content":"U"}]}`},
	{"literal in a dropped field", `{"model":"m","metadata":{"a":-},"messages":[{"role":"user","content":"U"}]}`},
	{"literal in a passthrough region", `{"model":"m","messages":[{"role":"user","content":"U","zzz":[+1]}]}`},
	{"literal valid edge cases", `{"model":"m","temperature":-0,"top_p":0.5E+1,"max_tokens":100,"zzz":{"a":-0.0e-0,"b":[1E2]},"messages":[{"role":"user","content":"U"}]}`},
	{"tools parameters number literals", `{"model":"m","tools":[{"type":"function","function":{"name":"f","parameters":{"enum":[1e0,2.50,-0.5e-3,100000000000000000000000,9007199254740993]}}}],"messages":[{"role":"user","content":"U"}]}`},
	{"string invalid escape", `{"model":"m","messages":[{"role":"user","content":"a\x"}]}`},
	{"string short u escape", `{"model":"m","messages":[{"role":"user","content":"a\u12G4"}]}`},
	{"string control character", "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"a\x01b\"}]}"},
	{"string invalid key escape", `{"model":"m","messages":[{"role":"user","content":"U","z\q":1}]}`},
	{"whitespace non-JSON", "{\"model\":\"m\",\x0b\"messages\":[{\"role\":\"user\",\"content\":\"U\"}]}"},
	{"stop empty", `{"model":"m","stop":[],"messages":[{"role":"user","content":"U"}]}`},
	{"stop null", `{"model":"m","stop":null,"messages":[{"role":"user","content":"U"}]}`},
	{"temperature null", `{"model":"m","temperature":null,"messages":[{"role":"user","content":"U"}]}`},
	{"temperature integer", `{"model":"m","temperature":1,"top_p":1e0,"messages":[{"role":"user","content":"U"}]}`},
	{"negative max_tokens", `{"model":"m","max_tokens":-5,"messages":[{"role":"user","content":"U"}]}`},
	{"message unknown fields", `{"model":"m","messages":[{"role":"user","content":"U","name":"n","reasoning_content":"r","refusal":"x","zzz":{"a":[1,2]}}]}`},
	{"top-level nested unknown", `{"model":"m","logit_bias":{"1":2},"metadata":{"k":"v"},"response_format":{"type":"json_schema","json_schema":{"schema":{"content":1}}},"messages":[{"role":"user","content":"U"}]}`},
	{"whitespace formatted", "{\n  \"model\" : \"m\" ,\n  \"messages\" : [ { \"role\" : \"user\" , \"content\" : \"U\" } ]\n}\n"},
	{"unicode escape", `{"model":"mA","messages":[{"role":"user","content":"€😀"}]}`},
}

func TestDifferential(t *testing.T) {
	fmt.Println("  case                     buffered vs streaming")
	fmt.Println("  ----------------------   ------------------------------------")
	for _, c := range diffCases {
		for _, cc := range []bool{false, true} {
			off, offOK := officialClaude(c.in, cc)
			for _, cs := range []int{1, 7, 4096} {
				str, ok, why := runStream(typed(streamxform.NewClaude(streamxform.ClaudeOptions{ClaudeCodeMode: cc})), c.in, cs)
				if !offOK {
					if ok {
						t.Errorf("%s(cc=%v): buffered path failed but streaming passed", c.name, cc)
					} else if cs == 4096 && !cc {
						fmt.Printf("  %-22s   buffered failed, streaming fell back ✓ (%s)\n", c.name, why)
					}
					continue
				}
				if !ok {
					t.Errorf("%s(cc=%v) chunk=%d: buffered path succeeded but streaming fell back: %s", c.name, cc, cs, why)
					continue
				}
				if d := diffMaps(off, str); d != "" {
					t.Errorf("%s(cc=%v) chunk=%d differs from the buffered path:\n    %s", c.name, cc, cs, d)
				} else if cs == 4096 && !cc {
					fmt.Printf("  %-22s   ✓ identical\n", c.name)
				}
			}
		}
	}
}

// canonNumbers normalizes the json.Number values of a subtree to the encoding/json float64 format.
//
// The buffered path decodes tools[].function.parameters through map[string]interface{} and re-encodes it: keys sorted, numbers
// reformatted as float64 (1e0→1, 2.50→2.5, big integers lose precision). Streaming passes that subtree through verbatim, keeping the client's
// literals with the same meaning and no precision loss. The comparison treats that subtree by numeric value; everywhere else literals must match exactly.
func canonNumbers(v any) any {
	switch x := v.(type) {
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return x
		}
		return f
	case map[string]any:
		for k, e := range x {
			x[k] = canonNumbers(e)
		}
	case []any:
		for i, e := range x {
			x[i] = canonNumbers(e)
		}
	}
	return v
}

func diffMaps(a, b map[string]any) string {
	// Claude: tools[].input_schema; Gemini: tools[].function_declarations[].parameters; DashScope: parameters.tools[].function.parameters
	canonNumbers(a["tools"])
	canonNumbers(b["tools"])
	if pa, ok := a["parameters"].(map[string]any); ok {
		canonNumbers(pa["tools"])
	}
	if pb, ok := b["parameters"].(map[string]any); ok {
		canonNumbers(pb["tools"])
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) == string(jb) {
		return ""
	}
	// note: every number in a map[string]any is a float64, so 1e0 and 1 cannot be told apart here;
	// the gateway differential compares raw bytes and once caught this kind of "same value, different literal" difference (now formatted like the buffered path).
	keys := map[string]bool{}
	for k := range a {
		keys[k] = true
	}
	for k := range b {
		keys[k] = true
	}
	out := ""
	for k := range keys {
		va, _ := json.Marshal(a[k])
		vb, _ := json.Marshal(b[k])
		if string(va) != string(vb) {
			out += fmt.Sprintf("[%s] buffered=%s streaming=%s  ", k, va, vb)
		}
	}
	return out
}

// decodeMap decodes with UseNumber: numbers keep their literal, so 1e0 and 1 differ. The buffered path reformats the number fields it reads
// through float64 and streaming must apply the same formatting, otherwise the bytes differ (caught by the gateway differential).
func decodeMap(b []byte) (map[string]any, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}
	return m, nil
}

// ---- source: fuzz_test.go ----
// Random OpenAI requests covering every field buildClaudeTextGenRequest reads,
// with random field order, values and message shapes.
func genRequest(r *rand.Rand) string {
	type kv struct{ k, v string }
	var fields []kv
	add := func(k, v string) { fields = append(fields, kv{k, v}) }
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	q := func(s string) string { b, _ := json.Marshal(s); return string(b) }

	if r.Intn(30) != 0 {
		add("model", q("m"+fmt.Sprint(r.Intn(5))))
	}
	if r.Intn(2) == 0 {
		add("max_tokens", pick(fmt.Sprint(r.Intn(30000)+1), "0", "-3"))
	}
	if r.Intn(3) == 0 {
		add("max_completion_tokens", pick(fmt.Sprint(r.Intn(30000)+1), "0"))
	}
	if r.Intn(2) == 0 {
		add("stream", pick("true", "false"))
	}
	if r.Intn(2) == 0 {
		add("temperature", pick(fmt.Sprintf("%.2f", r.Float64()), "0", "1", "0.0", "1e0"))
	}
	if r.Intn(2) == 0 {
		add("top_p", pick(fmt.Sprintf("%.2f", r.Float64()), "0", "null"))
	}
	if r.Intn(3) == 0 {
		n := r.Intn(3)
		var ss []string
		for i := 0; i < n; i++ {
			ss = append(ss, q(randText(r)))
		}
		stopV := "[" + strings.Join(ss, ",") + "]"
		switch r.Intn(12) {
		case 0:
			stopV = "null"
		case 1:
			stopV = "[" + strings.Join(append(ss, "1"), ",") + "]" // non-string element: buffered path fails
		case 2:
			stopV = q(randText(r)) // string: buffered path fails
		case 3:
			stopV = "[ " + strings.Join(ss, " , ") + " ]" // with whitespace
		}
		add("stop", stopV)
	}
	for _, f := range []string{"seed", "n", "presence_penalty", "frequency_penalty", "top_logprobs"} {
		if r.Intn(5) == 0 {
			add(f, fmt.Sprint(r.Intn(100)))
		}
	}
	if r.Intn(5) == 0 {
		add("logprobs", "true")
	}
	if r.Intn(5) == 0 {
		add("user", q(randText(r)))
	}
	if r.Intn(5) == 0 {
		add("metadata", `{"k":`+q(randText(r))+`,"model":"trap","messages":[{"role":"x"}]}`)
	}
	if r.Intn(5) == 0 {
		add("response_format", `{"type":"json_schema","json_schema":{"schema":{"content":{"role":1}}}}`)
	}
	if r.Intn(6) == 0 {
		add("stream_options", `{"include_usage":true}`)
	}
	if r.Intn(8) == 0 {
		add("unknown_"+fmt.Sprint(r.Intn(3)), pick(`1`, `"s"`, `{"a":[1,{"b":null}]}`, `[]`))
	}
	// reasoning / thinking
	if r.Intn(4) == 0 {
		add("reasoning_effort", q(pick("low", "medium", "high", "xyz", "")))
	}
	if r.Intn(5) == 0 {
		add("reasoning_max_tokens", pick(fmt.Sprint(r.Intn(40000)), "0"))
	}
	if r.Intn(6) == 0 {
		add("claude_thinking", pick(`{"type":"enabled","budget_tokens":`+fmt.Sprint(r.Intn(20000))+`}`, `{"type":"disabled"}`, `{"type":""}`, `null`, `{"type":"enabled","display":"summarized"}`))
	}
	if r.Intn(8) == 0 {
		add("claude_output_config", pick(`{"effort":"low"}`, `{"format":{"type":"json_schema","schema":{"a":1}}}`, `{"effort":"","format":null}`, `{}`))
	}
	if r.Intn(8) == 0 {
		add("claude_anthropic_beta", `["b1"]`)
	}
	// tools: random field order; missing / empty / wrong type / extra fields all covered
	if r.Intn(3) == 0 {
		n := r.Intn(3)
		var ts []string
		for i := 0; i < n; i++ {
			var fn []string
			if r.Intn(8) != 0 {
				fn = append(fn, `"name":`+q("f"+fmt.Sprint(i)))
			}
			switch r.Intn(4) {
			case 0:
				fn = append(fn, `"description":`+q(randText(r)))
			case 1:
				fn = append(fn, `"description":""`) // omitempty
			}
			switch r.Intn(9) {
			case 0:
				fn = append(fn, `"parameters":{}`)
			case 1:
				fn = append(fn, `"parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number","enum":[1,2.5,-0.5e-3]}},"required":["a"]}`)
			case 2:
				fn = append(fn, `"parameters":null`)
			case 3:
				fn = append(fn, "\"parameters\": {\n  \"type\" : \"object\" ,\n  \"properties\" : { \"x\" : { \"type\" : \"string\", \"description\": "+q(randText(r))+" } }\n}")
			case 4:
				fn = append(fn, `"parameters":[]`) // buffered path fails
			case 5:
				fn = append(fn, `"parameters":`+pick(`5`, `"s"`, `true`)) // buffered path fails
			}
			if r.Intn(4) == 0 {
				fn = append(fn, `"strict":true`)
			}
			r.Shuffle(len(fn), func(a, b int) { fn[a], fn[b] = fn[b], fn[a] })
			var tl []string
			switch r.Intn(10) {
			case 0: // function missing
			case 1:
				tl = append(tl, `"function":null`)
			case 2:
				tl = append(tl, `"function":`+pick(`5`, `"s"`, `[]`)) // buffered path fails
			default:
				tl = append(tl, `"function":{`+strings.Join(fn, ",")+`}`)
			}
			if r.Intn(6) != 0 {
				tl = append(tl, `"type":`+pick(`"function"`, `"function"`, `""`, `"x"`))
			}
			if r.Intn(5) == 0 {
				tl = append(tl, `"extra":{"a":[1]}`)
			}
			r.Shuffle(len(tl), func(a, b int) { tl[a], tl[b] = tl[b], tl[a] })
			el := `{` + strings.Join(tl, ",") + `}`
			if r.Intn(30) == 0 {
				el = pick(`5`, `"s"`, `[]`, `null`) // wrong element type: buffered path fails
			}
			ts = append(ts, el)
		}
		toolsV := "[" + strings.Join(ts, ",") + "]"
		switch r.Intn(15) {
		case 0:
			toolsV = "null"
		case 1:
			toolsV = pick(`{}`, `5`, `"s"`) // buffered path fails
		case 2:
			toolsV = "[ " + strings.Join(ts, " ,\n ") + " ]"
		}
		add("tools", toolsV)
	}
	if r.Intn(3) == 0 {
		add("tool_choice", pick(`"auto"`, `"none"`, `"required"`, `"function"`, `{"type":"function","function":{"name":"f1"}}`, `{"type":"function","function":{}}`, `{"type":"auto"}`, `null`, `{"type":"tool","function":{"name":"g"}}`))
	}
	if r.Intn(4) == 0 {
		add("parallel_tool_calls", pick("true", "false", "null"))
	}

	// messages
	nmsg := r.Intn(5)
	if r.Intn(20) != 0 && nmsg == 0 {
		nmsg = 1
	}
	var msgs []string
	for i := 0; i < nmsg; i++ {
		msgs = append(msgs, genMessage(r))
	}
	add("messages", pick("["+strings.Join(msgs, ",")+"]", "["+strings.Join(msgs, ",")+"]", "["+strings.Join(msgs, ",")+"]", "null"))

	r.Shuffle(len(fields), func(i, j int) { fields[i], fields[j] = fields[j], fields[i] })
	var out []string
	for _, f := range fields {
		out = append(out, `"`+f.k+`":`+f.v)
	}
	return "{" + strings.Join(out, ",") + "}"
}

func genMessage(r *rand.Rand) string {
	type kv struct{ k, v string }
	var parts []kv
	add := func(k, v string) { parts = append(parts, kv{k, v}) }
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	q := func(s string) string { b, _ := json.Marshal(s); return string(b) }

	role := pick("user", "user", "assistant", "assistant", "system", "developer", "tool", "tool", "")
	if r.Intn(12) != 0 {
		add("role", q(role))
	} else {
		role = ""
	}
	// content
	switch r.Intn(8) {
	case 0:
		// missing
	case 1:
		add("content", "null")
	case 2:
		add("content", genParts(r))
	case 3:
		add("content", pick("42", "true", `{"a":1}`))
	default:
		add("content", q(randText(r)))
	}
	if role == "assistant" && r.Intn(2) == 0 {
		n := r.Intn(3)
		var tcs []string
		for i := 0; i < n; i++ {
			args := pick(`"{\"a\":1}"`, `""`, `"not json"`, `"null"`, `"{}"`, q(`{"k":"`+randText(r)+`"}`))
			tc := `{"id":` + q("c"+fmt.Sprint(i)) + `,"type":"function","function":{"name":` + q("f"+fmt.Sprint(i)) + `,"arguments":` + args + `}}`
			if r.Intn(4) == 0 {
				tc = `{"type":"function","function":{"name":"f","arguments":"{}"}}`
			}
			tcs = append(tcs, tc)
		}
		add("tool_calls", pick("["+strings.Join(tcs, ",")+"]", "null"))
	}
	if role == "tool" && r.Intn(5) != 0 {
		add("tool_call_id", q("call_"+fmt.Sprint(r.Intn(9))))
	}
	if r.Intn(6) == 0 {
		add("name", q("n"))
	}
	if r.Intn(8) == 0 {
		add("reasoning_content", q(randText(r)))
	}
	if r.Intn(10) == 0 {
		add("zz_unknown", `{"role":"system","content":[{"type":"text"}]}`)
	}
	r.Shuffle(len(parts), func(i, j int) { parts[i], parts[j] = parts[j], parts[i] })
	var out []string
	for _, p := range parts {
		out = append(out, `"`+p.k+`":`+p.v)
	}
	return "{" + strings.Join(out, ",") + "}"
}

func genParts(r *rand.Rand) string {
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	q := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	n := r.Intn(4)
	var ps []string
	for i := 0; i < n; i++ {
		var kvs []string
		switch r.Intn(7) {
		case 0, 1:
			kvs = []string{`"type":"text"`, `"text":` + q(randText(r))}
		case 2:
			url := pick("data:image/png;base64,"+randB64(r), "data:image\\/jpeg;base64,"+randB64(r), "https://x/"+randB64(r), "data:text/plain,hi", "data:nosemicolon", "", "data:;base64,"+randB64(r), "data:image/png;charset=x;base64,"+randB64(r))
			img := `{"url":"` + url + `"`
			if r.Intn(2) == 0 {
				img += `,"detail":"high"`
			}
			img += `}`
			kvs = []string{`"type":"image_url"`, `"image_url":` + img}
		case 3:
			kvs = []string{`"type":"file"`, `"file":` + pick(`{"file_id":"f1"}`, `{"file_id":""}`, `{"file_id":"f2","file_name":"a.pdf"}`)}
		case 4:
			kvs = []string{`"type":"input_audio"`, `"input_audio":{"data":"AA","format":"wav"}`}
		case 5:
			kvs = []string{`"type":` + pick(`"video"`, `1`, `null`), `"text":"x"`}
		case 6:
			kvs = []string{`"text":"x"`} // type missing
		}
		if r.Intn(4) == 0 {
			kvs = append(kvs, `"cache_control":{"type":"ephemeral"}`)
		}
		if r.Intn(6) == 0 {
			kvs = append(kvs, `"text":1`) // wrong type
		}
		r.Shuffle(len(kvs), func(i, j int) { kvs[i], kvs[j] = kvs[j], kvs[i] })
		// duplicate keys make streaming Bail (last wins on the buffered path); deduplicated here to focus on comparing semantics
		seen := map[string]bool{}
		var uniq []string
		for _, kv := range kvs {
			k := kv[:strings.Index(kv, ":")]
			if seen[k] {
				continue
			}
			seen[k] = true
			uniq = append(uniq, kv)
		}
		ps = append(ps, "{"+strings.Join(uniq, ",")+"}")
	}
	if r.Intn(6) == 0 {
		ps = append(ps, pick(`"str"`, `1`, `null`, `[]`))
	}
	return "[" + strings.Join(ps, ",") + "]"
}

func randB64(r *rand.Rand) string {
	const a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	n := r.Intn(600)
	b := make([]byte, n)
	for i := range b {
		b[i] = a[r.Intn(len(a))]
	}
	return string(b)
}

func randText(r *rand.Rand) string {
	alph := []string{"a", "b", "€", "ß", `"`, `\`, "\n", "\t", "é", " ", "😀", "<", "&", " ", "\x01"}
	n := r.Intn(40)
	var sb strings.Builder
	for i := 0; i < n; i++ {
		sb.WriteString(alph[r.Intn(len(alph))])
	}
	return sb.String()
}

func TestFuzzDifferential(t *testing.T) {
	seed := fuzzSeed()
	r := rand.New(rand.NewSource(seed))
	N := fuzzN(20000)
	same, fallback, offFail, lenient := 0, 0, 0, 0
	for i := 0; i < N; i++ {
		in := genRequest(r)
		cc := r.Intn(4) == 0
		off, oerr := officialClaudeErr(in, cc)
		ok := oerr == nil
		chunk := []int{1, 3, 17, 64, 4096}[r.Intn(5)]
		str, sok, why := runStream(typed(streamxform.NewClaude(streamxform.ClaudeOptions{ClaudeCodeMode: cc})), in, chunk)
		if !ok {
			offFail++
			if sok {
				// the only lenience allowed: the buffered struct fails type validation of a dropped field (an array inside metadata, say).
				// Streaming does not read those fields and produces the same request the buffered path would with a valid value there.
				// A type error in a field that is read (tools / stop / messages / ...) must fail in streaming as well.
				if isDiscardedFieldTypeError(oerr) {
					lenient++
					continue
				}
				t.Fatalf("case %d: buffered path failed but streaming passed (chunk=%d): %v\n  input: %s", i, chunk, oerr, in)
			}
			continue
		}
		if !sok {
			fallback++
			if !strings.Contains(why, "duplicate key") && !strings.Contains(why, "the buffered path skips") && !strings.Contains(why, "panic") {
				t.Fatalf("case %d: unexpected fallback (chunk=%d): %s\n  input: %s", i, chunk, why, in)
			}
			continue
		}
		if d := diffMaps(off, str); d != "" {
			t.Fatalf("case %d: mismatch (chunk=%d cc=%v)\n  input: %s\n  diff: %s", i, chunk, cc, in, d)
		}
		same++
	}
	fmt.Printf("  random %d cases (seed=%d): identical %d, buffered failed %d (of which dropped-field type errors passed by streaming %d), known fallbacks %d, mismatches 0\n", N, seed, same, offFail, lenient, fallback)
}

// ---- source: openai_diff_test.go ----
// officialOpenAI reproduces the pure-function part of the default path (main.go normalizeOpenAiRequestBody →
// handleRequestBody: convertDeveloperRoleToSystem → the model rewrite of defaultTransformRequestBody).
func officialOpenAI(in string, mapping map[string]string, chat bool) (map[string]any, bool) {
	body := []byte(in)
	if chat {
		var err error
		if gjson.GetBytes(body, "stream").Bool() && (!gjson.GetBytes(body, "stream_options").Exists() || !gjson.GetBytes(body, "stream_options.include_usage").Exists()) {
			body, err = sjson.SetBytes(body, "stream_options.include_usage", true)
			if err != nil {
				return nil, false
			}
		}
		body, err = convertDeveloperRoleToSystem(body)
		if err != nil {
			return nil, false
		}
	}
	model := gjson.GetBytes(body, "model").String()
	body, err := sjson.SetBytes(body, "model", getMappedModel(model, mapping))
	if err != nil {
		return nil, false
	}
	return toMap(body)
}

func hasDeveloper(in string) bool {
	for _, m := range gjson.Get(in, "messages").Array() {
		if m.Get("role").String() == "developer" {
			return true
		}
	}
	return false
}

var oaiMapping = map[string]string{"m1": "mapped-1", "gpt-*": "g", "~^re(.*)$": "x$1", "*": "fallback"}

func TestOpenAIDifferential(t *testing.T) {
	cases := []string{
		`{"model":"m1","messages":[{"role":"user","content":"U"}]}`,
		`{"messages":[{"role":"user","content":"U"}],"model":"gpt-4o","stream":true}`,
		`{"model":"re9","stream":true,"stream_options":{"x":1},"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"zzz","stream":true,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m1","stream":false,"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m1","stream":"true","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m1","stream":1,"messages":[{"role":"user","content":"U"}]}`,
		`{"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m1","messages":[{"role":"user","content":[{"type":"text","text":"a"}]}],"tools":[{"x":{"y":[1,2,{"z":null}]}}],"anything":"goes","nested":{"model":"inner"}}`,
		`{"model":"m1","messages":[{"content":"U","role":"user","name":"n"}],"stream":true,"stream_options":{}}`,
		"{ \"model\" : \"m1\" ,\n \"messages\" : [ ] }",
		`{"model":"m1","messages":"not-an-array"}`,
		`{"model":"m1","messages":[1,"s",null,{"role":"user"}]}`,
		`{"model":"m1","messages":[{"role":"developer","content":"D"}]}`,
		`{"model":"m1","model":"dup","messages":[]}`,
	}
	fmt.Println("  OpenAI passthrough cases                         buffered vs streaming")
	for _, in := range cases {
		for _, chat := range []bool{true, false} {
			off, ok := officialOpenAI(in, oaiMapping, chat)
			if !ok {
				// on the chat path the buffered implementation Unmarshals the whole body (developer conversion) and fails on invalid types;
				// streaming passthrough does not validate and leaves the rejection to the upstream: a known, documented lenient difference.
				if chat {
					fmt.Printf("  %-48s   buffered struct validation failed, streaming passed through (known lenient difference)\n", trunc(in, 48))
				}
				continue
			}
			for _, cs := range []int{1, 5, 4096} {
				tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{
					MapModel:       func(m string) string { return getMappedModel(m, oaiMapping) },
					DetectStream:   chat,
					NormalizeUsage: chat,
					CheckMessages:  chat,
				})
				str, sok, why := runStream(tr, in, cs)
				if !sok {
					if chat && hasDeveloper(in) {
						if cs == 4096 {
							fmt.Printf("  %-48s   developer → fallback ✓\n", trunc(in, 48))
						}
						continue
					}
					t.Errorf("chunk=%d chat=%v unexpected fallback: %s\n  %s", cs, chat, why, in)
					continue
				}
				if d := diffMaps(off, str); d != "" {
					t.Errorf("chunk=%d chat=%v mismatch: %s\n  input: %s", cs, chat, d, in)
				} else if cs == 4096 && chat {
					fmt.Printf("  %-48s   ✓ identical\n", trunc(in, 48))
				}
			}
		}
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "..."
}

func TestOpenAIFuzz(t *testing.T) {
	seed := fuzzSeed()
	r := rand.New(rand.NewSource(seed))
	N := fuzzN(10000)
	same, fb := 0, 0
	for i := 0; i < N; i++ {
		in := genRequest(r)
		if r.Intn(3) == 0 {
			in = strings.Replace(in, `"model":`, `"stream":`+[]string{"true", "false", `"true"`, "1"}[r.Intn(4)]+`,"model":`, 1)
		}
		chat := r.Intn(4) != 0
		off, ok := officialOpenAI(in, oaiMapping, chat)
		if !ok {
			continue
		}
		chunk := []int{1, 3, 17, 64, 4096}[r.Intn(5)]
		tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{
			MapModel:       func(m string) string { return getMappedModel(m, oaiMapping) },
			DetectStream:   chat,
			NormalizeUsage: chat,
			CheckMessages:  chat,
		})
		str, sok, why := runStream(tr, in, chunk)
		if !sok {
			if chat && hasDeveloper(in) {
				fb++
				continue
			}
			t.Fatalf("case %d: unexpected fallback (chunk=%d chat=%v): %s\n  input: %s", i, chunk, chat, why, in)
		}
		if d := diffMaps(off, str); d != "" {
			t.Fatalf("case %d: mismatch (chunk=%d chat=%v)\n  input: %s\n  diff: %s", i, chunk, chat, in, d)
		}
		same++
	}
	fmt.Printf("  OpenAI passthrough random %d cases (seed=%d): identical %d, developer fallbacks %d, mismatches 0\n", N, seed, same, fb)
}

// ---- source: variants_diff_test.go ----
// The pure-function path of each buffered provider (main.go normalize → the preliminary steps of handleRequestBody → provider-specific logic → default model rewrite).
// Copied line by line from the buffered code, dropping only the ctx branches that depend on the host (the integration layer already declares those non-streaming).

func oaiPre(in string, chat bool) ([]byte, bool) {
	body := []byte(in)
	if chat {
		var err error
		if gjson.GetBytes(body, "stream").Bool() && (!gjson.GetBytes(body, "stream_options").Exists() || !gjson.GetBytes(body, "stream_options.include_usage").Exists()) {
			body, err = sjson.SetBytes(body, "stream_options.include_usage", true)
			if err != nil {
				return nil, false
			}
		}
		body, err = convertDeveloperRoleToSystem(body)
		if err != nil {
			return nil, false
		}
	}
	return body, true
}

func oaiDefaultModel(body []byte, mapping map[string]string) ([]byte, bool) {
	model := gjson.GetBytes(body, "model").String()
	body, err := sjson.SetBytes(body, "model", getMappedModel(model, mapping))
	return body, err == nil
}

func toMap(b []byte) (map[string]any, bool) {
	m, err := decodeMap(b)
	return m, err == nil
}

// officialQwenCompat: the compatible branch of qwen.go TransformRequestBodyHeaders
func officialQwenCompat(in string, mapping map[string]string, chat bool) (map[string]any, bool) {
	body, ok := oaiPre(in, chat)
	if !ok {
		return nil, false
	}
	modifiedBody := body
	model := gjson.GetBytes(modifiedBody, "model")
	mappedModel := ""
	if model.Exists() {
		rawModel := model.String()
		mappedModel = getMappedModel(rawModel, mapping)
		newBody, err := sjson.SetBytes(modifiedBody, "model", mappedModel)
		if err != nil {
			return nil, false
		}
		modifiedBody = newBody
	}
	if mappedModel != "" && requestBodyHasMessageReasoningContent(modifiedBody) && qwenSupportsPreserveThinking(mappedModel) {
		var err error
		modifiedBody, err = sjson.SetBytes(modifiedBody, "preserve_thinking", true)
		if err != nil {
			return nil, false
		}
	}
	return toMap(modifiedBody)
}

// officialZhipu: zhipuai.go TransformRequestBody (chat)
func officialZhipu(in string, mapping map[string]string, chat bool) (map[string]any, bool) {
	body, ok := oaiPre(in, chat)
	if !ok {
		return nil, false
	}
	if chat {
		reasoningEffort := gjson.GetBytes(body, "reasoning_effort").String()
		if reasoningEffort != "" {
			body, _ = sjson.SetBytes(body, "thinking", map[string]string{"type": "enabled"})
			body, _ = sjson.DeleteBytes(body, "reasoning_effort")
		}
		if requestBodyHasMessageReasoningContent(body) {
			body, _ = sjson.SetBytes(body, "thinking.clear_thinking", false)
		}
	}
	body, ok = oaiDefaultModel(body, mapping)
	if !ok {
		return nil, false
	}
	return toMap(body)
}

// officialOpenRouter: openrouter.go TransformRequestBody (chat, without the Claude ctx branch)
func officialOpenRouter(in string, mapping map[string]string, chat bool) (map[string]any, bool) {
	body, ok := oaiPre(in, chat)
	if !ok {
		return nil, false
	}
	if chat {
		reasoningMaxTokens := gjson.GetBytes(body, "reasoning_max_tokens")
		if reasoningMaxTokens.Exists() && reasoningMaxTokens.Int() != 0 {
			modifiedBody, err := sjson.DeleteBytes(body, "reasoning_effort")
			if err != nil {
				modifiedBody = body
			}
			modifiedBody, err = sjson.SetBytes(modifiedBody, "reasoning.max_tokens", reasoningMaxTokens.Int())
			if err != nil {
				return nil, false
			}
			modifiedBody, err = sjson.DeleteBytes(modifiedBody, "reasoning_max_tokens")
			if err != nil {
				return nil, false
			}
			body = modifiedBody
		}
	}
	body, ok = oaiDefaultModel(body, mapping)
	if !ok {
		return nil, false
	}
	return toMap(body)
}

var vMapping = map[string]string{"m1": "qwen3.6-plus-x", "k": "kimi-k2.6", "gpt-*": "g", "*": "fallback"}

type variantCase struct {
	name     string
	official func(in string, mapping map[string]string, chat bool) (map[string]any, bool)
	stream   func(chat bool) *streamxform.Transformer
	inputs   []string
}

func variantCases() []variantCase {
	mk := func(v func() streamxform.OpenAIVariant, onlyIfPresent, detect bool) func(bool) *streamxform.Transformer {
		return func(chat bool) *streamxform.Transformer {
			var vv streamxform.OpenAIVariant
			if v != nil && chat {
				vv = v()
			}
			return streamxform.NewOpenAI(streamxform.OpenAIOptions{
				MapModel:           func(m string) string { return getMappedModel(m, vMapping) },
				ModelOnlyIfPresent: onlyIfPresent,
				DetectStream:       detect && chat,
				NormalizeUsage:     chat,
				CheckMessages:      chat,
				Variant:            vv,
			})
		}
	}
	return []variantCase{
		{"qwen compatible", officialQwenCompat, func(chat bool) *streamxform.Transformer {
			return streamxform.NewOpenAI(streamxform.OpenAIOptions{
				MapModel:           func(m string) string { return getMappedModel(m, vMapping) },
				ModelOnlyIfPresent: true,
				NormalizeUsage:     chat,
				CheckMessages:      chat,
				Variant:            &streamxform.QwenVariant{SupportsPreserveThinking: qwenSupportsPreserveThinking},
			})
		}, []string{
			`{"model":"m1","messages":[{"role":"user","content":"U"}]}`,
			`{"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","messages":[{"role":"assistant","content":"A","reasoning_content":"think"},{"role":"user","content":"U"}]}`,
			`{"model":"k","messages":[{"role":"assistant","reasoning_content":"","content":"A"},{"role":"assistant","reasoning_content":"x"}]}`,
			`{"model":"zzz","messages":[{"role":"assistant","reasoning_content":"x"}]}`,
			`{"model":"m1","preserve_thinking":false,"messages":[{"role":"assistant","reasoning_content":"x"}]}`,
			`{"model":"m1","preserve_thinking":false,"messages":[{"role":"assistant","reasoning_content":""}]}`,
			`{"model":"m1","messages":[{"role":"assistant","reasoning_content":0},{"role":"user","content":"U"}]}`,
			`{"model":"m1","messages":[{"role":"assistant","reasoning_content":null}]}`,
			`{"model":"m1","messages":[{"role":"assistant","reasoning_content":{"a":1}}]}`,
			`{"model":"m1","messages":["s",1,null,{"reasoning_content":"x"}]}`,
			`{"model":"m1","stream":true,"messages":[{"role":"assistant","reasoning_content":"` + strings.Repeat("r", 5000) + `"}]}`,
			`{"messages":[{"role":"assistant","reasoning_content":"x"}],"model":"m1"}`,
		}},
		{"zhipu", officialZhipu, mk(func() streamxform.OpenAIVariant { return &streamxform.ZhipuVariant{} }, false, true), []string{
			`{"model":"m1","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_effort":"high","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_effort":"","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_effort":"high","thinking":{"type":"disabled","x":1},"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","thinking":{"type":"disabled"},"messages":[{"role":"assistant","reasoning_content":"x"}]}`,
			`{"model":"m1","messages":[{"role":"assistant","reasoning_content":"x"}]}`,
			`{"model":"m1","reasoning_effort":"low","messages":[{"role":"assistant","reasoning_content":"x"}]}`,
			`{"model":"m1","reasoning_effort":5,"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_effort":null,"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","thinking":{"clear_thinking":true},"messages":[{"role":"assistant","reasoning_content":"x"}]}`,
			`{"messages":[{"role":"assistant","reasoning_content":"x"}],"thinking":{"type":"enabled"},"reasoning_effort":"medium","model":"gpt-4"}`,
		}},
		{"openrouter", officialOpenRouter, mk(func() streamxform.OpenAIVariant { return &streamxform.OpenRouterVariant{} }, false, true), []string{
			`{"model":"m1","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":1000,"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":0,"reasoning_effort":"high","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":1000,"reasoning_effort":"high","reasoning":{"exclude":true,"max_tokens":5},"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":"200","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":"2.5","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":12.9,"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":true,"messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_max_tokens":null,"reasoning_effort":"low","messages":[{"role":"user","content":"U"}]}`,
			`{"model":"m1","reasoning_effort":"low","reasoning":{"effort":"x"},"messages":[{"role":"user","content":"U"}]}`,
			`{"reasoning_effort":"low","messages":[{"role":"user","content":"U"}],"reasoning_max_tokens":-5,"model":"m1"}`,
		}},
	}
}

func TestVariantsDifferential(t *testing.T) {
	for _, vc := range variantCases() {
		fmt.Printf("  == %s\n", vc.name)
		for _, in := range vc.inputs {
			for _, chat := range []bool{true, false} {
				off, ok := vc.official(in, vMapping, chat)
				if !ok {
					if chat {
						fmt.Printf("  %-52s buffered struct validation failed\n", trunc(in, 52))
					}
					continue
				}
				for _, cs := range []int{1, 5, 4096} {
					str, sok, why := runStream(vc.stream(chat), in, cs)
					if !sok {
						if chat && hasDeveloper(in) {
							continue
						}
						t.Errorf("%s chunk=%d chat=%v unexpected fallback: %s\n  %s", vc.name, cs, chat, why, in)
						continue
					}
					if d := diffMaps(off, str); d != "" {
						t.Errorf("%s chunk=%d chat=%v mismatch: %s\n  input: %s", vc.name, cs, chat, d, in)
					} else if cs == 4096 && chat {
						fmt.Printf("  %-52s ✓ identical\n", trunc(in, 52))
					}
				}
			}
		}
	}
}

// genVariantRequest adds the fields each variant cares about on top of genRequest
func genVariantRequest(r *rand.Rand) string {
	in := genRequest(r)
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	extra := []string{}
	if r.Intn(3) == 0 {
		extra = append(extra, `"preserve_thinking":`+pick("true", "false", "null", `"x"`))
	}
	if r.Intn(3) == 0 {
		extra = append(extra, `"thinking":`+pick(`{"type":"disabled"}`, `{"type":"enabled","clear_thinking":true}`, `{}`, `null`, `"str"`, `[1]`))
	}
	if r.Intn(3) == 0 {
		extra = append(extra, `"reasoning":`+pick(`{"effort":"low"}`, `{"max_tokens":5,"exclude":true}`, `{}`, `null`, `7`))
	}
	if r.Intn(3) == 0 {
		extra = append(extra, `"reasoning_max_tokens":`+pick("0", "100", "-3", "2.9", `"15"`, `"x"`, "true", "null", `""`))
	}
	if r.Intn(2) == 0 {
		in = strings.Replace(in, `"model":`, `"model_x":`, 1) // sometimes drop model
		in = strings.Replace(in, `"model_x":`, `"model":`, 0)
	}
	if len(extra) > 0 {
		in = "{" + strings.Join(extra, ",") + "," + in[1:]
	}
	// add reasoning_content in its various shapes to some messages
	if r.Intn(2) == 0 {
		in = strings.Replace(in, `"role":"assistant"`, `"role":"assistant","reasoning_content":`+pick(`"t"`, `""`, `null`, `0`, `1`, `{"a":1}`, `[]`, `"`+strings.Repeat("z", 3000)+`"`), 1)
	}
	return in
}

func TestVariantsFuzz(t *testing.T) {
	seed := fuzzSeed()
	r := rand.New(rand.NewSource(seed))
	N := fuzzN(6000)
	for _, vc := range variantCases() {
		same, offFail, fb := 0, 0, 0
		for i := 0; i < N; i++ {
			in := genVariantRequest(r)
			chat := r.Intn(4) != 0
			off, ok := vc.official(in, vMapping, chat)
			if !ok {
				offFail++
				continue
			}
			chunk := []int{1, 3, 17, 64, 4096}[r.Intn(5)]
			str, sok, why := runStream(vc.stream(chat), in, chunk)
			if !sok {
				if chat && hasDeveloper(in) || strings.Contains(why, "is not an object") || strings.Contains(why, "messages is not an array") || strings.Contains(why, "duplicate key") {
					fb++
					continue
				}
				t.Fatalf("%s case %d: unexpected fallback (chunk=%d chat=%v): %s\n  input: %s", vc.name, i, chunk, chat, why, in)
			}
			if d := diffMaps(off, str); d != "" {
				t.Fatalf("%s case %d: mismatch (chunk=%d chat=%v)\n  input: %s\n  diff: %s", vc.name, i, chunk, chat, in, d)
			}
			same++
		}
		fmt.Printf("  %s random %d cases (seed=%d): identical %d, buffered failed %d, known fallbacks %d, mismatches 0\n", vc.name, N, seed, same, offFail, fb)
	}
}

// discardedFields are the top-level fields buildClaudeTextGenRequest does not read and streaming drops directly.
// The buffered path fails on their type errors while streaming passes: the only lenience allowed.
var discardedFields = map[string]bool{
	"seed": true, "n": true, "presence_penalty": true, "frequency_penalty": true, "top_logprobs": true,
	"logprobs": true, "user": true, "metadata": true, "response_format": true, "stream_options": true,
	"logit_bias": true, "modalities": true, "audio": true, "prediction": true, "service_tier": true,
	"store": true, "web_search_options": true, "chat_template_kwargs": true, "enable_thinking": true,
	"thinking": true, "preserve_thinking": true, "search_parameters": true, "enable_search": true, "extra_body": true,
}

// geminiExtraDiscarded: top-level fields buildGeminiChatRequest additionally does not read (Claude reads them, Gemini does not).
var geminiExtraDiscarded = map[string]bool{
	"stop": true, "tool_choice": true, "parallel_tool_calls": true, "max_completion_tokens": true,
	"reasoning_effort": true, "reasoning_max_tokens": true, "claude_thinking": true, "claude_output_config": true,
	"claude_anthropic_beta": true, "claude_content_blocks": true, "claude_content_block_stop": true,
}

// qwenNativeExtraDiscarded: top-level fields buildQwenTextGenerationRequest additionally does not read.
var qwenNativeExtraDiscarded = map[string]bool{
	"stop": true, "tool_choice": true, "parallel_tool_calls": true, "max_completion_tokens": true,
	"reasoning_effort": true, "reasoning_max_tokens": true, "claude_thinking": true, "claude_output_config": true,
	"claude_anthropic_beta": true, "claude_content_blocks": true, "claude_content_block_stop": true,
	"modalities": true,
}

// typeErrorField extracts the field name (last segment) from an encoding/json type error.
func typeErrorField(err error) (string, bool) {
	msg := err.Error()
	if !strings.Contains(msg, "cannot unmarshal") {
		return "", false
	}
	// "json: cannot unmarshal X into Go struct field chatCompletionRequest.FIELD of type T"
	i := strings.Index(msg, "struct field ")
	if i < 0 {
		return "", false
	}
	rest := msg[i+len("struct field "):]
	if j := strings.Index(rest, " of type"); j >= 0 {
		rest = rest[:j]
	}
	// take the last segment of the field name (chatCompletionRequest.metadata / chatMessage.name ...)
	return rest[strings.LastIndex(rest, ".")+1:], true
}

func isDiscardedFieldTypeError(err error) bool {
	f, ok := typeErrorField(err)
	return ok && discardedFields[f]
}

// isDiscardedFieldTypeErrorIn: fields the target protocol additionally does not read count as dropped as well.
func isDiscardedFieldTypeErrorIn(err error, extra map[string]bool) bool {
	f, ok := typeErrorField(err)
	return ok && (discardedFields[f] || extra[f])
}

// 网关上实测分叉的那 12 个字段：缓冲路径按结构体解码会拒，流式此前放行给了供应商。
// 这里逐个钉住，任何一个回到"流式放行"都要立刻失败。
func TestRootFieldTypesMatchBufferedRejection(t *testing.T) {
	bad := map[string]string{
		"frequency_penalty": `"x"`,
		"logit_bias":        `"x"`,
		"logprobs":          `"true"`,
		"metadata":          `"m"`,
		"n":                 `"one"`,
		"presence_penalty":  `{}`,
		"response_format":   `"json"`,
		"seed":              `"abc"`,
		"service_tier":      `9`,
		"stream_options":    `"x"`,
		"top_logprobs":      `"3"`,
		"user":              `42`,
	}
	for f, v := range bad {
		in := `{"model":"m","messages":[{"role":"user","content":"U"}],"` + f + `":` + v + `}`
		if _, oerr := officialClaudeErr(in, false); oerr == nil {
			t.Fatalf("字段 %s：缓冲路径居然接受了 %s，用例失效", f, v)
		}
		for _, chunk := range []int{1, 17, 4096} {
			_, sok, _ := runStream(typed(streamxform.NewClaude(streamxform.ClaudeOptions{})), in, chunk)
			if sok {
				t.Errorf("字段 %s = %s（分块 %d）：缓冲路径拒绝而流式放行了", f, v, chunk)
			}
		}
	}
}

// 反方向同样要钉住：类型正确的值一个都不能被拦。
func TestRootFieldTypesAcceptValidValues(t *testing.T) {
	good := map[string]string{
		"frequency_penalty": `0.5`,
		"logit_bias":        `{"a":1}`,
		"logprobs":          `true`,
		"metadata":          `{"k":"v"}`,
		"n":                 `1`,
		"presence_penalty":  `0`,
		"response_format":   `{"type":"json_object"}`,
		"seed":              `42`,
		"service_tier":      `"auto"`,
		"stream_options":    `{"include_usage":true}`,
		"top_logprobs":      `3`,
		"user":              `"u"`,
		"max_tokens":        `null`,
		"temperature":       `null`,
		"stop":              `null`,
	}
	for f, v := range good {
		in := `{"model":"m","messages":[{"role":"user","content":"U"}],"` + f + `":` + v + `}`
		for _, chunk := range []int{1, 17, 4096} {
			if _, sok, why := runStream(typed(streamxform.NewClaude(streamxform.ClaudeOptions{})), in, chunk); !sok {
				t.Errorf("字段 %s = %s（分块 %d）被误拦：%s", f, v, chunk, why)
			}
		}
	}
}
