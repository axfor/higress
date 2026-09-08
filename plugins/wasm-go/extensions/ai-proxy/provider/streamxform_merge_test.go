package provider

// Differential tests of the mergeConsecutiveMessages stage against the buffered helper: hand-written shapes and a
// random corpus, at chunk sizes 1, 7 and 4096. When the helper merged, it re-serialised the request through its
// struct; the streaming output is compared after the same round trip. When it did not, the body is unchanged and
// the streaming output must be it, byte for byte after decoding.

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// officialMerge runs the buffered helper, reporting a panic (ParseContent asserts image_url.url and friends
// without checking) as crashed.
func officialMerge(in string) (out []byte, err error, crashed bool) {
	defer func() {
		if r := recover(); r != nil {
			crashed = true
		}
	}()
	out, err = mergeConsecutiveMessages([]byte(in))
	return
}

func checkMerge(t *testing.T, in string) {
	t.Helper()
	offB, err, crashed := officialMerge(in)
	changed := err == nil && !crashed && string(offB) != in
	off, decErr := decodeMap(offB)
	for _, chunk := range []int{1, 7, 4096} {
		str, ok, why := runStream(streamxform.NewMergeConsecutive(), in, chunk)
		if crashed {
			require.False(t, ok, "chunk=%d: the helper panics but the stage passed\n  %s", chunk, truncateIn(in))
			require.Contains(t, why, "panics")
			continue
		}
		if err != nil {
			// the helper keeps the body and logs: the stage either passes it unchanged or reports unsupported, and
			// the buffered path then does the same
			if ok {
				orig, err := decodeMap([]byte(in))
				require.NoError(t, err)
				require.Empty(t, diffMaps(orig, str), "chunk=%d: the helper failed (%v), the stage must leave the body alone\n  %s", chunk, err, truncateIn(in))
			}
			continue
		}
		require.NoError(t, decErr)
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", chunk, why, truncateIn(in))
		if changed {
			// both sides through the struct once more: the helper's typed nil content slice marshals as null, a
			// decoded null does not survive a re-marshal, so only a common round trip compares like with like
			require.Empty(t, diffMaps(roundTripChat(t, off), roundTripChat(t, str)), "chunk=%d (merged)\n  %s", chunk, truncateIn(in))
		} else {
			require.Empty(t, diffMaps(off, str), "chunk=%d (unchanged)\n  %s", chunk, truncateIn(in))
		}
	}
}

func mergeCases() []string {
	big := strings.Repeat("m", 70000)
	return []string{
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":"b"},{"role":"user","content":"c"},{"role":"assistant","content":"d"},{"role":"assistant","content":"e"}]}`,
		`{"model":"m","messages":[{"role":"system","content":"s1"},{"role":"system","content":"s2"},{"role":"user","content":"u"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a","name":"n1","foo":1},{"role":"user","content":"b","name":"n2","bar":2}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"a"}]},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":[{"type":"text","text":"b"},{"type":"image_url","image_url":{"url":"http://x/y.png","detail":"low"}},{"type":"input_audio","input_audio":{"data":"AA","format":"wav"}},{"type":"file","file":{"file_id":"f1"}},{"type":"other","x":1},5,"s"]}]}`,
		`{"model":"m","messages":[{"role":"user","content":null},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user"},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user","content":""},{"role":"user","content":""}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"a","reasoning_content":"r1"},{"role":"assistant","content":"b","reasoning_content":"r2"}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"a"},{"role":"assistant","content":"b","reasoning_content":"r2"}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"a","reasoning_content":"r1"},{"role":"assistant","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"a","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"assistant","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"a"},{"role":"assistant","content":"b","tool_calls":[]}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"a"},{"role":"assistant","content":"b","tool_calls":null}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"a","function_call":{"name":"f","arguments":"{}"}},{"role":"assistant","content":"b"}]}`,
		`{"role_less":1,"model":"m","messages":[{"content":"a"},{"content":"b"}]}`,
		`{"model":"m","messages":[{"role":"tool","content":"a","tool_call_id":"c1"},{"role":"tool","content":"b","tool_call_id":"c2"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a"}]}`,
		`{"model":"m","messages":[]}`,
		`{"model":"m","messages":null}`,
		`{"model":"m"}`,
		`{"model":"m","messages":"str"}`,
		`{"model":"m","messages":[5,{"role":"user","content":"a"}]}`,
		`{"model":"m","messages":[{"role":5,"content":"a"},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a","reasoning_content":5},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":[{"type":"image_url","image_url":{"detail":"low"}}]}]}`,
		`{"model":"m","messages":[{"role":"user","content":"` + big + `"},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":"` + big + `"}],"stream":true,"temperature":0}`,
		`{"model":"m","messages":[{"role":"user","content":"a\"b\\c\nd"},{"role":"user","content":"é"}]}`,
		`{"model":"m","messages":[{"role":"user","content":{"weird":1}},{"role":"user","content":"b"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":{"weird":1}}]}`,
	}
}

func TestMergeDifferential(t *testing.T) {
	for _, in := range mergeCases() {
		checkMerge(t, in)
	}
}

func genMergeRequest(r *rand.Rand) string {
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	q := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	content := func() string {
		switch r.Intn(8) {
		case 0:
			return "null"
		case 1:
			return ""
		case 2:
			return `[{"type":"text","text":` + q(pick("x", "", "y z")) + `}]`
		case 3:
			return `[{"type":"image_url","image_url":{"url":"http://x/` + fmt.Sprint(r.Intn(9)) + `.png"` + pick("", `,"detail":"low"`) + `}},{"type":"text","text":"t"}]`
		case 4:
			return `[{"type":"input_audio","input_audio":{"data":"AA","format":"mp3"}},{"type":"file","file":{"file_id":"f"}},{"type":"nope"},7]`
		default:
			return q(pick("a", "b", "hello world", "", "multi\nline"))
		}
	}
	var msgs []string
	for i := r.Intn(7); i > 0; i-- {
		var kv []string
		role := pick("user", "user", "assistant", "assistant", "system", "tool", "developer")
		if r.Intn(12) != 0 {
			kv = append(kv, `"role":`+q(role))
		}
		if c := content(); c != "" {
			kv = append(kv, `"content":`+c)
		}
		if r.Intn(4) == 0 {
			kv = append(kv, `"reasoning_content":`+q(pick("r", "", "think")))
		}
		if r.Intn(6) == 0 {
			kv = append(kv, `"tool_calls":`+pick(`[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]`, "[]", "null"))
		}
		if r.Intn(8) == 0 {
			kv = append(kv, `"function_call":{"name":"f","arguments":"{}"}`)
		}
		if r.Intn(4) == 0 {
			kv = append(kv, `"name":"n`+fmt.Sprint(r.Intn(3))+`"`)
		}
		if r.Intn(5) == 0 {
			kv = append(kv, `"extra":{"a":[1,2]}`)
		}
		r.Shuffle(len(kv), func(i, j int) { kv[i], kv[j] = kv[j], kv[i] })
		msgs = append(msgs, "{"+strings.Join(kv, ",")+"}")
	}
	top := []string{`"model":"m"`, `"messages":[` + strings.Join(msgs, ",") + `]`}
	if r.Intn(2) == 0 {
		top = append(top, `"stream":`+pick("true", "false"), `"temperature":`+pick("0", "0.5"))
	}
	r.Shuffle(len(top), func(i, j int) { top[i], top[j] = top[j], top[i] })
	return "{" + strings.Join(top, ",") + "}"
}

func TestMergeFuzz(t *testing.T) {
	r := rand.New(rand.NewSource(56))
	for i := 0; i < 400; i++ {
		checkMerge(t, genMergeRequest(r))
	}
}
