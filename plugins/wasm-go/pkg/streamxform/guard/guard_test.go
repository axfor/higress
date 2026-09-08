package guard

import (
	"strings"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func stubHost(t *testing.T, body []byte) {
	orig := hostRequestBody
	hostRequestBody = func(start, size int) ([]byte, error) {
		if start+size > len(body) {
			size = len(body) - start
		}
		return body[start : start+size], nil
	}
	t.Cleanup(func() { hostRequestBody = orig })
}

func feedAll(s *State, body []byte, chunk int) (out []byte, acts []types.Action) {
	for i := 0; i < len(body); i += chunk {
		j := i + chunk
		if j > len(body) {
			j = len(body)
		}
		o, a := s.Feed(body[i:j], j == len(body))
		acts = append(acts, a)
		out = append(out, o...)
	}
	return
}

func big(n int) string { return strings.Repeat("y", n) }

// PrefixTransform: the transformer sees the bytes before the commit point (model rewrite); after release the rest is forwarded verbatim.
func TestPrefixTransform(t *testing.T) {
	body := []byte(`{"model":"p/m1","messages":[{"role":"user","content":"` + big(200<<10) + `"}]}`)
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{MapModel: func(m string) string {
		if i := strings.Index(m, "/"); i >= 0 {
			return m[i+1:]
		}
		return m
	}})
	var committed, finished int
	s := New(&Plan{Tr: tr, Mode: PrefixTransform,
		OnCommit: func(pre streamxform.Prelude, last bool) bool { committed++; return pre.ModelSeen },
		OnFinish: func(streamxform.Prelude) { finished++ },
	})
	out, acts := feedAll(s, body, 4096)
	if acts[0] != types.ActionPause || acts[len(acts)-1] != types.ActionContinue {
		t.Fatalf("first chunk should Pause and the last should Continue: %v", acts[:2])
	}
	want := strings.Replace(string(body), `"model":"p/m1"`, `"model":"m1"`, 1)
	if string(out) != want {
		t.Fatalf("output differs from the expectation (len %d vs %d)", len(out), len(want))
	}
	if committed != 1 || finished != 1 || !s.raw {
		t.Fatalf("committed=%d finished=%d raw=%v", committed, finished, s.raw)
	}
}

// Observe: input forwarded verbatim; a bail only stops observing.
func TestObserve(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"` + big(100<<10) + `"}],"stream":tru}`)
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{})
	bailed := 0
	s := New(&Plan{Tr: tr, Mode: Observe, Metric: func(n string) {
		if n == "observe_bailed" {
			bailed++
		}
	}})
	out, acts := feedAll(s, body, 1000)
	for _, a := range acts {
		if a != types.ActionContinue {
			t.Fatalf("observe mode must always Continue: %v", acts)
		}
	}
	if string(out) != string(body) {
		t.Fatal("observe mode must forward the input verbatim")
	}
	if bailed != 1 {
		t.Fatalf("an invalid literal should stop observation once, bailed=%d", bailed)
	}
}

// Bail before the commit point → fallback to the buffered path (handed to Fallback once complete, and its written body read back).
func TestFallbackBeforeCommit(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"U"}],"stream":tru}`)
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{})
	var got []byte
	s := New(&Plan{Tr: tr, Mode: Transform, Fallback: func(b []byte) types.Action { got = b; return types.ActionContinue }})
	out, acts := feedAll(s, body, 16)
	if acts[len(acts)-1] != types.ActionContinue || string(got) != string(body) || string(out) != string(body) {
		t.Fatalf("the fallback should hand the last chunk to the buffered path and return its body: acts=%v got=%d out=%d", acts[len(acts)-1:], len(got), len(out))
	}
	if !s.FellBack() {
		t.Fatal("should be marked as fallen back")
	}
}

// Bail after the commit point (Transform mode) → Uncoverable.
func TestUncoverableAfterCommit(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"` + big(100<<10) + `"}],"stream":tru}`)
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{})
	unc := ""
	s := New(&Plan{Tr: tr, Mode: Transform, Uncoverable: func(r string) { unc = r }})
	_, acts := feedAll(s, body, 4096)
	if unc == "" || acts[len(acts)-1] != types.ActionPause {
		t.Fatalf("an invalid literal after the commit point should trigger Uncoverable: unc=%q last=%v", unc, acts[len(acts)-1])
	}
}

// Metrics are broken down by the engine's Error.Code: fallback.<code> / uncoverable.<code> / observe_bailed.<code>, never by message text.
func TestMetricsByCode(t *testing.T) {
	collect := func() (func(string), *[]string) {
		var got []string
		return func(n string) { got = append(got, n) }, &got
	}
	// syntax error before the commit point → fallback.syntax
	body := []byte(`{"a":1,"b":tru}`)
	stubHost(t, body)
	m, got := collect()
	s := New(&Plan{Tr: streamxform.NewTransformer(streamxform.BaseProtocol{}), Mode: Transform, Metric: m,
		Fallback: func([]byte) types.Action { return types.ActionContinue }})
	feedAll(s, body, 4)
	if strings.Join(*got, ",") != "fallback,fallback.syntax" {
		t.Fatalf("metrics for a syntax error: %v", *got)
	}
	// duplicate key before the commit point → fallback.duplicate_key
	body = []byte(`{"a":1,"a":2}`)
	stubHost(t, body)
	m, got = collect()
	tr := streamxform.NewTransformer(streamxform.BaseProtocol{})
	tr.DupKeyBail = true
	s = New(&Plan{Tr: tr, Mode: Transform, Metric: m, Fallback: func([]byte) types.Action { return types.ActionContinue }})
	feedAll(s, body, 4)
	if strings.Join(*got, ",") != "fallback,fallback.duplicate_key" {
		t.Fatalf("metrics for a duplicate key: %v", *got)
	}
	// syntax error after the commit point → uncoverable.syntax, and Uncoverable gets a reason with the offset
	body = []byte(`{"pad":"` + big(200<<10) + `","b":tru}`)
	stubHost(t, body)
	m, got = collect()
	var why string
	s = New(&Plan{Tr: streamxform.NewTransformer(streamxform.BaseProtocol{}), Mode: Transform, Metric: m,
		Uncoverable: func(r string) { why = r }})
	feedAll(s, body, 4096)
	if strings.Join(*got, ",") != "streamed,uncoverable,uncoverable.syntax" {
		t.Fatalf("metrics after the commit point: %v", *got)
	}
	if !strings.Contains(why, "incomplete literal at byte") {
		t.Fatalf("the Uncoverable reason should carry the offset: %q", why)
	}
	// Observe mode → observe_bailed.syntax, input still forwarded
	body = []byte(`{"a":1,"b":tru}`)
	m, got = collect()
	s = New(&Plan{Tr: streamxform.NewTransformer(streamxform.BaseProtocol{}), Mode: Observe, Metric: m})
	out, _ := feedAll(s, body, 4)
	if string(out) != string(body) || strings.Join(*got, ",") != "streamed,observe_bailed,observe_bailed.syntax" {
		t.Fatalf("observe mode: out=%q metrics=%v", out, *got)
	}
}

// The host can deliver the whole body before it signals the end of the stream: Envoy calls the body hook once
// more with an empty chunk and end_of_stream set. PrefixTransform used to drop the transformer as soon as it
// released, so the bytes the engine still held for Finish (the root's closing brace and any trailing
// whitespace) never reached the upstream and the body arrived truncated.
func TestPrefixTransformEndOfStreamArrivesEmpty(t *testing.T) {
	// Chunked the way Envoy delivers a 70KB body: the third chunk both crosses the 64KB commit point and
	// carries the closing brace, so the release and the end of the root happen in the same call.
	body := []byte(`{"model":"p/m1","messages":[{"role":"user","content":"` + big(70000) + `"}]}` + "\n")
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{MapModel: func(m string) string {
		if i := strings.Index(m, "/"); i >= 0 {
			return m[i+1:]
		}
		return m
	}})
	kept := 0
	s := New(&Plan{Tr: tr, Mode: PrefixTransform,
		OnCommit: func(pre streamxform.Prelude, last bool) bool { return pre.ModelSeen },
		Metric: func(n string) {
			if n == "prefix_kept_for_finish" {
				kept++
			}
		},
	})
	// The exact split Envoy produced on the gateway: the first two chunks stay just under the 64KB commit
	// point, so the third one crosses it and carries the closing brace at once.
	var out []byte
	for _, part := range [][]byte{body[:32585], body[32585:65353], body[65353:]} {
		o, _ := s.Feed(part, false) // never the last chunk: the end of the stream arrives on its own
		out = append(out, o...)
	}
	o, a := s.Feed(nil, true)
	out = append(out, o...)
	if a != types.ActionContinue {
		t.Fatalf("the end-of-stream call must continue, got %v", a)
	}
	want := strings.Replace(string(body), `"model":"p/m1"`, `"model":"m1"`, 1)
	if string(out) != want {
		t.Fatalf("body truncated: got %d bytes, want %d; tail %q vs %q",
			len(out), len(want), tailOf(string(out)), tailOf(want))
	}
	if kept != 1 {
		t.Fatalf("the branch that keeps the transformer for Finish should be counted once, got %d", kept)
	}
	if s.raw {
		t.Fatal("the transformer must not have been dropped: the engine still held the tail")
	}
}

func tailOf(s string) string {
	if len(s) > 24 {
		return s[len(s)-24:]
	}
	return s
}
