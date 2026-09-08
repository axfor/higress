package guard

import (
	"math/rand"
	"sort"
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

// 上一个测试用的是 Envoy 那一次的切分点。真正要守住的性质与切分无关：无论请求体怎么分块，
// 也无论宿主是把 end_of_stream 挂在最后一块上还是单独再回调一次，转发出去的字节都必须是完整的。
// 截断缺陷正是只在某些切分下出现（提交点与根闭合落在同一块），逐点扫一遍才不会再漏。
func TestPrefixTransformIsCompleteUnderAnySplit(t *testing.T) {
	body := []byte(`{"model":"p/m1","messages":[{"role":"user","content":"` + big(70000) + `"}],"max_tokens":16}` + "\n")
	want := strings.Replace(string(body), `"model":"p/m1"`, `"model":"m1"`, 1)

	run := func(t *testing.T, splits []int, eosSeparate bool) {
		t.Helper()
		stubHost(t, body)
		tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{MapModel: func(m string) string {
			if i := strings.Index(m, "/"); i >= 0 {
				return m[i+1:]
			}
			return m
		}})
		s := New(&Plan{Tr: tr, Mode: PrefixTransform,
			OnCommit: func(pre streamxform.Prelude, last bool) bool { return pre.ModelSeen },
		})
		var out []byte
		prev := 0
		for i, cut := range splits {
			last := !eosSeparate && i == len(splits)-1
			o, _ := s.Feed(body[prev:cut], last)
			out = append(out, o...)
			prev = cut
		}
		if eosSeparate {
			o, _ := s.Feed(nil, true)
			out = append(out, o...)
		}
		if string(out) != want {
			t.Fatalf("splits=%v eos单独=%v: 输出 %d 字节，应为 %d；尾部 %q vs %q",
				splits, eosSeparate, len(out), len(want), tailOf(string(out)), tailOf(want))
		}
	}

	// 两块：切点扫过提交点附近的每一个位置，这里最容易让根闭合和提交点撞在一起
	for cut := 60 << 10; cut < len(body); cut += 97 {
		for _, sep := range []bool{true, false} {
			run(t, []int{cut, len(body)}, sep)
		}
	}
	// 三块：第二刀落在提交点前后，第三块同时跨过提交点并带上根闭合
	for a := 30 << 10; a < 34<<10; a += 251 {
		for b := 63 << 10; b < 67<<10; b += 251 {
			if b <= a {
				continue
			}
			for _, sep := range []bool{true, false} {
				run(t, []int{a, b, len(body)}, sep)
			}
		}
	}
	// 随机切分：块数与位置都随机，兜住上面没枚举到的形状
	rnd := rand.New(rand.NewSource(7))
	for i := 0; i < 200; i++ {
		n := 1 + rnd.Intn(6)
		cuts := map[int]bool{}
		for j := 0; j < n; j++ {
			cuts[1+rnd.Intn(len(body)-1)] = true
		}
		var splits []int
		for c := range cuts {
			splits = append(splits, c)
		}
		sort.Ints(splits)
		splits = append(splits, len(body))
		run(t, splits, i%2 == 0)
	}
}

// PrefixTransform 释放之后会丢掉转换器，后续分块原样透传。此时 OnFinish 仍然要拿到协议扫出来的
// prelude —— 从已经置空的转换器上读会静默得到零值，调用方看不出任何异常。
func TestPrefixTransformOnFinishKeepsThePrelude(t *testing.T) {
	body := []byte(`{"model":"p/m1","stream":true,"messages":[{"role":"user","content":"` + big(200<<10) + `"}]}`)
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{MapModel: func(m string) string {
		if i := strings.Index(m, "/"); i >= 0 {
			return m[i+1:]
		}
		return m
	}})
	var atCommit, atFinish streamxform.Prelude
	s := New(&Plan{Tr: tr, Mode: PrefixTransform,
		OnCommit: func(pre streamxform.Prelude, last bool) bool { atCommit = pre; return pre.ModelSeen },
		OnFinish: func(pre streamxform.Prelude) { atFinish = pre },
	})
	feedAll(s, body, 4096)
	if !s.raw {
		t.Fatal("这份请求体应当走到透传阶段")
	}
	if !atCommit.ModelSeen {
		t.Fatal("提交时应当已经看到 model")
	}
	if atFinish != atCommit {
		t.Fatalf("OnFinish 拿到的 prelude 与提交时不一致：%+v vs %+v", atFinish, atCommit)
	}
}

// 提前提交只改变何时释放，不改变释放什么：有 EarlyCommit 的输出必须与没有的逐字节相同，
// 而且第一个带 model 的块之后就该放行，不再等 64KB。
func TestEarlyCommitReleasesOnceModelSeen(t *testing.T) {
	body := []byte(`{"model":"p/m1","messages":[{"role":"user","content":"` + big(200<<10) + `"}],"max_tokens":16}`)
	stubHost(t, body)
	mk := func(early bool) *State {
		tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{MapModel: func(m string) string {
			if i := strings.Index(m, "/"); i >= 0 {
				return m[i+1:]
			}
			return m
		}})
		p := &Plan{Tr: tr, Mode: Transform,
			OnCommit: func(pre streamxform.Prelude, last bool) bool { return pre.ModelSeen }}
		if early {
			p.EarlyCommit = func(pre streamxform.Prelude) bool { return pre.ModelSeen }
		}
		return New(p)
	}
	want := strings.Replace(string(body), `"model":"p/m1"`, `"model":"m1"`, 1)

	// 无提前提交：第一块（4KB）必须 Pause —— 还没到 64KB
	s0 := mk(false)
	if _, a := s0.Feed(body[:4096], false); a != types.ActionPause {
		t.Fatalf("没有 EarlyCommit 时第一块应当 Pause，得到 %v", a)
	}
	// 有提前提交：model 在第一块里，第一块就该 Continue 并放出内容
	s1 := mk(true)
	o, a := s1.Feed(body[:4096], false)
	if a != types.ActionContinue || len(o) == 0 {
		t.Fatalf("EarlyCommit 应当在第一块就放行，得到 action=%v out=%d", a, len(o))
	}
	// 整份跑完必须与无提前提交的结果逐字节一致
	full := func(s *State) string {
		var out []byte
		for i := 0; i < len(body); i += 4096 {
			j := i + 4096
			if j > len(body) {
				j = len(body)
			}
			o, _ := s.Feed(body[i:j], j == len(body))
			out = append(out, o...)
		}
		return string(out)
	}
	stubHost(t, body)
	if got := full(mk(true)); got != want {
		t.Fatalf("提前提交改变了输出：%d vs %d", len(got), len(want))
	}
	if got := full(mk(false)); got != want {
		t.Fatalf("参照本身不对：%d vs %d", len(got), len(want))
	}
}

// prefixRun feeds body through a PrefixTransform state at the given cuts and returns what went upstream.
func prefixRun(t *testing.T, body []byte, splits []int, eosSeparate bool, early bool) string {
	t.Helper()
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{MapModel: func(m string) string {
		if i := strings.Index(m, "/"); i >= 0 {
			return m[i+1:]
		}
		return m
	}})
	p := &Plan{Tr: tr, Mode: PrefixTransform,
		OnCommit: func(pre streamxform.Prelude, last bool) bool { return pre.ModelSeen }}
	if early {
		p.EarlyCommit = func(pre streamxform.Prelude) bool { return pre.ModelSeen }
	}
	s := New(p)
	var out []byte
	prev := 0
	for i, cut := range splits {
		last := !eosSeparate && i == len(splits)-1
		o, _ := s.Feed(body[prev:cut], last)
		out = append(out, o...)
		prev = cut
	}
	if eosSeparate {
		o, _ := s.Feed(nil, true)
		out = append(out, o...)
	}
	return string(out)
}

// The engine may still hold bytes it has consumed but not written when a chunk ends -- a key it is reading,
// a comma waiting for the next key. Dropping the transformer at that moment and forwarding the next chunk
// verbatim loses them. With early commit the release lands right after the model value, where exactly those
// bytes are pending, so every split of a body with fields after model has to come out complete.
func TestPrefixTransformEarlyCommitIsCompleteUnderAnySplit(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"` + big(3000) + `"}] , "model" : "p/m1" , "stream":true,"max_tokens":16}` + "\n")
	want := strings.Replace(string(body), `"model" : "p/m1"`, `"model" : "m1"`, 1)
	// every two-piece split
	for cut := 1; cut < len(body); cut++ {
		for _, sep := range []bool{true, false} {
			if got := prefixRun(t, body, []int{cut, len(body)}, sep, true); got != want {
				t.Fatalf("cut=%d eos单独=%v: %q", cut, sep, tailOf(got))
			}
		}
	}
	// fixed small chunks, the shape the model-router test uses
	for _, n := range []int{1, 3, 5, 7, 11, 64} {
		var splits []int
		for i := n; i < len(body); i += n {
			splits = append(splits, i)
		}
		splits = append(splits, len(body))
		if got := prefixRun(t, body, splits, false, true); got != want {
			t.Fatalf("chunk=%d: %q", n, tailOf(got))
		}
	}
}

// The same hazard exists without early commit: the chunk that crosses the window can end while the engine
// holds bytes at the root level, when fields follow the large one. Every split around and after the window
// must come out complete.
func TestPrefixTransformWindowCommitIsCompleteWhenFieldsFollowTheWindow(t *testing.T) {
	body := []byte(`{"model":"p/m1","messages":[{"role":"user","content":"` + big(66000) + `"}] , "stream" : true , "max_tokens":16,"user":"u"}` + "\n")
	want := strings.Replace(string(body), `"model":"p/m1"`, `"model":"m1"`, 1)
	for cut := 64<<10 - 64; cut < len(body); cut++ {
		for _, sep := range []bool{true, false} {
			if got := prefixRun(t, body, []int{cut, len(body)}, sep, false); got != want {
				t.Fatalf("cut=%d eos单独=%v: %q", cut, sep, tailOf(got))
			}
		}
	}
	// three pieces: the second cut anywhere in the trailing fields
	for a := 64<<10 - 32; a < 64<<10+32; a++ {
		for b := a + 1; b < len(body); b += 3 {
			if got := prefixRun(t, body, []int{a, b, len(body)}, false, false); got != want {
				t.Fatalf("cuts=%d,%d: %q", a, b, tailOf(got))
			}
		}
	}
}

// A plan whose transformer depends on the model: a probe finds model, Replan picks the real transformer, and
// the driver restarts it from the first byte. The output has to equal the chosen transformer run on its own,
// under any split, with the choice made exactly once; model beyond the window is the plan's own refusal.
func TestReplanRestartsTheChosenTransformerFromTheFirstByte(t *testing.T) {
	strip := func(m string) string {
		if i := strings.Index(m, "/"); i >= 0 {
			return m[i+1:]
		}
		return m
	}
	type run struct {
		out      string
		replans  int
		fallback int
	}
	feed := func(body []byte, splits []int, choose bool) run {
		stubHost(t, body)
		r := run{}
		p := &Plan{
			Tr:       streamxform.NewKeyProbe(streamxform.KeyProbeOptions{Keys: map[string]int{"model": 4096}, ModelKey: "model", Observe: true}),
			Mode:     Transform,
			OnCommit: func(pre streamxform.Prelude, last bool) bool { return pre.ModelSeen },
			Replan: func(pre streamxform.Prelude) (*streamxform.Transformer, string) {
				r.replans++
				if !choose {
					return nil, "no transformer for this model"
				}
				return streamxform.NewOpenAI(streamxform.OpenAIOptions{MapModel: strip}), ""
			},
			Metric: func(n string) {
				if n == "fallback" {
					r.fallback++
				}
			},
		}
		s := New(p)
		var out []byte
		prev := 0
		for i, cut := range splits {
			o, _ := s.Feed(body[prev:cut], i == len(splits)-1)
			out = append(out, o...)
			prev = cut
		}
		r.out = string(out)
		return r
	}
	every := func(n int, stride int) [][]int {
		var all [][]int
		for cut := 1; cut < n; cut += stride {
			all = append(all, []int{cut, n})
		}
		all = append(all, []int{n})
		return all
	}
	small := []byte(`{"messages":[{"role":"user","content":"` + big(3000) + `"}],"model":"p/m1","stream":true}`)
	large := []byte(`{"model":"p/m1","messages":[{"role":"user","content":"` + big(200<<10) + `"}],"max_tokens":16}`)
	late := []byte(`{"messages":[{"role":"user","content":"` + big(70000) + `"}],"model":"p/m1"}`)

	for _, c := range []struct {
		name   string
		body   []byte
		stride int
	}{{"model last, small", small, 1}, {"model first, large", large, 997}} {
		want := strings.Replace(string(c.body), `"p/m1"`, `"m1"`, 1)
		for _, sp := range every(len(c.body), c.stride) {
			r := feed(c.body, sp, true)
			if r.out != want {
				t.Fatalf("%s splits=%v: output differs: %q", c.name, sp, tailOf(r.out))
			}
			if r.replans != 1 || r.fallback != 0 {
				t.Fatalf("%s splits=%v: replans=%d fallback=%d", c.name, sp, r.replans, r.fallback)
			}
		}
	}
	// Replan declines: the buffered path gets the original body, once.
	for _, sp := range every(len(small), 7) {
		r := feed(small, sp, false)
		if r.out != string(small) || r.replans != 1 || r.fallback != 1 {
			t.Fatalf("declined replan splits=%v: replans=%d fallback=%d out=%q", sp, r.replans, r.fallback, tailOf(r.out))
		}
	}
	// model beyond the window. Delivered in chunks that end before it, OnCommit refuses at the window and Replan
	// is never consulted; a split that carries the window crossing and model in one chunk releases nothing
	// before model is known, so that one replans and must produce the transform. Never anything in between.
	wantLate := strings.Replace(string(late), `"p/m1"`, `"m1"`, 1)
	for _, sp := range every(len(late), 331) {
		r := feed(late, sp, true)
		switch {
		case r.replans == 1 && r.fallback == 0 && r.out == wantLate:
		case r.replans == 0 && r.fallback == 1 && r.out == string(late):
		default:
			t.Fatalf("late model splits=%v: replans=%d fallback=%d out=%q", sp, r.replans, r.fallback, tailOf(r.out))
		}
	}
	var fixed []int
	for i := 4096; i < len(late); i += 4096 {
		fixed = append(fixed, i)
	}
	fixed = append(fixed, len(late))
	if r := feed(late, fixed, true); r.replans != 0 || r.fallback != 1 || r.out != string(late) {
		t.Fatalf("late model in 4KB chunks: replans=%d fallback=%d out=%q", r.replans, r.fallback, tailOf(r.out))
	}
}
