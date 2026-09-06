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

// PrefixTransform：提交点前经转换器（model 改写），放行后剩余原样直通。
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
		t.Fatalf("首块应 Pause、末块应 Continue: %v", acts[:2])
	}
	want := strings.Replace(string(body), `"model":"p/m1"`, `"model":"m1"`, 1)
	if string(out) != want {
		t.Fatalf("输出与预期不同 (len %d vs %d)", len(out), len(want))
	}
	if committed != 1 || finished != 1 || !s.raw {
		t.Fatalf("committed=%d finished=%d raw=%v", committed, finished, s.raw)
	}
}

// Observe：输入原样转发；判定不支持只停止观察。
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
			t.Fatalf("观察形态永远 Continue: %v", acts)
		}
	}
	if string(out) != string(body) {
		t.Fatal("观察形态必须原样转发")
	}
	if bailed != 1 {
		t.Fatalf("非法字面量应停止观察一次，bailed=%d", bailed)
	}
}

// 提交点前判定不支持 → 回落到官方路径（收齐后一次交给 Fallback，并读回官方写回的内容）。
func TestFallbackBeforeCommit(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"U"}],"stream":tru}`)
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{})
	var got []byte
	s := New(&Plan{Tr: tr, Mode: Transform, Fallback: func(b []byte) types.Action { got = b; return types.ActionContinue }})
	out, acts := feedAll(s, body, 16)
	if acts[len(acts)-1] != types.ActionContinue || string(got) != string(body) || string(out) != string(body) {
		t.Fatalf("回落应在末块交给官方并返回其 body: acts=%v got=%d out=%d", acts[len(acts)-1:], len(got), len(out))
	}
	if !s.FellBack() {
		t.Fatal("应标记为已回落")
	}
}

// 提交点后判定不支持（Transform 形态）→ Uncoverable。
func TestUncoverableAfterCommit(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"` + big(100<<10) + `"}],"stream":tru}`)
	stubHost(t, body)
	tr := streamxform.NewOpenAI(streamxform.OpenAIOptions{})
	unc := ""
	s := New(&Plan{Tr: tr, Mode: Transform, Uncoverable: func(r string) { unc = r }})
	_, acts := feedAll(s, body, 4096)
	if unc == "" || acts[len(acts)-1] != types.ActionPause {
		t.Fatalf("提交点后非法字面量应触发 Uncoverable: unc=%q last=%v", unc, acts[len(acts)-1])
	}
}
