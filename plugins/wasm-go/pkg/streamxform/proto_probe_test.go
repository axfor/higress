package streamxform

import (
	"encoding/json"
	"math/rand"
	"strings"
	"testing"

	"github.com/tidwall/sjson"
)

func runKeyProbe(t *testing.T, tr *Transformer, in string, chunk int) (string, bool, string) {
	var sb strings.Builder
	for i := 0; i < len(in); i += chunk {
		j := i + chunk
		if j > len(in) {
			j = len(in)
		}
		tr.Write([]byte(in[i:j]))
		sb.Write(tr.Out())
	}
	sb.Write(tr.Finish())
	if u, why := tr.Unsupported(); u {
		return "", false, why
	}
	return sb.String(), true, ""
}

// 改写 model：输出必须与 sjson.SetBytes 逐字节一致（格式保留），且 Prelude 报告 model。
func TestKeyProbeRewriteMatchesSjson(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	bodies := []string{
		`{"model":"p/m","messages":[{"role":"user","content":"U"}]}`,
		`{"messages":[{"role":"user","content":"U"}],"model":"p/m","stream":true}`,
		"{\n  \"model\" : \"p/m\" ,\n  \"messages\" : [ ]\n}\n",
		`{"model":"p/m","model":"q/n"}`,
		`{"a":{"model":"inner"},"model":"p/m"}`,
		`{"model":"plain"}`,
		`{"model":123}`,
		`{"model":"p/m","x":"\u0041\n"}`,
	}
	for i := 0; i < 200; i++ {
		var sb strings.Builder
		sb.WriteString("{")
		n := r.Intn(6) + 1
		pos := r.Intn(n)
		for k := 0; k < n; k++ {
			if k > 0 {
				sb.WriteString(",")
			}
			if k == pos {
				sb.WriteString(`"model":"p/m` + strings.Repeat("x", r.Intn(50)) + `"`)
			} else {
				sb.WriteString(`"k` + string(rune('a'+k)) + `":` + []string{`1`, `"s"`, `[1,{"model":"nested"}]`, `{"model":"nested"}`, `null`, `true`}[r.Intn(6)])
			}
		}
		sb.WriteString("}")
		bodies = append(bodies, sb.String())
	}
	for _, in := range bodies {
		var wantModel string
		gotRaw := ""
		for _, cs := range []int{1, 3, 7, 4096} {
			tr := NewKeyProbe(KeyProbeOptions{Keys: map[string]int{"model": 4096}, ModelKey: "model",
				OnKey: func(t *Transformer, key string, raw []byte) ([]byte, bool) {
					var s string
					if json.Unmarshal(raw, &s) != nil {
						return nil, false
					}
					if i := strings.Index(s, "/"); i >= 0 {
						b, _ := json.Marshal(s[i+1:])
						return b, true
					}
					return nil, false
				}})
			out, ok, why := runKeyProbe(t, tr, in, cs)
			if !ok {
				t.Fatalf("%s chunk=%d: %s", in, cs, why)
			}
			pre := tr.Protocol().(Preluder).Prelude()
			// 期望：gjson 取首个 model；有 "/" 就 sjson 改首个
			var m map[string]any
			_ = json.Unmarshal([]byte(in), &m)
			want := in
			var s string
			if err := json.Unmarshal([]byte(firstModelRaw(in)), &s); err == nil {
				wantModel = s
				if i := strings.Index(s, "/"); i >= 0 {
					b, _ := sjson.SetBytes([]byte(in), "model", s[i+1:])
					want = string(b)
				}
			}
			if out != want {
				t.Fatalf("%s chunk=%d:\n got  %s\n want %s", in, cs, out, want)
			}
			if wantModel != "" && (!pre.ModelSeen || pre.Model != wantModel) {
				t.Fatalf("%s: Prelude.Model=%q seen=%v want %q", in, pre.Model, pre.ModelSeen, wantModel)
			}
			gotRaw = out
		}
		_ = gotRaw
	}
}

// firstModelRaw 取顶层第一个 "model" 的原始值（测试用的朴素实现）。
func firstModelRaw(in string) string {
	tr := NewKeyProbe(KeyProbeOptions{Keys: map[string]int{"model": 4096}, Observe: true})
	var raw string
	tr.Protocol().(*keyProbe).opt.OnKey = func(t *Transformer, key string, r []byte) ([]byte, bool) {
		if raw == "" {
			raw = string(r)
		}
		return nil, false
	}
	tr.Write([]byte(in))
	tr.Finish()
	return raw
}

// 观察形态：不改字节，回调拿到值。
func TestKeyProbeObserve(t *testing.T) {
	in := `{"stream":true,"model":"m","x":[1,2]}`
	var got []string
	tr := NewKeyProbe(KeyProbeOptions{Keys: map[string]int{"model": 64, "stream": 8}, ModelKey: "model", StreamKey: "stream", Observe: true,
		OnKey: func(t *Transformer, key string, raw []byte) ([]byte, bool) {
			got = append(got, key+"="+string(raw))
			return []byte(`"ignored"`), true
		}})
	out, ok, why := runKeyProbe(t, tr, in, 2)
	if !ok || out != in {
		t.Fatalf("观察形态必须原样: ok=%v why=%s out=%s", ok, why, out)
	}
	pre := tr.Protocol().(Preluder).Prelude()
	if !pre.ModelSeen || pre.Model != "m" || !pre.StreamSeen || !pre.Stream || strings.Join(got, " ") != `stream=true model="m"` {
		t.Fatalf("pre=%+v got=%v", pre, got)
	}
}
