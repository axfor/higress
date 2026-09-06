package streamxform

import "encoding/json"

// KeyProbe 是"只关心顶层若干 key"的透传型协议：命中的 key 整个值攒下来交给回调（可原位替换），
// 其余字节原样直通（空白、顺序、转义一个不改，与 sjson 原地改写的效果一致）。
//
// 供 model-router / ai-statistics 这类只需要 body 开头几个字段的插件使用；
// 同一个 key 出现多次时只有第一次触发回调，后面的原样保留——与 gjson 取首个、sjson 改首个一致。
type KeyProbeOptions struct {
	// Keys：要捕获的顶层 key 及其字节上限（超出上限判定不支持 → 回落）。
	Keys map[string]int
	// OnKey：值到齐时回调。raw 是值的原始 JSON 文本；返回 (replacement, true) 则用 replacement 原位替换，
	// 否则原样写回。回调内可用 t.Bail 判定不支持。
	OnKey func(t *Transformer, key string, raw []byte) (replacement []byte, replace bool)
	// ModelKey：把这个 key 的字符串值报告为 Prelude.Model（可为空）。
	ModelKey string
	// StreamKey：把这个 key 的布尔值报告为 Prelude.Stream（可为空）。
	StreamKey string
	// Observe：只观察不改写。key 与值原样直通，回调只拿到副本（此时 OnKey 的返回值被忽略）。
	Observe bool
}

type keyProbe struct {
	BaseProtocol
	opt  KeyProbeOptions
	seen map[string]bool
	pre  Prelude
}

// NewKeyProbe 构造顶层 key 探针。
func NewKeyProbe(opt KeyProbeOptions) *Transformer {
	p := &keyProbe{opt: opt, seen: map[string]bool{}}
	return NewTransformer(p)
}

func (p *keyProbe) Prelude() Prelude { return p.pre }

func (p *keyProbe) OnKey(t *Transformer) Action {
	if t.Depth() != 1 {
		return Pass()
	}
	k := t.Last()
	cap, want := p.opt.Keys[k]
	if !want || p.seen[k] {
		return Pass()
	}
	p.seen[k] = true
	if p.opt.Observe {
		return Observe(cap)
	}
	return Capture(cap)
}

func (p *keyProbe) OnValue(t *Transformer, raw []byte) {
	k := t.Last()
	if k == p.opt.ModelKey {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			p.pre.Model, p.pre.ModelSeen = s, true
		}
	}
	if k == p.opt.StreamKey {
		var b bool
		if json.Unmarshal(raw, &b) == nil {
			p.pre.Stream, p.pre.StreamSeen = b, true
		}
	}
	if p.opt.Observe {
		if p.opt.OnKey != nil {
			p.opt.OnKey(t, k, raw)
		}
		return
	}
	out := raw
	if p.opt.OnKey != nil {
		if r, ok := p.opt.OnKey(t, k, raw); ok {
			out = r
		}
	}
	w := t.W()
	w.KeyRaw(t.KeyRaw())
	w.Raw(out)
}
