// Package guard 是流式请求体转换的集成层（层 3）：把 Transformer 接到 wasm-go 的
// ProcessStreamingRequestBodyWithAction 钩子上，负责提交点、回落与失败的控制流。
//
// 三种形态：
//   - Transform：整份 body 经转换器（ai-proxy 的协议转换）。提交点后判定不支持只能失败。
//   - PrefixTransform：改写都发生在 body 开头（如 model-router 改 model）。提交点前经转换器，
//     放行之后剩余字节原样直通、不再扫描——CPU 接近零，语法错误也与官方一样原样送上游。
//   - Observe：只看不改（如 ai-statistics 数轮次、取 model）。输入原样转发，判定不支持只停止观察。
//
// 提交点（streamxform.CommitBytes）之前返回 ActionPause：原始字节留在宿主缓冲区，请求头扣住；
// 这期间判定不支持可以干净回落到插件的官方全量路径（Fallback）。
package guard

import (
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// Mode 是驱动形态。
type Mode uint8

const (
	Transform Mode = iota
	PrefixTransform
	Observe
)

// Plan 描述一次请求的流式处理。
type Plan struct {
	Tr   *streamxform.Transformer
	Mode Mode
	// Passthrough：官方对 body 一个字节都不动，逐块直接放行，不经转换器（Tr 可为 nil）。
	Passthrough bool
	// OnCommit 在首次放行（请求头下发）之前调用；返回 false 表示要回落——此时还没放行任何字节。
	// 依赖 body 事实的请求头 / 上下文键在这里施加。
	OnCommit func(pre streamxform.Prelude, last bool) bool
	// OnFinish 在整份 body 处理完（最后一块）时调用。
	OnFinish func(pre streamxform.Prelude)
	// Fallback 是插件的官方全量路径：收齐整份 body 后调用。官方 handler 若自行 ReplaceHttpRequestBody，
	// 驱动会把它写回的内容读回并作为本次返回值。返回 ActionPause 表示官方已本地应答或在等异步结果。
	Fallback func(body []byte) types.Action
	// Uncoverable：提交点后判定不支持（Transform 形态）。nil 时本地应答 500。
	Uncoverable func(reason string)
	// Metric 计数：streamed / fallback / uncoverable / observe_bailed。可为 nil。
	Metric func(name string)
	// Log 警告日志。可为 nil。
	Log func(format string, args ...interface{})
}

// State 是一次请求的驱动状态。
type State struct {
	plan     *Plan
	total    int
	fallback bool
	sent     bool
	raw      bool // PrefixTransform：已放行，剩余原样直通
	dead     bool // Observe：已停止观察
}

// New 构造驱动状态。
func New(p *Plan) *State { return &State{plan: p} }

// Prelude 取转换器协议报告的 Prelude（协议不实现 Preluder 时为零值）。
func Prelude(tr *streamxform.Transformer) streamxform.Prelude {
	if tr == nil {
		return streamxform.Prelude{}
	}
	if p, ok := tr.Protocol().(streamxform.Preluder); ok {
		return p.Prelude()
	}
	return streamxform.Prelude{}
}

// 可替换（测试）
var (
	hostRequestBody = proxywasm.GetHttpRequestBody
	hostSendError   = func(reason string) {
		_ = proxywasm.SendHttpResponseWithDetail(500, "streamxform.uncoverable", nil, []byte("streaming transform bailed after commit: "+reason), -1)
	}
)

func (s *State) metric(name string) {
	if s.plan.Metric != nil {
		s.plan.Metric(name)
	}
}

// bailed 记一次判定不支持：总计数 + 按分类的计数（fallback.duplicate_key、uncoverable.limit……）。
// 分类来自引擎的 Error.Code，不匹配文案；非引擎原因（OnCommit 要求回落）用调用方给的名字。
func (s *State) bailed(kind, code string) {
	s.metric(kind)
	s.metric(kind + "." + code)
}

// codeOf 取转换器判定不支持的分类名。
func codeOf(tr *streamxform.Transformer) string {
	if tr == nil || tr.Err() == nil {
		return "none"
	}
	return tr.Err().Code.String()
}

func (s *State) logf(format string, args ...interface{}) {
	if s.plan.Log != nil {
		s.plan.Log(format, args...)
	}
}

// Feed 喂一块请求体，返回要下发的字节与动作。
func (s *State) Feed(chunk []byte, last bool) ([]byte, types.Action) {
	s.total += len(chunk)
	if s.fallback {
		return s.feedFallback(last)
	}
	if s.plan.Passthrough || s.raw {
		if !s.sent {
			if s.plan.OnCommit != nil && !s.plan.OnCommit(Prelude(s.plan.Tr), last) {
				return s.toFallback(last, "OnCommit 要求回落", "oncommit")
			}
			s.sent = true
			s.metric("streamed")
		}
		if last && s.plan.OnFinish != nil {
			s.plan.OnFinish(Prelude(s.plan.Tr))
		}
		return chunk, types.ActionContinue
	}
	tr := s.plan.Tr
	if s.plan.Mode == Observe {
		if !s.dead {
			tr.Write(chunk)
			if last {
				tr.Finish()
			}
			tr.Out() // 观察形态的输出没人要，取走以免积累
			if bad, why := tr.Unsupported(); bad {
				s.dead = true
				code := codeOf(tr)
				s.bailed("observe_bailed", code)
				s.logf("[streamxform] 停止观察 (%s): %s (received=%d)", code, why, s.total)
			}
		}
		if !s.sent {
			if s.plan.OnCommit != nil {
				s.plan.OnCommit(Prelude(tr), last)
			}
			s.sent = true
			s.metric("streamed")
		}
		if last && s.plan.OnFinish != nil {
			s.plan.OnFinish(Prelude(tr))
		}
		return chunk, types.ActionContinue
	}
	tr.Write(chunk)
	var fin []byte
	if last {
		fin = tr.Finish() // Finish 把缓冲里剩余的输出一并取走，下面不能再指望 Out()
	}
	if bad, why := tr.Unsupported(); bad {
		code := codeOf(tr)
		if tr.Committed() && s.sent {
			// 已越过提交点：部分字节已发给上游，无法回落，只能失败。
			s.logf("[streamxform] 提交点之后判定不支持 (%s)，请求失败: %s", code, why)
			s.bailed("uncoverable", code)
			if s.plan.Uncoverable != nil {
				s.plan.Uncoverable(why)
			} else {
				hostSendError(why)
			}
			return nil, types.ActionPause
		}
		return s.toFallback(last, why, code)
	}
	if !tr.Committed() {
		return nil, types.ActionPause // 提交点之前：留在宿主缓冲区，继续攒
	}
	if !s.sent {
		if s.plan.OnCommit != nil && !s.plan.OnCommit(Prelude(tr), last) {
			return s.toFallback(last, "提交点前未满足放行条件", "oncommit")
		}
		s.sent = true
		s.metric("streamed")
	}
	out := append(tr.Out(), fin...)
	if last {
		if s.plan.OnFinish != nil {
			s.plan.OnFinish(Prelude(tr))
		}
	} else if s.plan.Mode == PrefixTransform {
		s.raw = true // 改写只在开头：之后原样直通，不再扫描
		s.plan.Tr = nil
	}
	return out, types.ActionContinue
}

func (s *State) toFallback(last bool, why, code string) ([]byte, types.Action) {
	s.logf("[streamxform] 回落到官方全量路径 (%s): %s (received=%d last=%v)", code, why, s.total, last)
	s.bailed("fallback", code)
	s.fallback = true
	s.plan.Tr = nil
	return s.feedFallback(last)
}

// feedFallback：官方全量路径。攒到末块，从宿主缓冲区取全量 body 交给官方 handler。
func (s *State) feedFallback(last bool) ([]byte, types.Action) {
	if !last {
		return nil, types.ActionPause
	}
	body, err := hostRequestBody(0, s.total)
	if err != nil {
		s.logf("[streamxform] 回落路径读取 body 失败: %v", err)
		return nil, types.ActionContinue
	}
	if s.plan.Fallback == nil {
		return body, types.ActionContinue
	}
	// 官方 handler 会自己 ReplaceHttpRequestBody；这里的返回值会再覆盖一次，所以把它写回的内容读回来返回。
	if s.plan.Fallback(body) == types.ActionPause {
		return nil, types.ActionPause
	}
	nb, err := hostRequestBody(0, 1<<30) // 上限只是读取上限：官方转换后的 body 可能比输入大
	if err != nil {
		return body, types.ActionContinue
	}
	return nb, types.ActionContinue
}

// Sent 报告是否已放行过（请求头已下发）。
func (s *State) Sent() bool { return s.sent }

// FellBack 报告是否已切到官方全量路径。
func (s *State) FellBack() bool { return s.fallback }

// ForceFallback 让本次请求从一开始就走官方全量路径（插件判定不适用流式时）。
func (s *State) ForceFallback() { s.fallback = true; s.plan.Tr = nil }

// NewMetric 返回一个按名字计数的函数：prefix + "." + name。
// 指标只是观测手段：宿主不支持（测试模拟器、或禁用了 metrics 的部署）时静默关闭，绝不能影响请求。
func NewMetric(prefix string) func(name string) {
	counters := map[string]proxywasm.MetricCounter{}
	off := false
	return func(name string) {
		if off {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				off = true
			}
		}()
		key := prefix + "." + name
		c, ok := counters[key]
		if !ok {
			c = proxywasm.DefineCounterMetric(key)
			counters[key] = c
		}
		c.Increment(1)
	}
}
