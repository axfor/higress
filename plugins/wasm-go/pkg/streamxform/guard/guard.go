// Package guard is the integration layer (layer 3) of streaming request body transformation: it connects a
// Transformer to the ProcessStreamingRequestBodyWithAction hook of wasm-go and owns the control flow of the commit point, fallback and failure.
//
// Three modes:
//   - Transform: the whole body goes through the transformer (ai-proxy protocol conversion). Bailing after the commit point can only fail.
//   - PrefixTransform: every rewrite happens at the start of the body (model-router rewriting model). The transformer sees the
//     bytes before the commit point; after release the rest is forwarded verbatim without scanning: near-zero CPU, and syntax errors reach the upstream unchanged, as on the buffered path.
//   - Observe: look but do not touch (ai-statistics counting turns, reading model). Input is forwarded as is; bailing only stops observing.
//
// Before the commit point (streamxform.CommitBytes) it returns ActionPause: the raw bytes stay in the host buffer and the request
// headers are held; bailing during that window falls back cleanly to the plugin's buffered path (Fallback).
package guard

import (
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// Mode is the driving mode.
type Mode uint8

const (
	Transform Mode = iota
	PrefixTransform
	Observe
)

// Plan describes the streaming handling of one request.
type Plan struct {
	Tr   *streamxform.Transformer
	Mode Mode
	// Passthrough: the buffered path does not touch the body at all; chunks are released directly without the transformer (Tr may be nil).
	Passthrough bool
	// OnCommit is called before the first release (headers sent); returning false asks for a fallback while nothing has been released yet.
	// Request headers and context keys that depend on body facts are applied here.
	OnCommit func(pre streamxform.Prelude, last bool) bool
	// OnFinish is called when the whole body has been handled (last chunk).
	OnFinish func(pre streamxform.Prelude)
	// Fallback is the plugin's buffered path, called once the whole body is collected. If the buffered handler calls
	// ReplaceHttpRequestBody itself, the driver reads that content back and returns it. ActionPause means the handler answered locally or waits for an async result.
	Fallback func(body []byte) types.Action
	// Uncoverable: bailed after the commit point (Transform mode). nil answers 500 locally.
	Uncoverable func(reason string)
	// Metric counts streamed / fallback / uncoverable / observe_bailed. May be nil.
	Metric func(name string)
	// Log is the warning logger. May be nil.
	Log func(format string, args ...interface{})
}

// State is the driving state of one request.
type State struct {
	plan     *Plan
	total    int
	fallback bool
	sent     bool
	raw      bool   // PrefixTransform: released, the rest is forwarded verbatim
	dead     bool   // Observe: observation stopped
	out      []byte // output of the current Feed when the transformer had to build one
	pre      streamxform.Prelude
	preKept  bool // pre holds the prelude captured when the transformer was dropped
}

// New builds the driving state.
//
// The output buffer is per request, not per VM: one Envoy worker interleaves many streams, and a transformer
// accumulates output across chunks until the commit point, so a buffer shared between streams would mix them.
// Within a request it is reused from chunk to chunk, and a chunk the transformer forwarded untouched costs no
// buffer at all — Feed hands back the caller's own bytes and the driver skips the host replace entirely.
// keyCache is shared by every transformer on this wasm VM. One VM serves one Envoy worker, whose streams
// interleave but never run concurrently, so a single cache is safe; the keys of one request are the keys of
// the next, so after the first request no key dispatch allocates.
var keyCache = streamxform.NewKeyCache()

// outBufs is a free list of output buffers, one per in-flight transforming stream.
//
// A stream cannot share a buffer with another -- output accumulates across chunks until the commit point, and
// one Envoy worker interleaves its streams -- but it can inherit the buffer a finished stream no longer needs.
// Without this every request grew its own buffer from nothing, doubling up to a commit window and leaving the
// old copies as garbage: about four allocations and a quarter of a megabyte per request, and most of what the
// GC watchdog was collecting. The peak is unchanged (each in-flight stream still holds one buffer); the churn
// is gone. The list is per VM, so it needs no locking.
var outBufs [][]byte

func takeOutBuf() []byte {
	if n := len(outBufs); n > 0 {
		b := outBufs[n-1]
		outBufs = outBufs[:n-1]
		return b[:0]
	}
	return make([]byte, 0, streamxform.OutBufferSize)
}

func putOutBuf(b []byte) {
	if cap(b) == 0 || len(outBufs) >= outBufsMax {
		return
	}
	outBufs = append(outBufs, b[:0])
}

// outBufsMax bounds the free list: past it a returned buffer is dropped for the GC. It is sized above the
// admission limit so that steady-state traffic never allocates, and stops a burst from pinning memory forever.
const outBufsMax = 1024

func New(p *Plan) *State {
	s := &State{plan: p}
	if p.Tr != nil {
		p.Tr.SetKeyCache(keyCache)
	}
	if p.Tr != nil && p.Mode != Observe && !p.Passthrough {
		s.out = takeOutBuf()
		p.Tr.SetOutBuffer(s.out) // the engine writes into it directly; Out hands it back without copying
	}
	return s
}

// Release returns the stream's output buffer to the free list. Feed calls it on the last chunk; a plugin that
// learns of an aborted stream through the host (ProcessStreamDone) should call it too, so an upload the client
// dropped mid-way does not keep its buffer out of circulation. Idempotent.
func (s *State) Release() {
	if s.out != nil {
		putOutBuf(s.out)
		s.out = nil
	}
}

// prelude returns the protocol's prelude, from the transformer while it is alive and from the copy kept when
// it was dropped. Reading it off a dropped transformer would silently yield a zero value.
func (s *State) prelude() streamxform.Prelude {
	if s.preKept {
		return s.pre
	}
	return Prelude(s.plan.Tr)
}

// Prelude returns the Prelude reported by the transformer's protocol (zero value when it does not implement Preluder).
func Prelude(tr *streamxform.Transformer) streamxform.Prelude {
	if tr == nil {
		return streamxform.Prelude{}
	}
	if p, ok := tr.Protocol().(streamxform.Preluder); ok {
		return p.Prelude()
	}
	return streamxform.Prelude{}
}

// replaceable (tests)
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

// bailed records one bail: the total counter plus a per-code counter (fallback.duplicate_key, uncoverable.limit, ...).
// The code comes from the engine's Error.Code, never from the message text; non-engine reasons (OnCommit asking to fall back) use the name the caller gives.
func (s *State) bailed(kind, code string) {
	s.metric(kind)
	s.metric(kind + "." + code)
}

// codeOf returns the code name of the transformer's bail.
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

// Feed takes one chunk of the request body and returns the bytes to forward and the action.
func (s *State) Feed(chunk []byte, last bool) ([]byte, types.Action) {
	s.total += len(chunk)
	if s.fallback {
		return s.feedFallback(last)
	}
	if s.plan.Passthrough || s.raw {
		if !s.sent {
			if s.plan.OnCommit != nil && !s.plan.OnCommit(s.prelude(), last) {
				return s.toFallback(last, "OnCommit asked for a fallback", "oncommit")
			}
			s.sent = true
			s.metric("streamed")
		}
		if last && s.plan.OnFinish != nil {
			s.plan.OnFinish(s.prelude())
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
			tr.Out() // nobody wants the output in Observe mode; take it so it does not accumulate
			if bad, why := tr.Unsupported(); bad {
				s.dead = true
				code := codeOf(tr)
				s.bailed("observe_bailed", code)
				s.logf("[streamxform] observation stopped (%s): %s (received=%d)", code, why, s.total)
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
	var out []byte
	if last {
		// Finish appends the tail to the buffer and returns all of it; taking Out first would drain the
		// buffer and then alias it with the tail, since both are the same caller-owned array.
		out = tr.Finish()
	} else {
		out = tr.Out() // nil before the commit point; past it, the caller-owned buffer, valid until the next Write
	}
	if bad, why := tr.Unsupported(); bad {
		code := codeOf(tr)
		if tr.Committed() && s.sent {
			// Past the commit point: some bytes already went upstream, no fallback is possible, only failure.
			s.logf("[streamxform] bailed after the commit point (%s), failing the request: %s", code, why)
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
		return nil, types.ActionPause // before the commit point: stay in the host buffer and keep collecting
	}
	if !s.sent {
		if s.plan.OnCommit != nil && !s.plan.OnCommit(Prelude(tr), last) {
			return s.toFallback(last, "release condition not met before the commit point", "oncommit")
		}
		s.sent = true
		s.metric("streamed")
	}
	if last {
		if s.plan.OnFinish != nil {
			s.plan.OnFinish(Prelude(tr))
		}
		// Released as Feed returns, before the caller reads out -- which is safe only because nothing can
		// reuse the buffer in between: takeOutBuf runs in New, on another stream's first chunk, and on a
		// single-threaded VM no other callback runs until this one has returned and the host has copied out.
		defer s.Release()
	} else if s.plan.Mode == PrefixTransform && tr.RootDone() {
		// The chunk that released also carried the end of the root. The engine still holds the closing token
		// and any trailing whitespace for Finish, so the transformer has to stay until the end of the stream.
		// Counted because this branch is easy to get wrong and its failure mode is a silently truncated body.
		s.metric("prefix_kept_for_finish")
	} else if s.plan.Mode == PrefixTransform {
		// Rewrites only happen at the start, so once the prefix is released the rest can be forwarded
		// verbatim and the transformer dropped -- but only while the engine has no bytes left to give,
		// which is what the RootDone check above establishes.
		// The prelude is what the protocol learnt while scanning; keep it, because dropping the transformer
		// would otherwise hand OnFinish a zero value on the last chunk.
		s.pre, s.preKept = Prelude(tr), true
		s.raw = true
		s.plan.Tr = nil
	}
	return out, types.ActionContinue
}

func (s *State) toFallback(last bool, why, code string) ([]byte, types.Action) {
	s.logf("[streamxform] falling back to the buffered path (%s): %s (received=%d last=%v)", code, why, s.total, last)
	s.bailed("fallback", code)
	s.fallback = true
	s.plan.Tr = nil
	return s.feedFallback(last)
}

// feedFallback is the buffered path: collect until the last chunk, then take the whole body from the host buffer and hand it to the buffered handler.
func (s *State) feedFallback(last bool) ([]byte, types.Action) {
	if !last {
		return nil, types.ActionPause
	}
	body, err := hostRequestBody(0, s.total)
	if err != nil {
		s.logf("[streamxform] fallback path failed to read the body: %v", err)
		return nil, types.ActionContinue
	}
	if s.plan.Fallback == nil {
		return body, types.ActionContinue
	}
	// The buffered handler calls ReplaceHttpRequestBody itself and our return value would overwrite it, so read back what it wrote and return that.
	if s.plan.Fallback(body) == types.ActionPause {
		return nil, types.ActionPause
	}
	nb, err := hostRequestBody(0, 1<<30) // only a read limit: the converted body may be larger than the input
	if err != nil {
		return body, types.ActionContinue
	}
	return nb, types.ActionContinue
}

// Sent reports whether anything has been released (request headers sent).
func (s *State) Sent() bool { return s.sent }

// FellBack reports whether the request switched to the buffered path.
func (s *State) FellBack() bool { return s.fallback }

// ForceFallback makes this request take the buffered path from the start (when the plugin decides streaming does not apply).
func (s *State) ForceFallback() { s.fallback = true; s.plan.Tr = nil }

// NewMetric returns a counter function keyed by name: prefix + "." + name.
// Metrics are observation only: when the host does not support them (the test emulator, or a deployment with metrics disabled) they switch off silently and never affect the request.
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
