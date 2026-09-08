package streamxform

// Xform is what the guard drives: a single Transformer, or a Pipeline of two.
type Xform interface {
	Write(p []byte)
	Finish() []byte
	Out() []byte
	SetSink(func([]byte))
	Committed() bool
	CommitNow()
	Unsupported() (bool, string)
	Err() *Error
	Protocol() Protocol
	RootDone() bool
	Aligned() bool
	SetKeyCache(*KeyCache)
	SetCommitBytes(int)
	SetFieldTree(*FieldTree)
}

var _ Xform = (*Transformer)(nil)
var _ Xform = (*Pipeline)(nil)

// Pipeline runs two transformers in series: the first releases its output as soon as it is produced, straight
// into the second, and the second is the one the caller sees -- its commit point, its prelude, its output. A
// document the first cannot handle is reported as unsupported before anything has left the second, so the
// caller's retreat works as with a single transformer.
//
// This is what a buffered path that transforms twice needs (Claude protocol input converted to OpenAI, then the
// provider's own conversion): each stage stays the protocol it is, and the pipeline is the composition.
type Pipeline struct {
	first  *Transformer
	second Xform
}

// NewPipeline composes first then second. The first stage's own commit window is set to one byte: it holds
// nothing back, the second stage's window is the one that counts.
func NewPipeline(first *Transformer, second Xform) *Pipeline {
	p := &Pipeline{first: first, second: second}
	first.SetCommitBytes(1)
	first.SetSink(func(b []byte) { second.Write(b) })
	return p
}

// Stages returns the two stages, for callers that need to configure one of them. The second may itself be a Pipeline.
func (p *Pipeline) Stages() (first *Transformer, second Xform) { return p.first, p.second }

func (p *Pipeline) Write(b []byte) {
	if bad, _ := p.first.Unsupported(); bad {
		return
	}
	p.first.Write(b)
}

func (p *Pipeline) Finish() []byte {
	if bad, _ := p.first.Unsupported(); !bad {
		p.first.Finish() // the remainder reaches the second stage through the sink
	}
	return p.second.Finish()
}

func (p *Pipeline) Out() []byte               { return p.second.Out() }
func (p *Pipeline) SetSink(f func([]byte))    { p.second.SetSink(f) }
func (p *Pipeline) Committed() bool           { return p.second.Committed() }
func (p *Pipeline) CommitNow()                { p.second.CommitNow() }
func (p *Pipeline) Protocol() Protocol        { return p.second.Protocol() }
func (p *Pipeline) RootDone() bool            { return p.second.RootDone() }
func (p *Pipeline) Aligned() bool             { return p.first.RootDone() && p.second.Aligned() }
func (p *Pipeline) SetCommitBytes(n int)      { p.second.SetCommitBytes(n) }
func (p *Pipeline) SetFieldTree(t *FieldTree) { p.second.SetFieldTree(t) }
func (p *Pipeline) SetKeyCache(c *KeyCache) {
	p.first.SetKeyCache(c)
	p.second.SetKeyCache(c)
}

// Unsupported reports the first stage's verdict first: it sees the input as it came.
func (p *Pipeline) Unsupported() (bool, string) {
	if bad, why := p.first.Unsupported(); bad {
		return true, why
	}
	return p.second.Unsupported()
}

func (p *Pipeline) Err() *Error {
	if e := p.first.Err(); e != nil {
		return e
	}
	return p.second.Err()
}
