package streamxform

import (
	"bytes"
	"mime"
	"net/textproto"
	"sort"
	"strings"
)

// Multipart is the streaming form of defaultTransformMultipartRequestBody, the buffered handling of a
// multipart/form-data body on the image edit and variation endpoints of the providers on the default path: the
// model field is read and mapped; when the mapping leaves it alone the body is forwarded untouched, otherwise
// the buffered path re-encodes the whole body with Go's multipart.Writer (rewriteMultipartFormModel) -- same
// boundary, header keys canonicalised and sorted, the model field's content replaced, a model field appended
// when there was none, preamble and epilogue dropped.
//
// Nothing is known until the model field has been read, so the body is held (bounded by the commit window)
// with both renderings kept side by side, and the one the buffered path would have produced goes out once the
// field is complete. The image parts that make these bodies large stream through after that. A model field
// beyond the window, a second model field, a header block over 8KB, a model value over 4KB and a body whose
// lines end in a bare LF are reported as unsupported; the buffered path takes those.
//
// It is not a JSON transformer: the Xform methods that only mean something for JSON (key cache, field tree) are
// no-ops, and the commit window counts input bytes the same way.
type MultipartOptions struct {
	// Boundary is the boundary parameter of the request's Content-Type.
	Boundary string
	// MapModel reproduces getMappedModel: the input unchanged when no mapping matches.
	MapModel func(model string) string
}

type mpState uint8

const (
	mpPreamble     mpState = iota
	mpBoundaryLine         // after "--boundary": padding, then CRLF (a part follows) or "--" (the end)
	mpHeaders
	mpBody
	mpEpilogue
)

type mpMode uint8

const (
	mpUndecided  mpMode = iota
	mpVerbatim          // the mapping changes nothing: bytes pass through
	mpNormalized        // the buffered re-encoding
	mpInPlace           // committed before the model field: verbatim, the model field's content replaced in place
)

const (
	mpHeaderCap = 8 << 10
	mpModelCap  = 4 << 10
)

// Multipart implements Xform for a multipart/form-data body.
type Multipart struct {
	opt   MultipartOptions
	delim []byte // "\r\n--boundary"
	open  []byte // "--boundary"

	buf     []byte // unconsumed input
	scanned int
	state   mpState
	mode    mpMode

	raw  []byte // the verbatim rendering, while undecided
	norm []byte // the normalized rendering, while undecided
	out  []byte // released output not yet handed over

	firstPart bool
	partName  string
	isModel   bool
	modelBuf  []byte
	model     string
	modelSeen bool
	mapped    string
	rootDone  bool
	finished  bool

	commitBytes int
	committed   bool
	sink        func([]byte)
	err         *Error
	pre         mpPrelude
}

type mpPrelude struct{ p *Multipart }

func (m mpPrelude) Prelude() Prelude { return Prelude{Model: m.p.model, ModelSeen: m.p.modelSeen} }

// NewMultipart builds the transformer.
func NewMultipart(opt MultipartOptions) *Multipart {
	if opt.MapModel == nil {
		opt.MapModel = func(m string) string { return m }
	}
	m := &Multipart{opt: opt, commitBytes: 64 << 10, firstPart: true}
	m.delim = []byte("\r\n--" + opt.Boundary)
	m.open = []byte("--" + opt.Boundary)
	m.pre = mpPrelude{m}
	return m
}

var _ Xform = (*Multipart)(nil)

func (m *Multipart) Protocol() Protocol      { return m.pre }
func (m *Multipart) SetKeyCache(*KeyCache)   {}
func (m *Multipart) SetFieldTree(*FieldTree) {}
func (m *Multipart) SetCommitBytes(n int)    { m.commitBytes = n }
func (m *Multipart) SetSink(f func([]byte))  { m.sink = f }
func (m *Multipart) Committed() bool         { return m.committed }
func (m *Multipart) RootDone() bool          { return m.rootDone }
func (m *Multipart) Aligned() bool           { return m.mode != mpUndecided && len(m.buf) == 0 && !m.isModel }
func (m *Multipart) Err() *Error             { return m.err }
func (m *Multipart) Unsupported() (bool, string) {
	if m.err == nil {
		return false, ""
	}
	return true, m.err.Error()
}

func (mp mpPrelude) OnKey(*Transformer) Action                         { return Pass() }
func (mp mpPrelude) OnElem(*Transformer) Action                        { return Pass() }
func (mp mpPrelude) OnStart(*Transformer, ValueKind) Action            { return Pass() }
func (mp mpPrelude) OnValue(*Transformer, []byte)                      {}
func (mp mpPrelude) OnPrefix(*Transformer, []byte, bool) (Action, int) { return Pass(), 0 }
func (mp mpPrelude) OnLeave(*Transformer)                              {}
func (mp mpPrelude) Tail(*Transformer)                                 {}

func (m *Multipart) bail(code Code, msg string) {
	if m.err == nil {
		m.err = &Error{Code: code, Msg: msg, Offset: int64(m.scanned)}
	}
}

// emit sends bytes to the chosen rendering(s).
func (m *Multipart) emitRaw(b []byte) {
	switch m.mode {
	case mpUndecided:
		m.raw = append(m.raw, b...)
	case mpVerbatim, mpInPlace:
		m.release(b)
	}
}

func (m *Multipart) emitNorm(b []byte) {
	switch m.mode {
	case mpUndecided:
		m.norm = append(m.norm, b...)
	case mpNormalized:
		m.release(b)
	}
}

// release hands output over: held until the commit point, then straight to the sink or Out.
func (m *Multipart) release(b []byte) {
	if len(b) == 0 {
		return
	}
	if m.committed && m.sink != nil {
		m.sink(b)
		return
	}
	m.out = append(m.out, b...)
}

// decide picks the rendering once the model is known (or the window is full).
func (m *Multipart) decide(mode mpMode) {
	if m.mode != mpUndecided {
		return
	}
	m.mode = mode
	switch mode {
	case mpNormalized:
		m.release(m.norm)
	default:
		m.release(m.raw)
	}
	m.raw, m.norm = nil, nil
}

// CommitNow releases the output accumulated so far and every byte after it.
func (m *Multipart) CommitNow() {
	if m.committed {
		return
	}
	if m.mode == mpUndecided {
		m.decide(mpInPlace)
	}
	m.committed = true
	if m.sink != nil && len(m.out) > 0 {
		m.sink(m.out)
		m.out = nil
	}
}

// Out returns the output released so far when no sink is set.
func (m *Multipart) Out() []byte {
	if m.err != nil || !m.committed || m.sink != nil {
		return nil
	}
	b := m.out
	m.out = nil
	return b
}

func (m *Multipart) Write(p []byte) {
	if m.err != nil || m.finished {
		return
	}
	m.buf = append(m.buf, p...)
	m.scanned += len(p)
	m.scan()
	if m.err != nil {
		return
	}
	if !m.committed && m.scanned >= m.commitBytes {
		m.CommitNow()
	}
}

func (m *Multipart) Finish() []byte {
	if m.err != nil || m.finished {
		return nil
	}
	m.finished = true
	if !m.rootDone {
		m.bail(ErrIncomplete, "multipart body ends before the closing boundary")
		return nil
	}
	// the epilogue: verbatim in the pass-through renderings, dropped by the re-encoding
	m.emitRaw(m.buf)
	m.buf = nil
	if m.mode == mpUndecided {
		// no model field and the window never filled: the buffered path leaves the body alone unless mapping
		// the empty model yields a name, which it then appends as a model field in the re-encoding
		m.mapped = m.opt.MapModel("")
		if m.mapped == "" {
			m.decide(mpVerbatim)
		} else {
			m.decide(mpNormalized)
		}
	}
	if m.mode == mpNormalized {
		if !m.modelSeen && m.mapped != "" { // rewriteMultipartFormModel: WriteField("model", model) before Close
			m.release(m.normalizedPart(textproto.MIMEHeader{"Content-Disposition": {`form-data; name="model"`}}))
			m.release([]byte(m.mapped))
		}
		m.release([]byte("\r\n--" + m.opt.Boundary + "--\r\n"))
	}
	// in place without a model field (committed before the end, RequireModelBeforeCommit off): the body went out
	// as it was; the field the buffered path would have appended is not added
	m.committed = true
	if m.sink != nil {
		if len(m.out) > 0 {
			m.sink(m.out)
			m.out = nil
		}
		return nil
	}
	return m.Out()
}

// ---- scanning ----

func (m *Multipart) scan() {
	for m.err == nil {
		switch m.state {
		case mpPreamble:
			if !m.scanPreamble() {
				return
			}
		case mpBoundaryLine:
			if !m.scanBoundaryLine() {
				return
			}
		case mpHeaders:
			if !m.scanHeaders() {
				return
			}
		case mpBody:
			if !m.scanBody() {
				return
			}
		case mpEpilogue:
			// everything after the closing boundary stays in buf until Finish
			return
		}
	}
}

// scanPreamble finds the first boundary line: at the start of the body or after a line break.
func (m *Multipart) scanPreamble() bool {
	if bytes.HasPrefix(m.buf, m.open) {
		m.emitRaw(m.buf[:len(m.open)])
		m.buf = m.buf[len(m.open):]
		m.state = mpBoundaryLine
		return true
	}
	if len(m.buf) < len(m.open) {
		return false
	}
	i := bytes.Index(m.buf, m.delim)
	if i < 0 {
		// a preamble: keep the last delim-1 bytes in case the delimiter is split across chunks
		keep := len(m.delim) - 1
		if len(m.buf) > keep {
			m.emitRaw(m.buf[:len(m.buf)-keep])
			m.buf = m.buf[len(m.buf)-keep:]
		}
		return false
	}
	if bytes.Contains(m.buf[:i], []byte("\n")) && !bytes.Contains(m.buf[:i], []byte("\r\n")) {
		m.bail(ErrUnsupported, "multipart lines end in a bare LF")
		return false
	}
	m.emitRaw(m.buf[:i+len(m.delim)]) // the preamble is dropped by the re-encoding
	m.buf = m.buf[i+len(m.delim):]
	m.state = mpBoundaryLine
	return true
}

// scanBoundaryLine reads what follows "--boundary": transport padding, then CRLF for a part or "--" for the end.
func (m *Multipart) scanBoundaryLine() bool {
	i := 0
	for i < len(m.buf) && (m.buf[i] == ' ' || m.buf[i] == '\t') {
		i++
	}
	if len(m.buf) < i+2 {
		return false
	}
	if m.buf[i] == '-' && m.buf[i+1] == '-' {
		m.emitRaw(m.buf[:i+2])
		m.buf = m.buf[i+2:]
		m.rootDone = true
		m.state = mpEpilogue
		return true
	}
	if m.buf[i] == '\r' && m.buf[i+1] == '\n' {
		m.emitRaw(m.buf[:i+2])
		m.buf = m.buf[i+2:]
		m.state = mpHeaders
		return true
	}
	if m.buf[i] == '\n' {
		m.bail(ErrUnsupported, "multipart lines end in a bare LF")
		return false
	}
	m.bail(ErrUnsupported, "unexpected bytes after a multipart boundary")
	return false
}

// scanHeaders reads a part's header block up to the empty line.
func (m *Multipart) scanHeaders() bool {
	var end, sep int // the header block, and the bytes that close it (a bare CRLF when there are no headers)
	if bytes.HasPrefix(m.buf, []byte("\r\n")) {
		end, sep = 0, 2
	} else {
		if len(m.buf) < 2 {
			return false
		}
		end = bytes.Index(m.buf, []byte("\r\n\r\n"))
		sep = 4
		if end < 0 {
			if len(m.buf) > mpHeaderCap {
				m.bail(ErrUnsupported, "multipart part headers exceed 8KB")
			}
			return false
		}
	}
	if end > mpHeaderCap {
		m.bail(ErrUnsupported, "multipart part headers exceed 8KB")
		return false
	}
	block := m.buf[:end]
	hdr := textproto.MIMEHeader{}
	if end > 0 {
		for _, line := range bytes.Split(block, []byte("\r\n")) {
			if len(line) == 0 {
				continue
			}
			if line[0] == ' ' || line[0] == '\t' {
				m.bail(ErrUnsupported, "multipart header continuation line")
				return false
			}
			if bytes.Contains(line, []byte("\n")) {
				m.bail(ErrUnsupported, "multipart lines end in a bare LF")
				return false
			}
			colon := bytes.IndexByte(line, ':')
			if colon < 0 {
				m.bail(ErrUnsupported, "malformed multipart header line, the buffered reader fails")
				return false
			}
			key := textproto.CanonicalMIMEHeaderKey(strings.TrimRight(string(line[:colon]), " \t"))
			if key == "" || strings.ContainsAny(key, " \t") {
				m.bail(ErrUnsupported, "malformed multipart header line, the buffered reader fails")
				return false
			}
			val := strings.TrimSpace(string(line[colon+1:]))
			hdr[key] = append(hdr[key], val)
		}
	}
	m.partName = ""
	if cd := hdr.Get("Content-Disposition"); cd != "" {
		if typ, params, err := mime.ParseMediaType(cd); err == nil && typ == "form-data" {
			m.partName = params["name"]
		}
	}
	m.isModel = m.partName == "model"
	if m.isModel && m.modelSeen {
		m.bail(ErrUnsupported, "a second model field, the buffered path reads the last and rewrites all")
		return false
	}
	m.emitRaw(m.buf[:end+sep])
	m.emitNorm(m.normalizedPart(hdr))
	m.buf = m.buf[end+sep:]
	m.firstPart = false
	m.modelBuf = m.modelBuf[:0]
	m.state = mpBody
	return true
}

// normalizedPart is what multipart.Writer.CreatePart writes for a part: the boundary line, the headers sorted
// by key, the empty line.
func (m *Multipart) normalizedPart(hdr textproto.MIMEHeader) []byte {
	var b []byte
	if m.firstPart {
		b = append(b, "--"+m.opt.Boundary+"\r\n"...)
	} else {
		b = append(b, "\r\n--"+m.opt.Boundary+"\r\n"...)
	}
	keys := make([]string, 0, len(hdr))
	for k := range hdr {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		for _, v := range hdr[k] {
			b = append(b, k+": "+v+"\r\n"...)
		}
	}
	return append(b, "\r\n"...)
}

// scanBody streams a part's content up to the next delimiter.
func (m *Multipart) scanBody() bool {
	i := bytes.Index(m.buf, m.delim)
	if i < 0 {
		keep := len(m.delim) - 1
		if len(m.buf) <= keep {
			return false
		}
		m.content(m.buf[:len(m.buf)-keep], false)
		m.buf = m.buf[len(m.buf)-keep:]
		return false
	}
	m.content(m.buf[:i], true)
	if m.err != nil {
		return false
	}
	m.emitRaw(m.buf[i : i+len(m.delim)])
	m.buf = m.buf[i+len(m.delim):]
	m.state = mpBoundaryLine
	return true
}

// content handles part content: streamed for every field but model, which is held whole and decides the mode.
func (m *Multipart) content(b []byte, complete bool) {
	if !m.isModel {
		m.emitRaw(b)
		m.emitNorm(b)
		return
	}
	m.modelBuf = append(m.modelBuf, b...)
	if len(m.modelBuf) > mpModelCap {
		m.bail(ErrUnsupported, "multipart model field exceeds 4KB")
		return
	}
	if !complete {
		return
	}
	m.isModel = false
	m.model = strings.TrimSpace(string(m.modelBuf))
	m.modelSeen = true
	m.mapped = m.opt.MapModel(m.model)
	unchanged := m.mapped == m.model || (m.mapped == "" && m.model == "")
	switch m.mode {
	case mpUndecided:
		if unchanged {
			m.emitRaw(m.modelBuf)
			m.decide(mpVerbatim)
		} else {
			m.emitNorm([]byte(m.mapped))
			m.decide(mpNormalized)
		}
	case mpInPlace:
		if unchanged {
			m.release(m.modelBuf)
		} else {
			m.release([]byte(m.mapped))
		}
	case mpVerbatim:
		m.release(m.modelBuf)
	case mpNormalized:
		m.release([]byte(m.mapped))
	}
}
