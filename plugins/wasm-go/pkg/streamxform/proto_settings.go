package streamxform

import "bytes"

// The customSettings stage: the streaming form of ReplaceByCustomSettings, which the buffered path runs on the
// body before any provider handler sees it. Each setting names a dotted path and a raw JSON value; with overwrite
// the value replaces whatever the request has there, without it the value is only added when the path is absent.
// The stage goes first in a pipeline, so the provider's own transformer reads the body as the buffered handler
// would have.
//
// Everything not on a setting's path passes through. A container on the way to a setting is entered and only its
// relevant keys are looked at. Overwrite values are written as soon as their container starts (the request's own
// value, wherever it is, is skipped), so a setting on model or stream reaches the next stage's prelude before
// anything else; add-if-absent values wait for the container to close, when absence is known. Missing intermediate
// objects are created there too, as sjson creates them.
//
// The settings are normalised once: a later setting on a path at or below an earlier overwrite is folded into
// that value; a later overwrite drops the earlier settings below it; an add-if-absent setting whose path an
// earlier setting is bound to create is dropped as the no-op it is. After that, the only nesting left between
// entries is an add-if-absent ancestor whose value already carries its descendants, for the case where the
// ancestor is absent from the request.
//
// sjson's other behaviours are reproduced where they are cheap and reported as unsupported where they are not:
// a scalar or null on the way to a deeper setting becomes an object (as sjson does), an array there is an error
// on the buffered path and unsupported here, a duplicate key is unsupported (sjson touches the first one).

// Setting is one customSettings entry: the path split on dots, the raw JSON value, and whether it overwrites.
type Setting struct {
	Path      []string
	Value     []byte
	Overwrite bool
}

type setEntry struct {
	path      []string
	value     []byte
	overwrite bool
}

type setNode struct {
	key      string
	entry    *setEntry
	children []*setNode
	byKey    map[string]*setNode
	// per request: the node is met at most once (duplicate keys are unsupported)
	state   uint8 // setAbsent until the key is met
	started bool  // the overwrite entries below it have been written
}

const (
	setAbsent   uint8 = iota
	setPresent        // an object that was entered, or a value that satisfies "exists"
	setReplaced       // a scalar / null where deeper settings need an object, or a value skipped for an overwrite
)

type settingsProto struct {
	root   setNode
	stack  []*setNode
	silent int    // inside a value an overwrite replaces: nothing is written, only the dropped settings' paths are checked
	bad    string // a setting the buffered path fails on: reported at the first callback
}

// NewSettings builds the customSettings stage.
func NewSettings(settings []Setting) *Transformer {
	p := &settingsProto{}
	p.root.byKey = map[string]*setNode{}
	entries, probes, bad := normalizeSettings(settings)
	p.bad = bad
	for _, e := range entries {
		p.add(e)
	}
	for _, path := range probes {
		p.add(&setEntry{path: path}).entry = nil // the path only has to be walked
	}
	p.stack = []*setNode{&p.root}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

// foldValue applies one setting to a raw value the way sjson would: inside an object; over a fresh object when
// the value is a scalar, a string or null (sjson replaces those); not at all over an array (sjson fails).
// The stage itself does the work: a nested run over the value.
func foldValue(value []byte, s Setting) ([]byte, bool) {
	v := bytes.TrimSpace(value)
	if len(v) == 0 || v[0] == '[' {
		return nil, false
	}
	if v[0] != '{' {
		v = []byte("{}")
	}
	t := NewSettings([]Setting{s})
	t.Write(v)
	out := append([]byte(nil), t.Out()...)
	out = append(out, t.Finish()...)
	if bad, _ := t.Unsupported(); bad {
		return nil, false
	}
	return out, true
}

func hasPathPrefix(path, prefix []string) bool {
	if len(prefix) > len(path) {
		return false
	}
	for i := range prefix {
		if path[i] != prefix[i] {
			return false
		}
	}
	return true
}

// normalizeSettings resolves the order dependencies between settings once, see the file comment. A setting the
// buffered path cannot apply (an array where it needs an object) is reported: the whole stage is then unsupported,
// so the buffered path produces its own outcome. The paths of settings a later overwrite makes moot are returned
// as probes: sjson still ran them first, and failed on an array in their way.
func normalizeSettings(in []Setting) (entries []*setEntry, probes [][]string, bad string) {
	var out []*setEntry
	for _, s := range in {
		e := &setEntry{path: s.Path, value: s.Value, overwrite: s.Overwrite}
		// settings at or below this one
		if e.overwrite {
			kept := out[:0]
			for _, o := range out {
				if hasPathPrefix(o.path, e.path) {
					if len(o.path) > len(e.path) {
						probes = append(probes, o.path)
					}
					continue
				}
				kept = append(kept, o)
			}
			out = kept
		} else {
			noop := false
			for _, o := range out {
				if hasPathPrefix(o.path, e.path) {
					noop = true // that setting creates this path, so it exists by the time this one runs
					break
				}
			}
			if noop {
				continue
			}
		}
		// ancestors: fold this value into theirs
		absorbed := false
		for _, o := range out {
			if len(o.path) < len(e.path) && hasPathPrefix(e.path, o.path) {
				v, ok := foldValue(o.value, Setting{Path: e.path[len(o.path):], Value: e.value, Overwrite: e.overwrite})
				if !ok {
					return nil, nil, "a setting below an earlier one whose value is not an object, sjson fails there"
				}
				o.value = v
				if o.overwrite {
					absorbed = true // the ancestor replaces the whole subtree, so this setting lives only inside it
				}
			}
		}
		if !absorbed {
			out = append(out, e)
		}
	}
	return out, probes, ""
}

func (p *settingsProto) add(e *setEntry) *setNode {
	n := &p.root
	for _, k := range e.path {
		c := n.byKey[k]
		if c == nil {
			c = &setNode{key: k, byKey: map[string]*setNode{}}
			n.byKey[k] = c
			n.children = append(n.children, c)
		}
		n = c
	}
	if n.entry == nil {
		n.entry = e
	}
	return n
}

func (p *settingsProto) top() *setNode { return p.stack[len(p.stack)-1] }

// start writes the overwrite entries of the container that just started, before its first key.
func (p *settingsProto) start(t *Transformer, n *setNode) {
	if n.started || p.silent > 0 {
		return
	}
	n.started = true
	w := t.W()
	for _, c := range n.children {
		if c.entry != nil && c.entry.overwrite {
			w.Key(c.key)
			w.Raw(c.entry.value)
		}
	}
}

func (p *settingsProto) OnKey(t *Transformer) Action {
	if p.bad != "" {
		return Bail(p.bad)
	}
	n := p.top()
	p.start(t, n)
	c := n.byKey[t.Last()]
	if c == nil {
		if p.silent > 0 {
			return Skip()
		}
		return Pass()
	}
	if c.entry != nil && c.entry.overwrite {
		c.state = setReplaced // the value went out when the container started
		if len(c.children) == 0 {
			return Skip()
		}
		return Probe() // dropped settings below it: their paths are walked, nothing is written
	}
	return Probe()
}

func (p *settingsProto) OnElem(t *Transformer) Action {
	return Bail("unexpected array: " + t.PathString())
}

func (p *settingsProto) OnStart(t *Transformer, kind ValueKind) Action {
	c := p.top().byKey[t.Last()]
	silent := p.silent > 0 || (c.entry != nil && c.entry.overwrite)
	switch kind {
	case KindObject:
		if !silent {
			c.state = setPresent
		}
		if len(c.children) > 0 {
			p.stack = append(p.stack, c)
			if silent {
				p.silent++
				return Enter().Lazy()
			}
			return Enter()
		}
		if silent {
			return Skip()
		}
		return Pass()
	case KindArray:
		if len(c.children) > 0 {
			return Bail("an array on the way to a setting, sjson fails there")
		}
		if silent {
			return Skip()
		}
		c.state = setPresent
		return Pass()
	}
	// string / number / bool / null
	if silent {
		return Skip()
	}
	if len(c.children) > 0 {
		c.state = setReplaced // sjson replaces it with the object the deeper settings build
		return Skip()
	}
	c.state = setPresent
	return Pass()
}

func (p *settingsProto) OnValue(t *Transformer, raw []byte) {}

func (p *settingsProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}

// leave completes a container that was present in the request: the add-if-absent values whose keys never came,
// and the objects sjson would have created for deeper settings.
func (p *settingsProto) leave(t *Transformer, n *setNode) {
	p.start(t, n) // an empty container never had a first key
	w := t.W()
	for _, c := range n.children {
		switch c.state {
		case setPresent:
			continue
		case setAbsent:
			if c.entry != nil {
				if !c.entry.overwrite {
					w.Key(c.key)
					w.Raw(c.entry.value) // carries its descendants already
				}
				continue
			}
		case setReplaced:
			if c.entry != nil && c.entry.overwrite {
				continue
			}
		}
		if len(c.children) > 0 {
			w.PushObj(c.key)
			p.absent(w, c)
			w.Pop()
		}
	}
}

// absent writes the subtree of a node that has no counterpart in the request.
func (p *settingsProto) absent(w *Writer, n *setNode) {
	for _, c := range n.children {
		if c.entry != nil {
			w.Key(c.key)
			w.Raw(c.entry.value)
			continue
		}
		w.PushObj(c.key)
		p.absent(w, c)
		w.Pop()
	}
}

func (p *settingsProto) OnLeave(t *Transformer) {
	if len(p.stack) == 1 {
		return // the root: Tail completes it
	}
	n := p.top()
	p.stack = p.stack[:len(p.stack)-1]
	if p.silent > 0 {
		p.silent--
		return
	}
	p.leave(t, n)
}

func (p *settingsProto) Tail(t *Transformer) {
	if p.bad != "" {
		t.Bail(p.bad)
		return
	}
	p.leave(t, &p.root)
}
