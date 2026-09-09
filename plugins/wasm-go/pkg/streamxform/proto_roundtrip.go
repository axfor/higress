package streamxform

import (
	"math"
	"strconv"

	"github.com/axfor/ason"
)

// The struct round trip: the streaming form of "json.Unmarshal into the request struct, json.Marshal it back",
// which the buffered path makes on the way whenever it edits a request through the struct (responseJsonSchema,
// the context insertion) rather than with sjson. The document keeps its meaning but not its shape: keys the
// struct has no field for are dropped, at every level the struct describes; the zero value of an omitempty field
// is left out (an empty string, 0, false, an empty array or object, null); a field the struct has and the
// document lacks is written as its zero value unless omitempty; a subtree the struct takes as an interface or
// through its own decoder passes verbatim.
//
// The stage is driven by the FieldTree derived from the struct, which now carries the marshal-side facts.
// Numbers are re-rendered as the marshal renders them (1.0 and 1e0 become 1, an int field takes only an integer
// literal), in the struct's own fields and inside the subtrees it takes as interfaces alike -- a decoded number
// is a float64 there. Strings are not: an escape stays an escape, which any receiver decodes the same. Where the
// tree cannot say what the marshal writes (a type that marshals itself, absent and not omitempty), the stage
// reports unsupported.
//
// Set adds the edit the buffered path made between the decode and the marshal: root keys whose value is
// replaced (or added) by the given raw JSON, written at the end.

// NewStructRoundTrip builds the stage for the struct behind tree; set replaces / adds root keys.
func NewStructRoundTrip(tree *FieldTree, set map[string][]byte) *Transformer {
	p := &roundTripProto{tree: tree, set: set}
	t := NewTransformer(p)
	t.DupKeyBail = true // Unmarshal takes the last of duplicate keys; the buffered path then has one, not two
	return t
}

type rtFrame struct {
	node    *FieldTree      // the container's node
	isMap   bool            // a map: every key takes node.Elem
	seen    map[string]bool // struct keys met, for the zero values written at the close
	child   *FieldTree      // the node of the value being decided (after OnKey / OnElem)
	key     string
	lazy    bool // the container is written lazily (omitempty: empty is left out)
	silent  bool // inside a dropped subtree: nothing is written
	generic bool // a subtree the struct takes as an interface: everything passes, numbers re-rendered
}

type roundTripProto struct {
	tree   *FieldTree
	set    map[string][]byte
	frames []*rtFrame
}

func (p *roundTripProto) top() *rtFrame { return p.frames[len(p.frames)-1] }

// childOf resolves what the tree says about the value under key (or an element) of the current container.
func (p *roundTripProto) childOf(f *rtFrame, key string, elem bool) *FieldTree {
	if f.node == nil {
		return nil
	}
	if elem || f.isMap {
		return f.node.Elem
	}
	if f.node.Keys == nil {
		return nil
	}
	return f.node.Keys[key]
}

func (p *roundTripProto) OnKey(t *Transformer) Action {
	if len(p.frames) == 0 {
		p.frames = append(p.frames, &rtFrame{node: p.tree, seen: map[string]bool{}})
	}
	f := p.top()
	key := t.Last()
	if f.silent {
		return Skip()
	}
	if f.generic {
		f.child, f.key = nil, key
		return Probe()
	}
	if len(p.frames) == 1 && p.set != nil {
		if _, ok := p.set[key]; ok {
			f.seen[key] = true
			return Skip() // replaced, written at the end
		}
	}
	if f.isMap {
		f.child, f.key = f.node.Elem, key
		return p.decide(t, f)
	}
	if f.node == nil || f.node.Keys == nil {
		f.child = nil
		return Probe() // deeper than the tree describes: the generic walk
	}
	c, ok := f.node.Keys[key]
	if !ok {
		return Skip() // no field for it: the decode drops it
	}
	f.seen[key] = true
	f.child, f.key = c, key
	return p.decide(t, f)
}

func (p *roundTripProto) OnElem(t *Transformer) Action {
	f := p.top()
	if f.silent {
		return Skip()
	}
	if f.generic || f.node == nil || f.node.Elem == nil {
		f.child, f.key = nil, ""
		return Probe() // the generic walk: numbers re-rendered, the rest verbatim
	}
	f.child, f.key = f.node.Elem, ""
	return p.decide(t, f)
}

// decide picks the action for the value f.child describes; the value's kind is needed for most of them.
func (p *roundTripProto) decide(t *Transformer, f *rtFrame) Action {
	c := f.child
	if c == nil {
		return Pass()
	}
	return Probe()
}

func (p *roundTripProto) OnStart(t *Transformer, kind ValueKind) Action {
	f := p.top()
	c := f.child
	if c == nil {
		return p.genericStart(kind)
	}
	if kind == KindNull {
		return p.onNull(t, f, c)
	}
	// a self-decoding type, an interface, or a subtree the tree does not describe: the generic walk
	if c.Any || c.Kind == ason.FieldInterface || c.Kind == ason.FieldOther {
		return p.genericStart(kind)
	}
	if !typeAllowed(c.Types, kind) {
		return Bail("a value of a type the buffered decode rejects")
	}
	switch kind {
	case KindString:
		if c.Omit && c.Kind == ason.FieldString {
			return Prefix(1) // "" is left out
		}
		return Pass()
	case KindNumber:
		return Capture(64) // re-rendered as the marshal renders it; 0 left out when omitempty
	case KindBool:
		if c.Omit && c.Kind == ason.FieldBool {
			return Capture(8) // false is left out
		}
		return Pass()
	case KindArray:
		if c.Kind != ason.FieldSlice {
			return Pass() // a fixed array or a []byte: as it came
		}
		fr := &rtFrame{node: c, seen: map[string]bool{}}
		p.frames = append(p.frames, fr)
		if c.Omit {
			fr.lazy = true
			return Enter().Lazy() // an empty slice is left out
		}
		return Enter()
	case KindObject:
		switch c.Kind {
		case ason.FieldMap:
			fr := &rtFrame{node: c, isMap: true, seen: map[string]bool{}}
			p.frames = append(p.frames, fr)
			if c.Omit {
				fr.lazy = true
				return Enter().Lazy()
			}
			return Enter()
		case ason.FieldStruct, ason.FieldPointer:
			if c.Keys == nil {
				return Pass() // deeper than the tree goes
			}
			fr := &rtFrame{node: c, seen: map[string]bool{}}
			p.frames = append(p.frames, fr)
			return Enter() // a struct is never omitted, a present pointer is not nil
		}
		return Pass()
	}
	return Pass()
}

// genericStart: a value in a subtree the struct takes as interface{}: numbers are decoded as float64 and
// rendered again, containers are walked for the numbers inside them, the rest passes.
func (p *roundTripProto) genericStart(kind ValueKind) Action {
	switch kind {
	case KindNumber:
		return Capture(64)
	case KindObject, KindArray:
		p.frames = append(p.frames, &rtFrame{generic: true})
		return Enter()
	}
	return Pass()
}

// renderNumber is what the marshal writes for a decoded number: an int field decimal, everything else a
// float64 in encoding/json's shortest form.
func renderNumber(raw []byte, isInt bool) ([]byte, bool) {
	if isInt {
		if !isIntLiteral(raw) {
			return nil, false
		}
		n, err := strconv.ParseInt(string(raw), 10, 64)
		if err != nil {
			return nil, false
		}
		return strconv.AppendInt(nil, n, 10), true
	}
	f, err := strconv.ParseFloat(string(raw), 64)
	if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
		return nil, false
	}
	abs := math.Abs(f)
	format := byte('f')
	if abs != 0 && (abs < 1e-6 || abs >= 1e21) {
		format = 'e'
	}
	b := strconv.AppendFloat(nil, f, format, -1, 64)
	if format == 'e' { // encoding/json: e-09 becomes e-9
		if n := len(b); n >= 4 && b[n-4] == 'e' && b[n-3] == '-' && b[n-2] == '0' {
			b[n-2] = b[n-1]
			b = b[:n-1]
		}
	}
	return b, true
}

// onNull: what the field's Go value becomes on null, and whether that is left out.
func (p *roundTripProto) onNull(t *Transformer, f *rtFrame, c *FieldTree) Action {
	w := t.W()
	if c.Kind == ason.FieldOther {
		return Pass() // a type that decodes null itself (json.RawMessage keeps it): as it came
	}
	if c.Omit {
		return Skip() // every kind's null decodes to its zero value, which omitempty leaves out
	}
	if c.Kind == ason.FieldStruct && !c.Any {
		zero, ok := c.ZeroJSON()
		if !ok {
			return Bail("the zero value of a field the buffered path writes back cannot be reproduced")
		}
		p.writeValue(w, f, []byte(zero))
		return Skip()
	}
	zero, ok := c.ZeroJSON()
	if !ok {
		return Bail("the zero value of a field the buffered path writes back cannot be reproduced")
	}
	p.writeValue(w, f, []byte(zero))
	return Skip()
}

// writeValue writes a value in place of the one being decided: under its key, or as the next element.
func (p *roundTripProto) writeValue(w *Writer, f *rtFrame, raw []byte) {
	if f.key != "" || !p.isArrayFrame(f) {
		w.Key(f.key)
	} else {
		w.Elem()
	}
	w.Raw(raw)
}

func (p *roundTripProto) isArrayFrame(f *rtFrame) bool {
	return f.node != nil && f.node.Kind == ason.FieldSlice
}

func (p *roundTripProto) OnValue(t *Transformer, raw []byte) {
	f := p.top()
	c := f.child
	w := t.W()
	out := raw
	if c == nil { // a number met by the generic walk
		r, ok := renderNumber(raw, false)
		if !ok {
			t.Bail("a number the buffered decode rejects")
			return
		}
		out = r
	} else {
		// a captured scalar of a field: written unless omitempty leaves its zero value out
		switch c.Kind {
		case ason.FieldNumber:
			r, ok := renderNumber(raw, c.Int)
			if !ok {
				t.Bail("a number the buffered decode rejects (a fraction or an exponent in an integer field)")
				return
			}
			if c.Omit && isZeroNum(raw) {
				return
			}
			out = r
		case ason.FieldBool:
			if c.Omit && string(raw) == "false" {
				return
			}
		}
	}
	if f.key != "" {
		w.KeyRaw(t.KeyRaw())
	} else {
		w.Elem()
	}
	w.Raw(out)
}

func (p *roundTripProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	f := p.top()
	w := t.W()
	if complete && len(raw) == 0 {
		return Skip(), 0 // "" of an omitempty string field
	}
	if f.key != "" {
		w.KeyRaw(t.KeyRaw())
	} else {
		w.Elem()
	}
	w.Byte('"')
	return Pass().Wrap(nil, lit0), 0
}

func (p *roundTripProto) OnLeave(t *Transformer) {
	if len(p.frames) <= 1 {
		return // the root closes in Tail
	}
	f := p.top()
	p.frames = p.frames[:len(p.frames)-1]
	if f.silent || f.generic {
		return
	}
	if !f.isMap && f.node != nil && f.node.Keys != nil {
		if !p.zerosFor(t, f) {
			return
		}
	}
}

// zerosFor writes the zero values of the struct's non-omitempty fields the document did not carry.
func (p *roundTripProto) zerosFor(t *Transformer, f *rtFrame) bool {
	w := t.W()
	for _, k := range f.node.SortedKeys() {
		c := f.node.Keys[k]
		if c.Omit || f.seen[k] {
			continue
		}
		zero, ok := c.ZeroJSON()
		if !ok {
			t.Bail("the zero value of a field the buffered path writes back cannot be reproduced: " + k)
			return false
		}
		w.Key(k)
		w.RawString(zero)
	}
	return true
}

func (p *roundTripProto) Tail(t *Transformer) {
	if len(p.frames) == 0 {
		p.frames = append(p.frames, &rtFrame{node: p.tree, seen: map[string]bool{}})
	}
	f := p.frames[0]
	w := t.W()
	for k, v := range p.set {
		w.Key(k)
		w.Raw(v)
		f.seen[k] = true
	}
	if f.node != nil && f.node.Keys != nil {
		p.zerosFor(t, f)
	}
}

// typeAllowed: whether the tree lets a value of this kind into the field (null is always allowed).
func typeAllowed(types FieldTypes, kind ValueKind) bool {
	switch kind {
	case KindString:
		return types&TypeString != 0
	case KindNumber:
		return types&TypeNumber != 0
	case KindBool:
		return types&TypeBool != 0
	case KindObject:
		return types&TypeObject != 0
	case KindArray:
		return types&TypeArray != 0
	}
	return true
}
