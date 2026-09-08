package streamxform

// The engine comes from github.com/axfor/ason (protocol-agnostic streaming JSON transformation: scanner, actions, lazy writer, strict validation).
// Its API is exposed in this package through aliases so the protocol files (proto_*.go, hook_tools.go) do not see the package boundary.

import "github.com/axfor/ason"

type (
	Transformer  = ason.Transformer
	Action       = ason.Action
	ValueKind    = ason.ValueKind
	Protocol     = ason.Protocol
	BaseProtocol = ason.BaseProtocol
	Writer       = ason.Writer
	DeferredKV   = ason.DeferredKV
	// Error / Code: details and classification of a bail (tr.Err()). The integration layer classifies fallbacks by Code, never by message text.
	Error = ason.Error
	Code  = ason.Code
)

const (
	ErrNone          = ason.ErrNone
	ErrSyntax        = ason.ErrSyntax
	ErrIncomplete    = ason.ErrIncomplete
	ErrRoot          = ason.ErrRoot
	ErrTrailing      = ason.ErrTrailing
	ErrDuplicateKey  = ason.ErrDuplicateKey
	ErrLimit         = ason.ErrLimit
	ErrLeftoverDefer = ason.ErrLeftoverDefer
	ErrUnsupported   = ason.ErrUnsupported
	ErrMisuse        = ason.ErrMisuse
)

const (
	KindString = ason.KindString
	KindObject = ason.KindObject
	KindArray  = ason.KindArray
	KindNull   = ason.KindNull
	KindBool   = ason.KindBool
	KindNumber = ason.KindNumber

	// CommitBytes is the commit window: no output is released before this many input bytes have been scanned.
	CommitBytes = ason.CommitBytes
)

// OutBufferSize is the scratch buffer each wasm VM keeps for building output: one commit window plus a
// generous chunk. Sized once per VM, never per request.
const OutBufferSize = 128 << 10

type (
	// FieldTypes is the set of JSON types a root-level field may have.
	FieldTypes = ason.FieldTypes
	// FieldTree is the recursive form: what a value may be, and what its children may be.
	FieldTree = ason.FieldTree
)

const (
	TypeString = ason.TypeString
	TypeNumber = ason.TypeNumber
	TypeBool   = ason.TypeBool
	TypeObject = ason.TypeObject
	TypeArray  = ason.TypeArray
	TypeAny    = ason.TypeAny
)

var (
	// FieldTypesOf derives the root-level field type table from the struct the buffered path unmarshals into.
	FieldTypesOf = ason.FieldTypesOf
	// FieldTreeOf derives the recursive form. Nothing consumes it yet: it is here so the evidence for nested
	// checking can be built before the behaviour changes.
	FieldTreeOf = ason.FieldTreeOf

	NewTransformer = ason.NewTransformer
	Pass           = ason.Pass
	Skip           = ason.Skip
	Capture        = ason.Capture
	Observe        = ason.Observe
	Defer          = ason.Defer
	Enter          = ason.Enter
	Probe          = ason.Probe
	Prefix         = ason.Prefix
	Bail           = ason.Bail
	BailCode       = ason.BailCode
)

// helpers shared by the protocol implementations
func appendJSONString(dst []byte, s string) []byte { return ason.AppendJSONString(dst, s) }
func jsonUnquote(raw []byte) (string, bool)        { return ason.JSONUnquote(raw) }
func unescapePrefix(b []byte) ([]byte, []int)      { return ason.UnescapePrefix(b) }
func isIntLiteral(s []byte) bool                   { return ason.IsIntLiteral(s) }
func isNumLiteral(s []byte) bool                   { return ason.IsNumLiteral(s) }
func isZeroNum(s []byte) bool                      { return ason.IsZeroNum(s) }
