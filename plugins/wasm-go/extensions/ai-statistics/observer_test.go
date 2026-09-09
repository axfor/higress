package main

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/tidwall/gjson"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// buffered logic (gjson): read model and count user turns.
func officialModelAndRounds(body []byte) (string, int) {
	requestModel := "UNKNOWN"
	if m := gjson.GetBytes(body, "model"); m.Exists() {
		requestModel = m.String()
	}
	n := 0
	if messages := gjson.GetBytes(body, "messages"); messages.Exists() && messages.IsArray() {
		for _, msg := range messages.Array() {
			if msg.Get("role").String() == "user" {
				n++
			}
		}
	} else if contents := gjson.GetBytes(body, "contents"); contents.Exists() && contents.IsArray() {
		for _, c := range contents.Array() {
			if !c.Get("role").Exists() || c.Get("role").String() == "user" {
				n++
			}
		}
	}
	return requestModel, n
}

func observe(body []byte, chunk int) (string, int, bool) {
	tr := newRequestObserver()
	for i := 0; i < len(body); i += chunk {
		j := i + chunk
		if j > len(body) {
			j = len(body)
		}
		tr.Write(body[i:j])
		tr.Out()
	}
	tr.Finish()
	if bad, _ := tr.Unsupported(); bad {
		return "", 0, false
	}
	p := tr.Protocol().(*requestObserver)
	model := "UNKNOWN"
	if p.modelSeen {
		model = p.model
	}
	return model, p.rounds(), true
}

func TestObserverMatchesOfficial(t *testing.T) {
	fixed := []string{
		`{"model":"m","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":[{"type":"text","text":"c"}]}]}`,
		`{"messages":[{"role":"user"}],"model":"m2","model":"dup"}`,
		`{"messages":"nope","contents":[{"parts":[]},{"role":"user"},{"role":"model"}]}`,
		`{"messages":[],"contents":[{"role":"user"}]}`,
		`{"contents":[{"role":123},{"role":null},{}]}`,
		`{"messages":[1,"x",null,{"role":"user","role":"assistant"},{"role":["user"]},{"role":"USER"}]}`,
		`{"model":123,"messages":[{"role":"user"}]}`,
		`{"model":null}`,
		`{"model":{"a":1}}`,
		`{"model":true,"x":{"model":"inner","messages":[{"role":"user"}]}}`,
		"{\n \"model\" : \"m\" ,\n \"messages\" : [ { \"role\" : \"user\" } ]\n}\n",
		`{}`,
	}
	r := rand.New(rand.NewSource(11))
	roles := []string{`"user"`, `"assistant"`, `"system"`, `"tool"`, `null`, `1`, `"User"`, `""`}
	for i := 0; i < 300; i++ {
		var els []string
		for k := 0; k < r.Intn(6); k++ {
			switch r.Intn(6) {
			case 0:
				els = append(els, `{"content":"x","role":`+roles[r.Intn(len(roles))]+`}`)
			case 1:
				els = append(els, `{"role":`+roles[r.Intn(len(roles))]+`,"content":[{"type":"text","text":"`+strings.Repeat("y", r.Intn(2000))+`"}]}`)
			case 2:
				els = append(els, `{"parts":[{"text":"t"}]}`)
			case 3:
				els = append(els, `5`)
			case 4:
				els = append(els, `{"role":"user","meta":{"role":"assistant"}}`)
			case 5:
				els = append(els, `{"role":`+roles[r.Intn(len(roles))]+`,"role":"user"}`)
			}
		}
		arr := "[" + strings.Join(els, ",") + "]"
		var fields []string
		if r.Intn(4) != 0 {
			fields = append(fields, `"model":`+[]string{`"m1"`, `"p/m"`, `null`, `7`, `""`}[r.Intn(5)])
		}
		switch r.Intn(4) {
		case 0:
			fields = append(fields, `"messages":`+arr)
		case 1:
			fields = append(fields, `"contents":`+arr)
		case 2:
			fields = append(fields, `"messages":`+arr, `"contents":`+arr)
		case 3:
			fields = append(fields, `"messages":`+[]string{`{}`, `"s"`, `null`}[r.Intn(3)], `"contents":`+arr)
		}
		if r.Intn(3) == 0 {
			fields = append(fields, `"stream":true`, `"temperature":0.5`)
		}
		r.Shuffle(len(fields), func(a, b int) { fields[a], fields[b] = fields[b], fields[a] })
		fixed = append(fixed, "{"+strings.Join(fields, ",")+"}")
	}
	for _, body := range fixed {
		wm, wn := officialModelAndRounds([]byte(body))
		for _, cs := range []int{1, 3, 4096} {
			gm, gn, ok := observe([]byte(body), cs)
			if !ok {
				t.Fatalf("%s chunk=%d: observation was marked unsupported", body, cs)
			}
			if gm != wm || gn != wn {
				t.Fatalf("%s chunk=%d: got (%q,%d) want (%q,%d)", body, cs, gm, gn, wm, wn)
			}
		}
	}
	_ = fmt.Sprint
	_ = streamxform.CommitBytes
}
