package streamxform

import (
	"strings"
	"testing"
)

func stripPrefix(prefix string) func(string) string {
	return func(m string) string { return strings.TrimPrefix(m, prefix) }
}

// Two OpenAI passthrough stages in series: the composition is the composition of the two rewrites, under any
// chunking, with the second stage's window the one that holds output back.
func TestPipelineComposesTwoStages(t *testing.T) {
	body := `{"model":"a/b/m","messages":[{"role":"user","content":"` + strings.Repeat("y", 100000) + `"}],"stream":true}`
	want := strings.Replace(body, `"a/b/m"`, `"m"`, 1)
	for _, chunk := range []int{1, 7, 4096, len(body)} {
		p := NewPipeline(NewOpenAI(OpenAIOptions{MapModel: stripPrefix("a/")}), NewOpenAI(OpenAIOptions{MapModel: stripPrefix("b/"), DetectStream: true}))
		var out []byte
		p.SetSink(func(b []byte) { out = append(out, b...) })
		released := -1
		for i := 0; i < len(body); i += chunk {
			j := i + chunk
			if j > len(body) {
				j = len(body)
			}
			p.Write([]byte(body[i:j]))
			if released < 0 && p.Committed() {
				released = j
			}
		}
		p.Finish()
		if bad, why := p.Unsupported(); bad {
			t.Fatalf("chunk=%d: unsupported: %s", chunk, why)
		}
		if string(out) != want {
			t.Fatalf("chunk=%d: output differs: %q", chunk, out[len(out)-60:])
		}
		// the first stage shortens the model by two bytes, so the second stage's window fills two input bytes later
		if chunk < len(body) && (released < CommitBytes || released >= CommitBytes+chunk+2) {
			t.Fatalf("chunk=%d: the second stage's window should decide the release, released at %d", chunk, released)
		}
		pre := p.Protocol().(Preluder).Prelude()
		if pre.Model != "b/m" || !pre.Stream {
			t.Fatalf("prelude should be the second stage's view: %+v", pre)
		}
	}
}

// A document the first stage rejects is unsupported before the second stage has released anything.
func TestPipelineFirstStageBailsBeforeRelease(t *testing.T) {
	body := `{"model":"a/m","messages":[{"role":"user","content":"x"},{"role":"user","content":"y"}],"model":"dup"}`
	first := NewOpenAI(OpenAIOptions{MapModel: stripPrefix("a/")})
	first.DupKeyBail = true
	p := NewPipeline(first, NewOpenAI(OpenAIOptions{}))
	var out []byte
	p.SetSink(func(b []byte) { out = append(out, b...) })
	p.Write([]byte(body))
	p.Finish()
	if bad, _ := p.Unsupported(); !bad {
		t.Fatal("the first stage's duplicate key should make the pipeline unsupported")
	}
	if p.Committed() {
		t.Fatal("nothing should have been released before the commit point")
	}
}
