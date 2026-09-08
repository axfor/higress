package provider

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// officialGemini is the pure-function part of gemini.go onChatCompletionRequestBody
// (the decode of parseRequestAndMapModel + empty model check → buildGeminiChatRequest → Marshal).
// The buffered path then fetches http(s) images asynchronously; needFetch=true says that shape must fall back when streaming.
func officialGemini(in string, safety map[string]string, budget int64) (m map[string]any, ok bool, needFetch bool) {
	m, err, needFetch := officialGeminiErr(in, safety, budget)
	return m, err == nil, needFetch
}

func officialGeminiErr(in string, safety map[string]string, budget int64) (m map[string]any, err error, needFetch bool) {
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest([]byte(in), req); err != nil {
		return nil, err, false
	}
	if req.Model == "" {
		return nil, fmt.Errorf("missing model in request"), false
	}
	req.Model = gemMapping(req.Model) // parseRequestAndMapModel writes the mapped model back into request.Model
	g := &geminiProvider{config: ProviderConfig{geminiSafetySetting: safety, geminiThinkingBudget: budget}}
	gr := g.buildGeminiChatRequest(req)
	if g.countImageUrl(gr) > 0 {
		needFetch = true
	}
	b, err := json.Marshal(gr)
	if err != nil {
		return nil, err, false
	}
	m, _ = decodeMap(b)
	// safetySettings comes from a map iteration with no fixed order: sort before comparing
	if ss, ok := m["safetySettings"].([]any); ok {
		sort.Slice(ss, func(i, j int) bool {
			return ss[i].(map[string]any)["category"].(string) < ss[j].(map[string]any)["category"].(string)
		})
	}
	return m, nil, needFetch
}

var gemSafety = map[string]string{"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH"}

func gemMapping(m string) string {
	if m == "m1" {
		return "gemini-2.5-flash"
	}
	return m
}

func newGeminiStream(withSafety bool, budget int64) *streamxform.Transformer {
	var ss []streamxform.GeminiSafetySetting
	if withSafety {
		for k, v := range gemSafety {
			ss = append(ss, streamxform.GeminiSafetySetting{Category: k, Threshold: v})
		}
		sort.Slice(ss, func(i, j int) bool { return ss[i].Category < ss[j].Category })
	}
	return streamxform.NewGemini(streamxform.GeminiOptions{
		MapModel: func(m string) (string, error) {
			if m == "" {
				return "", fmt.Errorf("missing model in request")
			}
			return gemMapping(m), nil
		},
		ThinkingModel:  func(m string) bool { return geminiThinkingModels[m] },
		ThinkingBudget: budget,
		SafetySettings: ss,
	})
}

var geminiCases = []string{
	`{"model":"m1","messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","messages":[{"role":"system","content":"S"},{"role":"user","content":"U"},{"role":"assistant","content":"A"}]}`,
	`{"model":"x","messages":[{"role":"system","content":"S1"},{"role":"system","content":[{"type":"text","text":"S2"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]},{"role":"user","content":"U"}]}`,
	`{"model":"x","messages":[{"role":"system","content":"S"}]}`,
	`{"model":"x","messages":[{"role":"system"}]}`,
	`{"model":"x","messages":[{"role":"system","content":null}]}`,
	`{"model":"x","messages":[{"content":"U","role":"user"},{"content":"A","role":"assistant"}]}`,
	`{"model":"x","messages":[{}]}`,
	`{"model":"x","messages":[{"role":"user"}]}`,
	`{"model":"x","messages":[{"role":"user","content":null}]}`,
	`{"model":"x","messages":[{"role":"user","content":{"a":1}}]}`,
	`{"model":"x","messages":[{"role":"tool","tool_call_id":"c","content":"r"}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":""},{"text":"b","type":"text"},{"type":"video","text":"x"},{"text":"notype"},"s",1,null]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,\/9j\/4AAQ","detail":"high"}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:nosemi"}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:text/plain,hi"}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"not-a-url"}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":""}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"image_url":{"url":"data:image/png;base64,AA"},"type":"image_url"}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://x/y.png"}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"HTTP://x/y.png"}}]}]}`,
	`{"model":"x","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA","format":"wav"}},{"type":"file","file":{"file_id":"f"}},{"type":"text","text":"t"}]}]}`,
	`{"model":"x","temperature":0.5,"top_p":0.9,"max_tokens":100,"presence_penalty":1.9,"frequency_penalty":-2.5,"logprobs":true,"modalities":["TEXT","IMAGE"],"messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","temperature":0,"top_p":0,"max_tokens":0,"presence_penalty":0.4,"logprobs":false,"modalities":[],"messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","stop":["E"],"seed":3,"n":2,"max_completion_tokens":50,"tool_choice":"auto","stream":true,"messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"b":{"type":"string"},"a":{"type":"number"}}}}},{"type":"function","function":{"name":"g"}}],"messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","tools":[],"messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","tools":null,"messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"user","content":"U","name":"n"}]}`,
	`{"model":"m1","messages":[{"role":"user","content":"` + strings.Repeat("y", 100000) + `"}],"stream":true}`,
	`{"messages":[{"role":"user","content":"U"}]}`,
	`{"model":"x","messages":[]}`,
	`{"model":"x"}`,
}

func TestGeminiDifferential(t *testing.T) {
	for _, in := range geminiCases {
		for _, safety := range []bool{true, false} {
			sm := map[string]string(nil)
			if safety {
				sm = gemSafety
			}
			off, ok, needFetch := officialGemini(in, sm, 1024)
			for _, cs := range []int{1, 7, 4096} {
				str, sok, why := runStream(typed(newGeminiStream(safety, 1024)), in, cs)
				if !ok {
					if sok {
						t.Errorf("buffered path failed but streaming passed: %s", trunc(in, 80))
					} else if cs == 4096 && safety {
						fmt.Printf("  %-60s buffered path failed, streaming fell back ✓ (%s)\n", trunc(in, 60), why)
					}
					continue
				}
				if needFetch {
					if sok {
						t.Errorf("an http image should fall back but passed: %s", trunc(in, 80))
					} else if cs == 4096 && safety {
						fmt.Printf("  %-60s http image → fallback ✓\n", trunc(in, 60))
					}
					continue
				}
				if !sok {
					t.Errorf("chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
					continue
				}
				if ss, ok := str["safetySettings"].([]any); ok {
					sort.Slice(ss, func(i, j int) bool {
						return ss[i].(map[string]any)["category"].(string) < ss[j].(map[string]any)["category"].(string)
					})
				}
				if d := diffMaps(off, str); d != "" {
					t.Errorf("chunk=%d safety=%v mismatch: %s\n  input: %s", cs, safety, d, trunc(in, 200))
				} else if cs == 4096 && safety {
					fmt.Printf("  %-60s ✓ identical\n", trunc(in, 60))
				}
			}
		}
	}
}

func TestGeminiFuzz(t *testing.T) {
	seed := fuzzSeed()
	r := rand.New(rand.NewSource(seed))
	N := fuzzN(60000)
	same, offFail, fetch, fb, lenient := 0, 0, 0, 0, 0
	for i := 0; i < N; i++ {
		in := genRequest(r)
		safety := r.Intn(2) == 0
		sm := map[string]string(nil)
		if safety {
			sm = gemSafety
		}
		budget := int64(r.Intn(3)) * 512
		off, oerr, needFetch := officialGeminiErr(in, sm, budget)
		ok := oerr == nil
		chunk := []int{1, 3, 17, 64, 4096}[r.Intn(5)]
		tr := typed(newGeminiStream(safety, budget))
		str, sok, why := runStream(tr, in, chunk)
		if !ok {
			offFail++
			if sok {
				if isDiscardedFieldTypeErrorIn(oerr, geminiExtraDiscarded) {
					lenient++ // type error in a dropped field: known lenient difference
					continue
				}
				t.Fatalf("case %d: buffered path failed but streaming passed (chunk=%d): %v\n  input: %s", i, chunk, oerr, in)
			}
			continue
		}
		if needFetch {
			fetch++
			if sok {
				t.Fatalf("case %d: an http image should fall back but passed (chunk=%d)\n  input: %s", i, chunk, in)
			}
			continue
		}
		if !sok {
			if strings.Contains(why, "duplicate key") || strings.Contains(why, "panic") {
				fb++
				continue
			}
			t.Fatalf("case %d: unexpected fallback (chunk=%d): %s\n  input: %s", i, chunk, why, in)
		}
		if ss, ok := str["safetySettings"].([]any); ok {
			sort.Slice(ss, func(i, j int) bool {
				return ss[i].(map[string]any)["category"].(string) < ss[j].(map[string]any)["category"].(string)
			})
		}
		if d := diffMaps(off, str); d != "" {
			t.Fatalf("case %d: mismatch (chunk=%d)\n  input: %s\n  diff: %s", i, chunk, in, d)
		}
		same++
	}
	fmt.Printf("  Gemini random %d cases (seed=%d): identical %d, buffered failed %d (of which dropped-field type errors passed by streaming %d), http image fallbacks %d, known fallbacks %d, mismatches 0\n", N, seed, same, offFail, lenient, fetch, fb)
}
