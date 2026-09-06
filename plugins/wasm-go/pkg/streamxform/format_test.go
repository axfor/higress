package streamxform

import "testing"

// 透传协议对没动的字节必须一个不改——包括 key 周围的空白与缩进，
// 与官方 sjson 原地改写的效果一致（fireworks / galadriel 的用例就靠这一点断言）。
func TestPassthroughPreservesFormatting(t *testing.T) {
	in := "{\n  \"model\": \"llama3.1\",\n  \"messages\": [ {\"role\": \"user\", \"content\": \"Hi\"} ],\n  \"temperature\": 0.7,\n  \"stream\": true\n}"
	want := "{\n  \"model\": \"llama3.1\",\n  \"messages\": [ {\"role\": \"user\", \"content\": \"Hi\"} ],\n  \"temperature\": 0.7,\n  \"stream\": true,\"stream_options\":{\"include_usage\":true}\n}"
	for _, cs := range []int{1, 4, 4096} {
		tr := NewOpenAI(OpenAIOptions{DetectStream: true, NormalizeUsage: true, CheckMessages: true})
		var out []byte
		for i := 0; i < len(in); i += cs {
			j := i + cs
			if j > len(in) {
				j = len(in)
			}
			tr.Write([]byte(in[i:j]))
			out = append(out, tr.Out()...)
		}
		out = append(out, tr.Finish()...)
		if bad, why := tr.Unsupported(); bad {
			t.Fatalf("chunk=%d 意外回落: %s", cs, why)
		}
		if string(out) != want {
			t.Errorf("chunk=%d 格式未保留:\n得到: %q\n期望: %q", cs, out, want)
		}
	}
}

// 值与逗号之间的空白也要保留（sjson 原地改写会保留它）。
func TestTrailingWhitespaceBeforeComma(t *testing.T) {
	in := "{\n  \"model\" : \"m\" ,\n  \"messages\" : [ 1 , { \"a\" : 1 , \"b\" : 2 } , 3 ] ,\n  \"stream\" : true\n}\n"
	for _, cs := range []int{1, 3, 4096} {
		out, ok, why := feedAll(NewOpenAI(OpenAIOptions{}), in, cs)
		if !ok {
			t.Fatalf("chunk=%d: %s", cs, why)
		}
		if out != in {
			t.Fatalf("chunk=%d 透传不保真:\n got  %q\n want %q", cs, out, in)
		}
	}
}
