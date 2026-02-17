package ail

import (
	"encoding/json"
	"strings"
	"testing"
)

// ─── Text delta: 1:1 forwarding ─────────────────────────────────────────────

func TestStreamConverter_TextDelta_OpenAIToAnthropic(t *testing.T) {
	conv, err := NewStreamConverter(StyleChatCompletions, StyleAnthropic)
	if err != nil {
		t.Fatal(err)
	}

	// OpenAI role chunk → Anthropic message_start
	role := `{"id":"chatcmpl-x","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`
	outputs, err := conv.Push([]byte(role))
	if err != nil {
		t.Fatalf("push role: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("role: want 1 output, got %d", len(outputs))
	}
	assertJSONField(t, outputs[0], "type", "message_start")

	// OpenAI text delta → Anthropic content_block_delta
	delta := `{"id":"chatcmpl-x","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`
	outputs, err = conv.Push([]byte(delta))
	if err != nil {
		t.Fatalf("push delta: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("delta: want 1 output, got %d", len(outputs))
	}
	assertJSONField(t, outputs[0], "type", "content_block_delta")

	// OpenAI finish → Anthropic message_delta + message_stop (2 events)
	finish := `{"id":"chatcmpl-x","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	outputs, err = conv.Push([]byte(finish))
	if err != nil {
		t.Fatalf("push finish: %v", err)
	}
	if len(outputs) != 2 {
		t.Fatalf("finish: want 2 outputs (message_delta + message_stop), got %d", len(outputs))
	}
	assertJSONField(t, outputs[0], "type", "message_delta")
	assertJSONField(t, outputs[1], "type", "message_stop")
}

func TestStreamConverter_TextDelta_AnthropicToOpenAI(t *testing.T) {
	conv, err := NewStreamConverter(StyleAnthropic, StyleChatCompletions)
	if err != nil {
		t.Fatal(err)
	}

	chunks := []string{
		`{"type":"message_start","message":{"id":"msg_01","model":"claude-3-opus"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	}

	var allOutputs [][]byte
	for i, chunk := range chunks {
		outputs, err := conv.Push([]byte(chunk))
		if err != nil {
			t.Fatalf("push chunk %d: %v", i, err)
		}
		allOutputs = append(allOutputs, outputs...)
	}

	if len(allOutputs) < 4 {
		t.Fatalf("want >= 4 output chunks, got %d", len(allOutputs))
	}

	// Every OpenAI chunk should have id and model (carried from message_start)
	for i, out := range allOutputs {
		var m map[string]any
		if json.Unmarshal(out, &m) != nil {
			continue
		}
		if _, ok := m["id"]; !ok {
			t.Errorf("chunk %d: missing 'id' field", i)
		}
		if _, ok := m["model"]; !ok {
			t.Errorf("chunk %d: missing 'model' field", i)
		}
	}

	// Last chunk should have finish_reason
	var lastChunk map[string]any
	json.Unmarshal(allOutputs[len(allOutputs)-1], &lastChunk)
	choices, ok := lastChunk["choices"].([]any)
	if ok && len(choices) > 0 {
		choice := choices[0].(map[string]any)
		if fr, ok := choice["finish_reason"]; ok {
			if fr != "stop" {
				t.Errorf("finish_reason: got %v, want 'stop'", fr)
			}
		}
	}
}

// ─── Metadata carry-over ────────────────────────────────────────────────────

func TestStreamConverter_MetadataCarryOver(t *testing.T) {
	conv, err := NewStreamConverter(StyleAnthropic, StyleChatCompletions)
	if err != nil {
		t.Fatal(err)
	}

	// First chunk has metadata
	start := `{"type":"message_start","message":{"id":"msg_42","model":"claude-3-5-sonnet"}}`
	_, err = conv.Push([]byte(start))
	if err != nil {
		t.Fatal(err)
	}

	// Second chunk is a text delta with NO metadata in the source
	delta := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`
	outputs, err := conv.Push([]byte(delta))
	if err != nil {
		t.Fatal(err)
	}

	if len(outputs) != 1 {
		t.Fatalf("want 1 output, got %d", len(outputs))
	}

	// The output should have id and model injected
	var m map[string]any
	json.Unmarshal(outputs[0], &m)
	if m["id"] != "msg_42" {
		t.Errorf("injected id: got %v, want msg_42", m["id"])
	}
	if m["model"] != "claude-3-5-sonnet" {
		t.Errorf("injected model: got %v, want claude-3-5-sonnet", m["model"])
	}
}

// ─── Tool call: OpenAI → Anthropic (1:1, split into events) ────────────────

func TestStreamConverter_ToolCall_OpenAIToAnthropic(t *testing.T) {
	conv, err := NewStreamConverter(StyleChatCompletions, StyleAnthropic)
	if err != nil {
		t.Fatal(err)
	}

	// Role chunk
	role := `{"id":"chatcmpl-t","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`
	outputs, err := conv.Push([]byte(role))
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 {
		t.Fatalf("role: want 1 output, got %d", len(outputs))
	}

	// Tool call start (id + name)
	toolStart := `{"id":"chatcmpl-t","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`
	outputs, err = conv.Push([]byte(toolStart))
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 {
		t.Fatalf("tool start: want 1 output, got %d", len(outputs))
	}
	assertJSONField(t, outputs[0], "type", "content_block_start")

	// Tool call argument fragments
	args1 := `{"id":"chatcmpl-t","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}`
	outputs, err = conv.Push([]byte(args1))
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 {
		t.Fatalf("args1: want 1 output, got %d", len(outputs))
	}
	assertJSONField(t, outputs[0], "type", "content_block_delta")

	args2 := `{"id":"chatcmpl-t","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]},"finish_reason":null}]}`
	outputs, err = conv.Push([]byte(args2))
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 {
		t.Fatalf("args2: want 1 output, got %d", len(outputs))
	}

	// Finish
	finish := `{"id":"chatcmpl-t","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`
	outputs, err = conv.Push([]byte(finish))
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 2 {
		t.Fatalf("finish: want 2 outputs, got %d", len(outputs))
	}
	assertJSONField(t, outputs[0], "type", "message_delta")
	assertJSONField(t, outputs[1], "type", "message_stop")
}

// ─── Tool call: Anthropic → OpenAI (1:1, no buffering) ─────────────────────

func TestStreamConverter_ToolCall_AnthropicToOpenAI(t *testing.T) {
	conv, err := NewStreamConverter(StyleAnthropic, StyleChatCompletions)
	if err != nil {
		t.Fatal(err)
	}

	chunks := []string{
		`{"type":"message_start","message":{"id":"msg_tc","model":"claude-3-opus"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\":\"NYC\"}"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}

	var allOutputs [][]byte
	for i, chunk := range chunks {
		outputs, err := conv.Push([]byte(chunk))
		if err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
		allOutputs = append(allOutputs, outputs...)
	}

	// Should have at least: start, tool_start, tool_delta1, tool_delta2, done, end
	if len(allOutputs) < 5 {
		t.Fatalf("want >= 5 outputs, got %d", len(allOutputs))
	}

	// Check that tool call data survives
	foundToolCall := false
	for _, out := range allOutputs {
		if strings.Contains(string(out), "get_weather") {
			foundToolCall = true
			break
		}
	}
	if !foundToolCall {
		t.Error("no output contained 'get_weather' tool call")
	}
}

// ─── Tool call buffering: Any → Google GenAI ────────────────────────────────

func TestStreamConverter_ToolBuffering_ToGoogle(t *testing.T) {
	conv, err := NewStreamConverter(StyleChatCompletions, StyleGoogleGenAI)
	if err != nil {
		t.Fatal(err)
	}

	// Tool call start (name only, no args yet)
	toolStart := `{"id":"chatcmpl-g","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`
	outputs, err := conv.Push([]byte(toolStart))
	if err != nil {
		t.Fatal(err)
	}
	// Should be buffered, no output yet (empty arguments aren't useful for Google)
	// Actually, the STREAM_TOOL_DELTA is buffered, but metadata may still emit.
	// The key point is no functionCall output yet.
	for _, out := range outputs {
		if strings.Contains(string(out), "functionCall") {
			t.Error("tool call should be buffered, not emitted yet")
		}
	}

	// Argument fragments
	args1 := `{"id":"chatcmpl-g","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":"}}]},"finish_reason":null}]}`
	outputs, err = conv.Push([]byte(args1))
	if err != nil {
		t.Fatal(err)
	}
	for _, out := range outputs {
		if strings.Contains(string(out), "functionCall") {
			t.Error("tool call should still be buffered")
		}
	}

	args2 := `{"id":"chatcmpl-g","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"NYC\"}"}}]},"finish_reason":null}]}`
	conv.Push([]byte(args2))

	// Finish triggers flush
	finish := `{"id":"chatcmpl-g","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`
	outputs, err = conv.Push([]byte(finish))
	if err != nil {
		t.Fatal(err)
	}

	// Should now have the flushed tool call
	foundFC := false
	for _, out := range outputs {
		if strings.Contains(string(out), "functionCall") || strings.Contains(string(out), "get_weather") {
			foundFC = true
			break
		}
	}
	if !foundFC {
		t.Error("expected flushed tool call with functionCall/get_weather in output")
		for i, out := range outputs {
			t.Logf("  output %d: %s", i, string(out))
		}
	}
}

func TestStreamConverter_Flush_PendingTools(t *testing.T) {
	conv, err := NewStreamConverter(StyleChatCompletions, StyleGoogleGenAI)
	if err != nil {
		t.Fatal(err)
	}

	// Push tool deltas without a finish event
	toolStart := `{"id":"chatcmpl-f","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_f","type":"function","function":{"name":"search","arguments":"{\"q\":\"hello\"}"}}]},"finish_reason":null}]}`
	conv.Push([]byte(toolStart))

	// Flush should emit the buffered tool
	flushed, err := conv.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if len(flushed) != 1 {
		t.Fatalf("flush: want 1 output, got %d", len(flushed))
	}
	if !strings.Contains(string(flushed[0]), "search") {
		t.Errorf("flushed output should contain tool name 'search': %s", string(flushed[0]))
	}

	// Second flush should return nothing
	flushed2, err := conv.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if len(flushed2) != 0 {
		t.Errorf("second flush: want 0 outputs, got %d", len(flushed2))
	}
}

// ─── Same-style passthrough ─────────────────────────────────────────────────

func TestStreamConverter_SameStylePassthrough(t *testing.T) {
	conv, err := NewStreamConverter(StyleChatCompletions, StyleChatCompletions)
	if err != nil {
		t.Fatal(err)
	}

	chunk := `{"id":"chatcmpl-p","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`
	outputs, err := conv.Push([]byte(chunk))
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 {
		t.Fatalf("passthrough: want 1 output, got %d", len(outputs))
	}

	// Should still be valid OpenAI format
	var m map[string]any
	if err := json.Unmarshal(outputs[0], &m); err != nil {
		t.Fatalf("invalid JSON output: %v", err)
	}
	if m["object"] != "chat.completion.chunk" {
		t.Errorf("object: got %v, want 'chat.completion.chunk'", m["object"])
	}
}

// ─── Anthropic → Anthropic (splitting behavior) ────────────────────────────

func TestStreamConverter_AnthropicToAnthropic(t *testing.T) {
	conv, err := NewStreamConverter(StyleAnthropic, StyleAnthropic)
	if err != nil {
		t.Fatal(err)
	}

	chunks := []string{
		`{"type":"message_start","message":{"id":"msg_rr","model":"claude-3-haiku"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`,
	}

	var allOutputs [][]byte
	for i, chunk := range chunks {
		outputs, err := conv.Push([]byte(chunk))
		if err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
		allOutputs = append(allOutputs, outputs...)
	}

	if len(allOutputs) != 4 {
		t.Fatalf("want 4 outputs, got %d", len(allOutputs))
	}

	// Verify event types preserved
	expectedTypes := []string{"message_start", "content_block_delta", "message_delta", "message_stop"}
	for i, expected := range expectedTypes {
		var m map[string]any
		json.Unmarshal(allOutputs[i], &m)
		if m["type"] != expected {
			t.Errorf("output %d: type=%v, want %s", i, m["type"], expected)
		}
	}
}

// ─── ConvertStreamChunk stateless convenience ───────────────────────────────

func TestConvertStreamChunk_Stateless(t *testing.T) {
	chunk := `{"id":"chatcmpl-s","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`

	out, err := ConvertStreamChunk([]byte(chunk), StyleChatCompletions, StyleAnthropic)
	if err != nil {
		t.Fatal(err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	// Should be an Anthropic content_block_delta
	if m["type"] != "content_block_delta" {
		t.Errorf("type: got %v, want content_block_delta", m["type"])
	}
}

// ─── ConvertResponse convenience ────────────────────────────────────────────

func TestConvertResponse(t *testing.T) {
	anthropicResp := `{
		"id": "msg_01abc",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-opus-20240229",
		"content": [{"type": "text", "text": "Hello!"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 3}
	}`

	out, err := ConvertResponse([]byte(anthropicResp), StyleAnthropic, StyleChatCompletions)
	if err != nil {
		t.Fatal(err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	// Should be an OpenAI chat completion
	if m["object"] != "chat.completion" {
		t.Errorf("object: got %v, want chat.completion", m["object"])
	}
	if m["id"] != "msg_01abc" {
		t.Errorf("id: got %v", m["id"])
	}
}

// ─── Error cases ────────────────────────────────────────────────────────────

func TestNewStreamConverter_InvalidStyles(t *testing.T) {
	_, err := NewStreamConverter("invalid-source", StyleChatCompletions)
	if err == nil {
		t.Error("expected error for invalid source style")
	}

	_, err = NewStreamConverter(StyleChatCompletions, "invalid-target")
	if err == nil {
		t.Error("expected error for invalid target style")
	}
}

func TestStreamConverter_InvalidJSON(t *testing.T) {
	conv, err := NewStreamConverter(StyleChatCompletions, StyleAnthropic)
	if err != nil {
		t.Fatal(err)
	}

	_, err = conv.Push([]byte(`not json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestStreamConverter_EmptyChunk(t *testing.T) {
	conv, err := NewStreamConverter(StyleChatCompletions, StyleAnthropic)
	if err != nil {
		t.Fatal(err)
	}

	// Valid JSON but no meaningful content
	outputs, err := conv.Push([]byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	// Should produce zero or minimal output (ping)
	if len(outputs) > 1 {
		t.Errorf("empty chunk: want <=1 output, got %d", len(outputs))
	}
}

// ─── Multi-tool call streaming ──────────────────────────────────────────────

func TestStreamConverter_MultiToolCall_Buffered(t *testing.T) {
	conv, err := NewStreamConverter(StyleAnthropic, StyleGoogleGenAI)
	if err != nil {
		t.Fatal(err)
	}

	chunks := []string{
		`{"type":"message_start","message":{"id":"msg_mt","model":"claude-3-opus"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"search"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"AI\"}"}}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_2","name":"fetch"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"url\":\"http://x\"}"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}

	var allOutputs [][]byte
	for i, chunk := range chunks {
		outputs, err := conv.Push([]byte(chunk))
		if err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
		allOutputs = append(allOutputs, outputs...)
	}

	// Check that both tool names appear in the output
	combined := ""
	for _, out := range allOutputs {
		combined += string(out)
	}
	if !strings.Contains(combined, "search") {
		t.Error("missing tool 'search' in output")
	}
	if !strings.Contains(combined, "fetch") {
		t.Error("missing tool 'fetch' in output")
	}
}

// ─── Anthropic emitter USAGE fix ────────────────────────────────────────────

func TestAnthropicEmitter_UsageInMessageDelta(t *testing.T) {
	prog := NewProgram()
	prog.EmitString(RESP_DONE, "stop")
	prog.EmitJSON(USAGE, json.RawMessage(`{"completion_tokens":42}`))

	emitter := &AnthropicEmitter{}
	out, err := emitter.EmitStreamChunk(prog)
	if err != nil {
		t.Fatal(err)
	}

	var m map[string]any
	json.Unmarshal(out, &m)
	if m["type"] != "message_delta" {
		t.Errorf("type: got %v, want message_delta", m["type"])
	}
	if _, ok := m["usage"]; !ok {
		t.Error("message_delta should include usage when present in program")
	}
}

// ─── helpers ────────────────────────────────────────────────────────────────

func assertJSONField(t *testing.T, data []byte, field, expected string) {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("assertJSONField: invalid JSON: %v\n  data: %s", err, string(data))
	}
	got, ok := m[field]
	if !ok {
		t.Errorf("assertJSONField: missing field %q in: %s", field, string(data))
		return
	}
	if got != expected {
		t.Errorf("assertJSONField: %s=%v, want %s", field, got, expected)
	}
}


