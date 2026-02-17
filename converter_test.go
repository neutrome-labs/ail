package ail

import (
	"encoding/json"
	"testing"
)

func TestChatCompletionsRequestRoundTrip(t *testing.T) {
	input := `{
		"model": "gpt-4o",
		"temperature": 0.7,
		"max_tokens": 1024,
		"stream": true,
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Hello!"},
			{"role": "assistant", "content": "Hi there!"},
			{"role": "user", "content": "What is 2+2?"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "calculator",
					"description": "Do math",
					"parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
				}
			}
		]
	}`

	parser := &ChatCompletionsParser{}
	prog, err := parser.ParseRequest([]byte(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	// Verify program structure
	if m := prog.GetModel(); m != "gpt-4o" {
		t.Errorf("model: got %q, want gpt-4o", m)
	}
	if !prog.IsStreaming() {
		t.Error("expected streaming")
	}

	// Verify disassembly includes key instructions
	asm := prog.Disasm()
	t.Logf("Disassembly:\n%s", asm)

	// Emit back to Chat Completions
	emitter := &ChatCompletionsEmitter{}
	out, err := emitter.EmitRequest(prog)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	// Parse the output to verify structure
	var result map[string]json.RawMessage
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}

	// Model
	var model string
	json.Unmarshal(result["model"], &model)
	if model != "gpt-4o" {
		t.Errorf("output model: got %q, want gpt-4o", model)
	}

	// Messages
	var messages []map[string]any
	json.Unmarshal(result["messages"], &messages)
	if len(messages) != 4 {
		t.Errorf("output messages count: got %d, want 4", len(messages))
	}
	if messages[0]["role"] != "system" {
		t.Errorf("first message role: got %v, want system", messages[0]["role"])
	}

	// Tools
	var tools []map[string]any
	json.Unmarshal(result["tools"], &tools)
	if len(tools) != 1 {
		t.Errorf("output tools count: got %d, want 1", len(tools))
	}

	// Stream
	var stream bool
	json.Unmarshal(result["stream"], &stream)
	if !stream {
		t.Error("output stream: expected true")
	}
}

func TestChatCompletionsToolCallRoundTrip(t *testing.T) {
	input := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "What is the weather in NYC?"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_abc123",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"location\":\"NYC\"}"
					}
				}]
			},
			{
				"role": "tool",
				"tool_call_id": "call_abc123",
				"content": "72°F, sunny"
			}
		]
	}`

	parser := &ChatCompletionsParser{}
	prog, err := parser.ParseRequest([]byte(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	t.Logf("Disassembly:\n%s", prog.Disasm())

	emitter := &ChatCompletionsEmitter{}
	out, err := emitter.EmitRequest(prog)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)

	var messages []map[string]any
	json.Unmarshal(result["messages"], &messages)
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}

	// Tool message should have tool_call_id
	toolMsg := messages[2]
	if toolMsg["role"] != "tool" {
		t.Errorf("tool message role: got %v", toolMsg["role"])
	}
	if toolMsg["tool_call_id"] != "call_abc123" {
		t.Errorf("tool_call_id: got %v", toolMsg["tool_call_id"])
	}
}

func TestChatCompletionsToAnthropicConversion(t *testing.T) {
	input := `{
		"model": "claude-3-opus",
		"temperature": 0.5,
		"max_tokens": 2048,
		"messages": [
			{"role": "system", "content": "You are a scientist."},
			{"role": "user", "content": "Explain quantum physics."}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "search",
					"description": "Search the web",
					"parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
				}
			}
		]
	}`

	// Parse as Chat Completions
	parser := &ChatCompletionsParser{}
	prog, err := parser.ParseRequest([]byte(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	t.Logf("AIL Program:\n%s", prog.Disasm())

	// Emit as Anthropic
	emitter := &AnthropicEmitter{}
	out, err := emitter.EmitRequest(prog)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)

	// System should be top-level in Anthropic
	var system string
	json.Unmarshal(result["system"], &system)
	if system != "You are a scientist." {
		t.Errorf("anthropic system: got %q", system)
	}

	// Messages should NOT contain system message
	var messages []map[string]any
	json.Unmarshal(result["messages"], &messages)
	if len(messages) != 1 {
		t.Errorf("anthropic messages: got %d, want 1 (user only)", len(messages))
	}
	if messages[0]["role"] != "user" {
		t.Errorf("first message role: got %v, want user", messages[0]["role"])
	}

	// Tools should use input_schema (not parameters)
	var tools []map[string]any
	json.Unmarshal(result["tools"], &tools)
	if len(tools) != 1 {
		t.Fatalf("anthropic tools count: got %d, want 1", len(tools))
	}
	if tools[0]["name"] != "search" {
		t.Errorf("tool name: got %v", tools[0]["name"])
	}
	if _, ok := tools[0]["input_schema"]; !ok {
		t.Error("tool should have input_schema, not parameters")
	}

	// max_tokens should be present (required in Anthropic)
	var maxTokens float64
	json.Unmarshal(result["max_tokens"], &maxTokens)
	if int(maxTokens) != 2048 {
		t.Errorf("max_tokens: got %v, want 2048", maxTokens)
	}
}

func TestChatCompletionsToGoogleConversion(t *testing.T) {
	input := `{
		"model": "gemini-pro",
		"temperature": 0.3,
		"max_tokens": 512,
		"messages": [
			{"role": "system", "content": "Be concise."},
			{"role": "user", "content": "Hello!"}
		]
	}`

	parser := &ChatCompletionsParser{}
	prog, err := parser.ParseRequest([]byte(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	emitter := &GoogleGenAIEmitter{}
	out, err := emitter.EmitRequest(prog)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	t.Logf("Google output: %s", string(out))

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)

	// system_instruction should exist
	if _, ok := result["system_instruction"]; !ok {
		t.Error("expected system_instruction in Google output")
	}

	// contents should have user message with role "user"
	var contents []map[string]any
	json.Unmarshal(result["contents"], &contents)
	if len(contents) != 1 {
		t.Fatalf("contents: got %d, want 1", len(contents))
	}
	if contents[0]["role"] != "user" {
		t.Errorf("role: got %v, want user", contents[0]["role"])
	}

	// generation_config
	var genConfig map[string]any
	json.Unmarshal(result["generation_config"], &genConfig)
	if genConfig["temperature"] != 0.3 {
		t.Errorf("temperature: got %v, want 0.3", genConfig["temperature"])
	}
}

func TestChatCompletionsToResponsesConversion(t *testing.T) {
	input := `{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "Be helpful"},
			{"role": "user", "content": "Hello"}
		],
		"max_tokens": 100,
		"stream": true
	}`

	parser := &ChatCompletionsParser{}
	prog, err := parser.ParseRequest([]byte(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	emitter := &ResponsesEmitter{}
	out, err := emitter.EmitRequest(prog)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	t.Logf("Responses output: %s", string(out))

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)

	// instructions (from system message)
	var instructions string
	json.Unmarshal(result["instructions"], &instructions)
	if instructions != "Be helpful" {
		t.Errorf("instructions: got %q, want 'Be helpful'", instructions)
	}

	// input (from user message)
	var input2 []map[string]any
	json.Unmarshal(result["input"], &input2)
	if len(input2) != 1 {
		t.Fatalf("input: got %d, want 1", len(input2))
	}

	// max_output_tokens
	var maxOut float64
	json.Unmarshal(result["max_output_tokens"], &maxOut)
	if int(maxOut) != 100 {
		t.Errorf("max_output_tokens: got %v, want 100", maxOut)
	}
}

func TestChatCompletionsResponseParse(t *testing.T) {
	resp := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Hello! How can I help?"
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
	}`

	parser := &ChatCompletionsParser{}
	prog, err := parser.ParseResponse([]byte(resp))
	if err != nil {
		t.Fatalf("parse response: %v", err)
	}

	t.Logf("Response AIL:\n%s", prog.Disasm())

	// Emit back
	emitter := &ChatCompletionsEmitter{}
	out, err := emitter.EmitResponse(prog)
	if err != nil {
		t.Fatalf("emit response: %v", err)
	}

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)

	var id string
	json.Unmarshal(result["id"], &id)
	if id != "chatcmpl-abc123" {
		t.Errorf("id: got %q", id)
	}

	var choices []map[string]any
	json.Unmarshal(result["choices"], &choices)
	if len(choices) != 1 {
		t.Fatalf("choices: got %d", len(choices))
	}
}

func TestStreamChunkRoundTrip(t *testing.T) {
	// First chunk: role
	chunk1 := `{
		"id": "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
	}`

	// Content chunk
	chunk2 := `{
		"id": "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
	}`

	// Final chunk
	chunk3 := `{
		"id": "chatcmpl-abc",
		"model": "gpt-4o",
		"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
	}`

	parser := &ChatCompletionsParser{}
	emitter := &ChatCompletionsEmitter{}

	for i, chunk := range []string{chunk1, chunk2, chunk3} {
		prog, err := parser.ParseStreamChunk([]byte(chunk))
		if err != nil {
			t.Fatalf("chunk %d parse: %v", i, err)
		}
		t.Logf("Chunk %d AIL:\n%s", i, prog.Disasm())

		out, err := emitter.EmitStreamChunk(prog)
		if err != nil {
			t.Fatalf("chunk %d emit: %v", i, err)
		}
		t.Logf("Chunk %d output: %s", i, string(out))
	}
}

func TestConvertRequest(t *testing.T) {
	input := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "Hi"}
		]
	}`

	// Same style passthrough
	out, err := ConvertRequest([]byte(input), "openai-chat-completions", "openai-chat-completions")
	if err != nil {
		t.Fatalf("passthrough: %v", err)
	}

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)
	var model string
	json.Unmarshal(result["model"], &model)
	if model != "gpt-4" {
		t.Errorf("passthrough model: got %q", model)
	}

	// Cross-style conversion
	out, err = ConvertRequest([]byte(input), "openai-chat-completions", "anthropic-messages")
	if err != nil {
		t.Fatalf("convert: %v", err)
	}

	json.Unmarshal(out, &result)
	var msgs []map[string]any
	json.Unmarshal(result["messages"], &msgs)
	if len(msgs) != 1 {
		t.Errorf("anthropic messages: got %d, want 1", len(msgs))
	}
}

func TestExtDataPassthrough(t *testing.T) {
	input := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "Hi"}],
		"response_format": {"type": "json_object"},
		"seed": 42,
		"logprobs": true
	}`

	parser := &ChatCompletionsParser{}
	prog, err := parser.ParseRequest([]byte(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	// Count EXT_DATA instructions
	extCount := 0
	for _, inst := range prog.Code {
		if inst.Op == EXT_DATA {
			extCount++
		}
	}
	if extCount < 2 {
		t.Errorf("expected at least 2 EXT_DATA, got %d\n%s", extCount, prog.Disasm())
	}

	// Emit back - EXT_DATA should survive
	emitter := &ChatCompletionsEmitter{}
	out, err := emitter.EmitRequest(prog)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)
	if _, ok := result["response_format"]; !ok {
		t.Error("response_format should survive round-trip via EXT_DATA")
	}
	if _, ok := result["seed"]; !ok {
		t.Error("seed should survive round-trip via EXT_DATA")
	}
}

func TestAnthropicResponseParse(t *testing.T) {
	resp := `{
		"id": "msg_01abc",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-opus-20240229",
		"content": [
			{"type": "text", "text": "Hello! How can I help?"}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 8}
	}`

	parser := &AnthropicParser{}
	prog, err := parser.ParseResponse([]byte(resp))
	if err != nil {
		t.Fatalf("parse response: %v", err)
	}

	t.Logf("Anthropic Response AIL:\n%s", prog.Disasm())

	// Verify structure
	foundID := false
	foundText := false
	foundDone := false
	for _, inst := range prog.Code {
		if inst.Op == RESP_ID && inst.Str == "msg_01abc" {
			foundID = true
		}
		if inst.Op == TXT_CHUNK && inst.Str == "Hello! How can I help?" {
			foundText = true
		}
		if inst.Op == RESP_DONE && inst.Str == "stop" {
			foundDone = true
		}
	}
	if !foundID {
		t.Error("missing RESP_ID")
	}
	if !foundText {
		t.Error("missing text content")
	}
	if !foundDone {
		t.Error("missing RESP_DONE")
	}

	// Round-trip through Anthropic emitter
	emitter := &AnthropicEmitter{}
	out, err := emitter.EmitResponse(prog)
	if err != nil {
		t.Fatalf("emit response: %v", err)
	}

	var result map[string]any
	json.Unmarshal(out, &result)
	if result["id"] != "msg_01abc" {
		t.Errorf("id: got %v", result["id"])
	}
	if result["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason: got %v", result["stop_reason"])
	}
}

func TestAnthropicResponseToolUse(t *testing.T) {
	resp := `{
		"id": "msg_02xyz",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-sonnet",
		"content": [
			{"type": "text", "text": "I'll check the weather."},
			{"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "NYC"}}
		],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 20, "output_tokens": 15}
	}`

	parser := &AnthropicParser{}
	prog, err := parser.ParseResponse([]byte(resp))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	t.Logf("Anthropic Tool Use Response AIL:\n%s", prog.Disasm())

	foundCall := false
	for _, inst := range prog.Code {
		if inst.Op == CALL_NAME && inst.Str == "get_weather" {
			foundCall = true
		}
	}
	if !foundCall {
		t.Error("missing tool call")
	}
}

func TestGoogleGenAIResponseParse(t *testing.T) {
	resp := `{
		"candidates": [{
			"content": {
				"parts": [{"text": "Hello from Gemini!"}],
				"role": "model"
			},
			"finishReason": "STOP",
			"index": 0
		}],
		"usageMetadata": {
			"promptTokenCount": 5,
			"candidatesTokenCount": 10,
			"totalTokenCount": 15
		},
		"modelVersion": "gemini-1.5-pro"
	}`

	parser := &GoogleGenAIParser{}
	prog, err := parser.ParseResponse([]byte(resp))
	if err != nil {
		t.Fatalf("parse response: %v", err)
	}

	t.Logf("Google Response AIL:\n%s", prog.Disasm())

	foundText := false
	foundDone := false
	for _, inst := range prog.Code {
		if inst.Op == TXT_CHUNK && inst.Str == "Hello from Gemini!" {
			foundText = true
		}
		if inst.Op == RESP_DONE && inst.Str == "stop" {
			foundDone = true
		}
	}
	if !foundText {
		t.Error("missing text content")
	}
	if !foundDone {
		t.Error("missing RESP_DONE")
	}

	// Round-trip through Google emitter
	emitter := &GoogleGenAIEmitter{}
	out, err := emitter.EmitResponse(prog)
	if err != nil {
		t.Fatalf("emit response: %v", err)
	}

	var result map[string]any
	json.Unmarshal(out, &result)
	if result["modelVersion"] != "gemini-1.5-pro" {
		t.Errorf("modelVersion: got %v", result["modelVersion"])
	}
}

func TestMediaTypePreservation(t *testing.T) {
	// Anthropic image with explicit media type
	input := `{
		"model": "claude-3",
		"max_tokens": 1024,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What is this?"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": "AAAA"}}
			]
		}]
	}`

	parser := &AnthropicParser{}
	prog, err := parser.ParseRequest([]byte(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	t.Logf("Program with media_type:\n%s", prog.Disasm())

	// Verify SET_META with media_type was emitted
	foundMediaType := false
	for _, inst := range prog.Code {
		if inst.Op == SET_META && inst.Key == "media_type" && inst.Str == "image/webp" {
			foundMediaType = true
		}
	}
	if !foundMediaType {
		t.Error("expected SET_META with media_type=image/webp")
	}

	// Emit back to Anthropic — should use the stored media type
	emitter := &AnthropicEmitter{}
	out, err := emitter.EmitRequest(prog)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	var result map[string]any
	json.Unmarshal(out, &result)

	msgs := result["messages"].([]any)
	msg := msgs[0].(map[string]any)
	content := msg["content"].([]any)

	// Find the image block
	for _, block := range content {
		b := block.(map[string]any)
		if b["type"] == "image" {
			source := b["source"].(map[string]any)
			if source["media_type"] != "image/webp" {
				t.Errorf("media_type: got %v, want image/webp", source["media_type"])
			}
		}
	}
}

func TestConverterRegistryCompleteness(t *testing.T) {
	styles := []Style{StyleChatCompletions, StyleResponses, StyleAnthropic, StyleGoogleGenAI}

	for _, style := range styles {
		if _, err := GetParser(style); err != nil {
			t.Errorf("GetParser(%s): %v", style, err)
		}
		if _, err := GetEmitter(style); err != nil {
			t.Errorf("GetEmitter(%s): %v", style, err)
		}
		if _, err := GetResponseParser(style); err != nil {
			t.Errorf("GetResponseParser(%s): %v", style, err)
		}
		if _, err := GetStreamChunkParser(style); err != nil {
			t.Errorf("GetStreamChunkParser(%s): %v", style, err)
		}
	}

	// Response emitters (not all have them, but OpenAI+Anthropic+Google should)
	for _, style := range []Style{StyleChatCompletions, StyleAnthropic, StyleGoogleGenAI} {
		if _, err := GetResponseEmitter(style); err != nil {
			t.Errorf("GetResponseEmitter(%s): %v", style, err)
		}
		if _, err := GetStreamChunkEmitter(style); err != nil {
			t.Errorf("GetStreamChunkEmitter(%s): %v", style, err)
		}
	}
}
