package ail

import (
	"encoding/json"
)

func (e *ChatCompletionsEmitter) EmitStreamChunk(prog *Program) ([]byte, error) {
	result := map[string]any{
		"object": "chat.completion.chunk",
	}

	var choices []map[string]any
	var delta map[string]any

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID:
			result["id"] = inst.Str
		case RESP_MODEL:
			result["model"] = inst.Str
		case USAGE:
			result["usage"] = json.RawMessage(inst.JSON)

		case STREAM_START:
			delta = make(map[string]any)
			delta["role"] = "assistant"
			choices = append(choices, map[string]any{
				"index": 0,
				"delta": delta,
			})

		case STREAM_DELTA:
			if delta == nil {
				delta = make(map[string]any)
				choices = append(choices, map[string]any{
					"index": 0,
					"delta": delta,
				})
			}
			delta["content"] = inst.Str

		case STREAM_TOOL_DELTA:
			if delta == nil {
				delta = make(map[string]any)
				choices = append(choices, map[string]any{
					"index": 0,
					"delta": delta,
				})
			}
			var toolDelta map[string]any
			if err := json.Unmarshal(inst.JSON, &toolDelta); err == nil {
				// Reconstruct tool_calls array in delta
				tc := map[string]any{
					"index": toolDelta["index"],
					"type":  "function",
				}
				if id, ok := toolDelta["id"]; ok {
					tc["id"] = id
				}
				fn := make(map[string]any)
				if name, ok := toolDelta["name"]; ok {
					fn["name"] = name
				}
				if args, ok := toolDelta["arguments"]; ok {
					fn["arguments"] = args
				}
				if len(fn) > 0 {
					tc["function"] = fn
				}
				delta["tool_calls"] = []any{tc}
			}

		case RESP_DONE:
			choice := map[string]any{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": inst.Str,
			}
			choices = append(choices, choice)

		case EXT_DATA:
			result[inst.Key] = json.RawMessage(inst.JSON)

		case STREAM_END:
			// end marker - no additional data
		}
	}

	if choices != nil {
		result["choices"] = choices
	} else {
		// Empty chunk with just metadata
		result["choices"] = []any{}
	}

	return json.Marshal(result)
}
