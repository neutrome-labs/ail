package ail

import (
	"encoding/json"
)

func (e *ChatCompletionsEmitter) EmitResponse(prog *Program) ([]byte, error) {
	result := map[string]any{
		"object": "chat.completion",
	}

	var choices []map[string]any
	var currentChoice map[string]any
	var currentMessage map[string]any
	var textContent string
	var toolCalls []map[string]any
	inMessage := false

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID:
			result["id"] = inst.Str
		case RESP_MODEL:
			result["model"] = inst.Str
		case USAGE:
			result["usage"] = json.RawMessage(inst.JSON)

		case MSG_START:
			inMessage = true
			currentChoice = map[string]any{"index": len(choices)}
			currentMessage = make(map[string]any)
			textContent = ""
			toolCalls = nil

		case ROLE_AST:
			if inMessage {
				currentMessage["role"] = "assistant"
			}

		case TXT_CHUNK:
			if inMessage {
				textContent += inst.Str
			}

		case CALL_START:
			tc := map[string]any{
				"id":   inst.Str,
				"type": "function",
			}
			toolCalls = append(toolCalls, tc)

		case CALL_NAME:
			if len(toolCalls) > 0 {
				last := toolCalls[len(toolCalls)-1]
				fn, _ := last["function"].(map[string]any)
				if fn == nil {
					fn = make(map[string]any)
				}
				fn["name"] = inst.Str
				last["function"] = fn
			}

		case CALL_ARGS:
			if len(toolCalls) > 0 {
				last := toolCalls[len(toolCalls)-1]
				fn, _ := last["function"].(map[string]any)
				if fn == nil {
					fn = make(map[string]any)
				}
				fn["arguments"] = string(inst.JSON)
				last["function"] = fn
			}

		case CALL_END:
			// already tracked

		case RESP_DONE:
			if currentChoice != nil {
				currentChoice["finish_reason"] = inst.Str
			}

		case EXT_DATA:
			result[inst.Key] = json.RawMessage(inst.JSON)

		case MSG_END:
			if inMessage && currentChoice != nil {
				if textContent != "" {
					currentMessage["content"] = textContent
				}
				if len(toolCalls) > 0 {
					currentMessage["tool_calls"] = toolCalls
				}
				currentChoice["message"] = currentMessage
				choices = append(choices, currentChoice)
				inMessage = false
			}
		}
	}

	if choices != nil {
		result["choices"] = choices
	}

	return json.Marshal(result)
}
