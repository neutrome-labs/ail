package ail

import (
	"encoding/json"
)

func (e *AnthropicEmitter) EmitResponse(prog *Program) ([]byte, error) {
	result := map[string]any{
		"type": "message",
		"role": "assistant",
	}

	var contentBlocks []any
	var textContent string
	inMessage := false

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID:
			result["id"] = inst.Str
		case RESP_MODEL:
			result["model"] = inst.Str
		case USAGE:
			// Convert standard usage to Anthropic format
			var usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}
			if json.Unmarshal(inst.JSON, &usage) == nil {
				result["usage"] = map[string]int{
					"input_tokens":  usage.PromptTokens,
					"output_tokens": usage.CompletionTokens,
				}
			}

		case MSG_START:
			inMessage = true
			contentBlocks = nil
			textContent = ""

		case TXT_CHUNK:
			if inMessage {
				textContent += inst.Str
			}

		case CALL_START:
			if inMessage {
				if textContent != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": textContent,
					})
					textContent = ""
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "tool_use",
					"id":   inst.Str,
				})
			}

		case CALL_NAME:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					last["name"] = inst.Str
				}
			}

		case CALL_ARGS:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					last["input"] = json.RawMessage(inst.JSON)
				}
			}

		case RESP_DONE:
			switch inst.Str {
			case "stop":
				result["stop_reason"] = "end_turn"
			case "tool_calls":
				result["stop_reason"] = "tool_use"
			case "length":
				result["stop_reason"] = "max_tokens"
			default:
				result["stop_reason"] = inst.Str
			}

		case EXT_DATA:
			result[inst.Key] = json.RawMessage(inst.JSON)

		case MSG_END:
			if inMessage {
				if textContent != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": textContent,
					})
				}
				inMessage = false
			}
		}
	}

	if len(contentBlocks) > 0 {
		result["content"] = contentBlocks
	} else {
		result["content"] = []any{}
	}

	return json.Marshal(result)
}
