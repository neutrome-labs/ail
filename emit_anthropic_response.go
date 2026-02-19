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
	ec := NewExtrasCollector()
	inMessage := false

	// Thinking block state
	inThinking := false
	var thinkingText string
	var thinkingSignature string

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
			ec.Push()
			inMessage = true
			contentBlocks = nil
			textContent = ""

		case TXT_CHUNK:
			if inMessage {
				textContent += inst.Str
			}

		case THINK_START:
			inThinking = true
			thinkingText = ""
			thinkingSignature = ""

		case THINK_CHUNK:
			if inThinking {
				thinkingText += inst.Str
			}

		case THINK_REF:
			if inThinking && int(inst.Ref) < len(prog.Buffers) {
				thinkingSignature = string(prog.Buffers[inst.Ref])
			}

		case THINK_END:
			if inThinking && inMessage {
				if textContent != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": textContent,
					})
					textContent = ""
				}
				block := map[string]any{
					"type":     "thinking",
					"thinking": thinkingText,
				}
				if thinkingSignature != "" {
					block["signature"] = thinkingSignature
				}
				contentBlocks = append(contentBlocks, block)
			}
			inThinking = false

		case CALL_START:
			ec.Push()
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

		case CALL_END:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					ec.MergeInto(last)
				}
			}
			ec.Pop()

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
			ec.AddJSON(inst.Key, inst.JSON)

		case SET_META:
			if inst.Key != "media_type" {
				ec.AddString(inst.Key, inst.Str)
			}

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
			// Anthropic is flat — MSG-level extras go to result directly
			ec.MergeInto(result)
			ec.Pop()
		}
	}

	if len(contentBlocks) > 0 {
		result["content"] = contentBlocks
	} else {
		result["content"] = []any{}
	}

	ec.MergeInto(result)
	return json.Marshal(result)
}
