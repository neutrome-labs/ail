package ail

import (
	"encoding/json"
)

func (e *GoogleGenAIEmitter) EmitResponse(prog *Program) ([]byte, error) {
	result := make(map[string]any)

	var candidates []map[string]any
	var parts []any
	inMessage := false
	var finishReason string
	ec := NewExtrasCollector()

	// Thinking block state
	inThinking := false
	var thinkingText string
	var thinkingSig string

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_MODEL:
			result["modelVersion"] = inst.Str

		case USAGE:
			var usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}
			if json.Unmarshal(inst.JSON, &usage) == nil {
				result["usageMetadata"] = map[string]int{
					"promptTokenCount":     usage.PromptTokens,
					"candidatesTokenCount": usage.CompletionTokens,
					"totalTokenCount":      usage.TotalTokens,
				}
			}

		case MSG_START:
			ec.Push()
			inMessage = true
			parts = nil
			finishReason = ""

		case THINK_START:
			inThinking = true
			thinkingText = ""
			thinkingSig = ""
		case THINK_CHUNK:
			if inThinking {
				thinkingText += inst.Str
			}
		case THINK_REF:
			if inThinking && int(inst.Ref) < len(prog.Buffers) {
				thinkingSig = string(prog.Buffers[inst.Ref])
			}
		case THINK_END:
			if inThinking && inMessage {
				p := map[string]any{"thought": true, "text": thinkingText}
				if thinkingSig != "" {
					p["thoughtSignature"] = thinkingSig
				}
				parts = append(parts, p)
			}
			inThinking = false

		case TXT_CHUNK:
			if inMessage {
				parts = append(parts, map[string]any{"text": inst.Str})
			}

		case CALL_START:
			ec.Push()
			if inMessage {
				parts = append(parts, map[string]any{
					"functionCall": map[string]any{},
				})
			}

		case CALL_NAME:
			if len(parts) > 0 {
				last := parts[len(parts)-1].(map[string]any)
				if fc, ok := last["functionCall"].(map[string]any); ok {
					fc["name"] = inst.Str
				}
			}

		case CALL_ARGS:
			if len(parts) > 0 {
				last := parts[len(parts)-1].(map[string]any)
				if fc, ok := last["functionCall"].(map[string]any); ok {
					fc["args"] = json.RawMessage(inst.JSON)
				}
			}

		case CALL_END:
			if len(parts) > 0 {
				last := parts[len(parts)-1].(map[string]any)
				if fc, ok := last["functionCall"].(map[string]any); ok {
					ec.MergeInto(fc)
				}
			}
			ec.Pop()

		case RESP_DONE:
			switch inst.Str {
			case "stop":
				finishReason = "STOP"
			case "length":
				finishReason = "MAX_TOKENS"
			default:
				finishReason = inst.Str
			}

		case EXT_DATA:
			ec.AddJSON(inst.Key, inst.JSON)

		case SET_META:
			if inst.Key != "media_type" {
				ec.AddString(inst.Key, inst.Str)
			}

		case MSG_END:
			if inMessage {
				cand := map[string]any{
					"content": map[string]any{
						"role":  "model",
						"parts": parts,
					},
					"index": len(candidates),
				}
				if finishReason != "" {
					cand["finishReason"] = finishReason
				}
				ec.MergeInto(cand)
				candidates = append(candidates, cand)
				inMessage = false
			}
			ec.Pop()
		}
	}

	if candidates != nil {
		result["candidates"] = candidates
	}

	ec.MergeInto(result)
	return json.Marshal(result)
}
