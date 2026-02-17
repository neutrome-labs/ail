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
			inMessage = true
			parts = nil
			finishReason = ""

		case TXT_CHUNK:
			if inMessage {
				parts = append(parts, map[string]any{"text": inst.Str})
			}

		case CALL_START:
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
			result[inst.Key] = json.RawMessage(inst.JSON)

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
				candidates = append(candidates, cand)
				inMessage = false
			}
		}
	}

	if candidates != nil {
		result["candidates"] = candidates
	}

	return json.Marshal(result)
}
