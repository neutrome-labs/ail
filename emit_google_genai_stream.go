package ail

import (
	"encoding/json"
)

func (e *GoogleGenAIEmitter) EmitStreamChunk(prog *Program) ([]byte, error) {
	result := make(map[string]any)
	ec := NewExtrasCollector()

	var parts []any
	var finishReason string

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_MODEL:
			result["modelVersion"] = inst.Str

		case STREAM_DELTA:
			parts = append(parts, map[string]any{"text": inst.Str})

		case STREAM_TOOL_DELTA:
			var td map[string]any
			if json.Unmarshal(inst.JSON, &td) == nil {
				fc := map[string]any{}
				if name, ok := td["name"]; ok {
					fc["name"] = name
				}
				if args, ok := td["arguments"]; ok {
					fc["args"] = json.RawMessage(args.(string))
				}
				parts = append(parts, map[string]any{"functionCall": fc})
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

		case EXT_DATA:
			ec.AddJSON(inst.Key, inst.JSON)

		case SET_META:
			if inst.Key != "media_type" {
				ec.AddString(inst.Key, inst.Str)
			}
		}
	}

	cand := map[string]any{"index": 0}
	if len(parts) > 0 {
		cand["content"] = map[string]any{
			"role":  "model",
			"parts": parts,
		}
	}
	if finishReason != "" {
		cand["finishReason"] = finishReason
	}
	result["candidates"] = []any{cand}

	ec.MergeInto(result)
	return json.Marshal(result)
}
