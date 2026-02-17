package ail

import (
	"encoding/json"
	"fmt"
)

func (p *GoogleGenAIParser) ParseStreamChunk(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse google genai stream chunk: %w", err)
	}

	prog := NewProgram()

	// Model version
	if modelRaw, ok := raw["modelVersion"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(RESP_MODEL, model)
		}
	}

	// Candidates
	if candidatesRaw, ok := raw["candidates"]; ok {
		var candidates []struct {
			Content *struct {
				Parts []struct {
					Text         string `json:"text,omitempty"`
					FunctionCall *struct {
						Name string          `json:"name"`
						Args json.RawMessage `json:"args"`
					} `json:"functionCall,omitempty"`
				} `json:"parts"`
			} `json:"content,omitempty"`
			FinishReason string `json:"finishReason,omitempty"`
		}
		if json.Unmarshal(candidatesRaw, &candidates) == nil {
			for _, cand := range candidates {
				if cand.Content != nil {
					for _, part := range cand.Content.Parts {
						if part.Text != "" {
							prog.EmitString(STREAM_DELTA, part.Text)
						}
						if part.FunctionCall != nil {
							td := map[string]any{
								"index": 0,
								"name":  part.FunctionCall.Name,
							}
							if len(part.FunctionCall.Args) > 0 {
								td["arguments"] = string(part.FunctionCall.Args)
							}
							j, _ := json.Marshal(td)
							prog.EmitJSON(STREAM_TOOL_DELTA, j)
						}
					}
				}
				if cand.FinishReason != "" {
					switch cand.FinishReason {
					case "STOP":
						prog.EmitString(RESP_DONE, "stop")
					case "MAX_TOKENS":
						prog.EmitString(RESP_DONE, "length")
					default:
						prog.EmitString(RESP_DONE, cand.FinishReason)
					}
					prog.Emit(STREAM_END)
				}
			}
		}
	}

	// Usage in final chunk
	if usageRaw, ok := raw["usageMetadata"]; ok {
		var u struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		}
		if json.Unmarshal(usageRaw, &u) == nil {
			stdUsage, _ := json.Marshal(map[string]int{
				"prompt_tokens":     u.PromptTokenCount,
				"completion_tokens": u.CandidatesTokenCount,
				"total_tokens":      u.TotalTokenCount,
			})
			prog.EmitJSON(USAGE, stdUsage)
		}
		delete(raw, "usageMetadata")
	}

	delete(raw, "candidates")

	// Passthrough remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}
