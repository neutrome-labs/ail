package ail

import (
	"encoding/json"
	"fmt"
)

func (p *GoogleGenAIParser) ParseResponse(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse google genai response: %w", err)
	}

	prog := NewProgram()

	// Model version
	if modelRaw, ok := raw["modelVersion"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(RESP_MODEL, model)
		}
		delete(raw, "modelVersion")
	}

	// Usage metadata
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

	// Candidates → messages
	if candidatesRaw, ok := raw["candidates"]; ok {
		var rawCands []json.RawMessage
		if json.Unmarshal(candidatesRaw, &rawCands) == nil {
			for _, rc := range rawCands {
				var candMap map[string]json.RawMessage
				if json.Unmarshal(rc, &candMap) != nil {
					continue
				}

				prog.Emit(MSG_START)
				prog.Emit(ROLE_AST)

				if contentRaw, ok := candMap["content"]; ok {
					var content struct {
						Parts []struct {
							Text             string `json:"text,omitempty"`
							Thought          *bool  `json:"thought,omitempty"`
							ThoughtSignature string `json:"thoughtSignature,omitempty"`
							FunctionCall     *struct {
								Name string          `json:"name"`
								Args json.RawMessage `json:"args"`
							} `json:"functionCall,omitempty"`
						} `json:"parts"`
					}
					if json.Unmarshal(contentRaw, &content) == nil {
						for _, part := range content.Parts {
							if part.Thought != nil && *part.Thought {
								prog.Emit(THINK_START)
								if part.Text != "" {
									prog.EmitString(THINK_CHUNK, part.Text)
								}
								if part.ThoughtSignature != "" {
									ref := prog.AddBuffer([]byte(part.ThoughtSignature))
									prog.EmitRef(THINK_REF, ref)
								}
								prog.Emit(THINK_END)
							} else if part.Text != "" {
								prog.EmitString(TXT_CHUNK, part.Text)
							}
							if part.FunctionCall != nil {
								prog.EmitString(CALL_START, "")
								prog.EmitString(CALL_NAME, part.FunctionCall.Name)
								if len(part.FunctionCall.Args) > 0 {
									prog.EmitJSON(CALL_ARGS, part.FunctionCall.Args)
								}
								prog.Emit(CALL_END)
							}
						}
					}
					delete(candMap, "content")
				}

				if frRaw, ok := candMap["finishReason"]; ok {
					var fr string
					if json.Unmarshal(frRaw, &fr) == nil && fr != "" {
						switch fr {
						case "STOP":
							prog.EmitString(RESP_DONE, "stop")
						case "MAX_TOKENS":
							prog.EmitString(RESP_DONE, "length")
						default:
							prog.EmitString(RESP_DONE, fr)
						}
					}
					delete(candMap, "finishReason")
				}

				// Remove index (reconstructed by emitter from ordering)
				delete(candMap, "index")

				// Passthrough remaining candidate-level fields as EXT_DATA
				// inside the MSG block (e.g. safetyRatings, citationMetadata).
				for key, val := range candMap {
					prog.EmitKeyJSON(EXT_DATA, key, val)
				}

				prog.Emit(MSG_END)
			}
		}
	}
	delete(raw, "candidates")

	// Passthrough remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}
