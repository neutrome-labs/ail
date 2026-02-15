package ail

import (
	"encoding/json"
	"fmt"
)

// ─── Google GenAI Parser ─────────────────────────────────────────────────────

// GoogleGenAIParser parses Google GenAI JSON into AIL.
type GoogleGenAIParser struct{}

func (p *GoogleGenAIParser) ParseRequest(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse google genai request: %w", err)
	}

	prog := NewProgram()

	// Model (in Google this is typically a URL param, but may be in body)
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(SET_MODEL, model)
		}
		delete(raw, "model")
	}

	// generationConfig
	if gcRaw, ok := raw["generationConfig"]; ok {
		var gc struct {
			Temperature     *float64 `json:"temperature,omitempty"`
			TopP            *float64 `json:"topP,omitempty"`
			MaxOutputTokens *int32   `json:"maxOutputTokens,omitempty"`
			StopSequences   []string `json:"stopSequences,omitempty"`
		}
		if json.Unmarshal(gcRaw, &gc) == nil {
			if gc.Temperature != nil {
				prog.EmitFloat(SET_TEMP, *gc.Temperature)
			}
			if gc.TopP != nil {
				prog.EmitFloat(SET_TOPP, *gc.TopP)
			}
			if gc.MaxOutputTokens != nil {
				prog.EmitInt(SET_MAX, *gc.MaxOutputTokens)
			}
			for _, s := range gc.StopSequences {
				prog.EmitString(SET_STOP, s)
			}
		}
		delete(raw, "generationConfig")
	}

	// system_instruction
	if sysRaw, ok := raw["system_instruction"]; ok {
		var sysParts struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		}
		if json.Unmarshal(sysRaw, &sysParts) == nil {
			for _, part := range sysParts.Parts {
				prog.Emit(MSG_START)
				prog.Emit(ROLE_SYS)
				prog.EmitString(TXT_CHUNK, part.Text)
				prog.Emit(MSG_END)
			}
		}
		delete(raw, "system_instruction")
	}

	// Tools
	if toolsRaw, ok := raw["tools"]; ok {
		var toolSets []struct {
			FunctionDeclarations []struct {
				Name        string          `json:"name"`
				Description string          `json:"description,omitempty"`
				Parameters  json.RawMessage `json:"parameters,omitempty"`
			} `json:"function_declarations,omitempty"`
		}
		if json.Unmarshal(toolsRaw, &toolSets) == nil {
			prog.Emit(DEF_START)
			for _, ts := range toolSets {
				for _, fd := range ts.FunctionDeclarations {
					prog.EmitString(DEF_NAME, fd.Name)
					if fd.Description != "" {
						prog.EmitString(DEF_DESC, fd.Description)
					}
					if len(fd.Parameters) > 0 {
						prog.EmitJSON(DEF_SCHEMA, fd.Parameters)
					}
				}
			}
			prog.Emit(DEF_END)
		}
		delete(raw, "tools")
	}

	// Contents (messages)
	if contentsRaw, ok := raw["contents"]; ok {
		var contents []struct {
			Role  string `json:"role"`
			Parts []struct {
				Text         string `json:"text,omitempty"`
				FunctionCall *struct {
					Name string          `json:"name"`
					Args json.RawMessage `json:"args"`
				} `json:"functionCall,omitempty"`
				FunctionResponse *struct {
					Name     string          `json:"name"`
					Response json.RawMessage `json:"response"`
				} `json:"functionResponse,omitempty"`
				InlineData *struct {
					MimeType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData,omitempty"`
			} `json:"parts"`
		}
		if json.Unmarshal(contentsRaw, &contents) == nil {
			for _, content := range contents {
				prog.Emit(MSG_START)

				switch content.Role {
				case "user":
					prog.Emit(ROLE_USR)
				case "model":
					prog.Emit(ROLE_AST)
				case "function":
					prog.Emit(ROLE_TOOL)
				}

				for _, part := range content.Parts {
					if part.Text != "" {
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
					if part.FunctionResponse != nil {
						prog.EmitString(RESULT_START, part.FunctionResponse.Name)
						prog.EmitString(RESULT_DATA, string(part.FunctionResponse.Response))
						prog.Emit(RESULT_END)
					}
					if part.InlineData != nil {
						ref := prog.AddBuffer([]byte(part.InlineData.Data))
						if part.InlineData.MimeType != "" {
							prog.EmitKeyVal(SET_META, "media_type", part.InlineData.MimeType)
						}
						if isAudioMime(part.InlineData.MimeType) {
							prog.EmitRef(AUD_REF, ref)
						} else {
							prog.EmitRef(IMG_REF, ref)
						}
					}
				}

				prog.Emit(MSG_END)
			}
		}
		delete(raw, "contents")
	}

	// Remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}

func isAudioMime(mime string) bool {
	return len(mime) > 6 && mime[:6] == "audio/"
}

// ParseResponse parses a Google GenAI response into AIL.
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
	}

	// Candidates → messages
	if candidatesRaw, ok := raw["candidates"]; ok {
		var candidates []struct {
			Content *struct {
				Role  string `json:"role"`
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
				prog.Emit(MSG_START)
				prog.Emit(ROLE_AST)

				if cand.Content != nil {
					for _, part := range cand.Content.Parts {
						if part.Text != "" {
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

				if cand.FinishReason != "" {
					switch cand.FinishReason {
					case "STOP":
						prog.EmitString(RESP_DONE, "stop")
					case "MAX_TOKENS":
						prog.EmitString(RESP_DONE, "length")
					default:
						prog.EmitString(RESP_DONE, cand.FinishReason)
					}
				}

				prog.Emit(MSG_END)
			}
		}
	}

	return prog, nil
}

// ParseStreamChunk parses a Google GenAI streaming chunk into AIL.
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
	}

	return prog, nil
}
