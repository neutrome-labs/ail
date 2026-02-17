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

	// generation_config
	if gcRaw, ok := raw["generation_config"]; ok {
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
		delete(raw, "generation_config")
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
			} `json:"functionDeclarations,omitempty"`
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
