package ail

import (
	"encoding/json"
	"fmt"
)

func (p *ChatCompletionsParser) ParseResponse(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse chat completions response: %w", err)
	}

	prog := NewProgram()

	// Response ID
	if idRaw, ok := raw["id"]; ok {
		var id string
		if json.Unmarshal(idRaw, &id) == nil {
			prog.EmitString(RESP_ID, id)
		}
		delete(raw, "id")
	}

	// Model
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(RESP_MODEL, model)
		}
		delete(raw, "model")
	}

	// Usage
	if usageRaw, ok := raw["usage"]; ok {
		prog.EmitJSON(USAGE, usageRaw)
		delete(raw, "usage")
	}

	// Choices
	if choicesRaw, ok := raw["choices"]; ok {
		var choices []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Message      *struct {
				Role      string          `json:"role"`
				Content   json.RawMessage `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function *struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls,omitempty"`
			} `json:"message,omitempty"`
		}
		if err := json.Unmarshal(choicesRaw, &choices); err == nil {
			for _, choice := range choices {
				prog.Emit(MSG_START)
				if choice.Message != nil {
					switch choice.Message.Role {
					case "assistant":
						prog.Emit(ROLE_AST)
					}

					// Content
					if choice.Message.Content != nil {
						var contentStr string
						if json.Unmarshal(choice.Message.Content, &contentStr) == nil && contentStr != "" {
							prog.EmitString(TXT_CHUNK, contentStr)
						}
					}

					// Tool calls
					for _, tc := range choice.Message.ToolCalls {
						prog.EmitString(CALL_START, tc.ID)
						if tc.Function != nil {
							prog.EmitString(CALL_NAME, tc.Function.Name)
							if tc.Function.Arguments != "" {
								prog.EmitJSON(CALL_ARGS, json.RawMessage(tc.Function.Arguments))
							}
						}
						prog.Emit(CALL_END)
					}
				}

				if choice.FinishReason != "" {
					prog.EmitString(RESP_DONE, choice.FinishReason)
				}
				prog.Emit(MSG_END)
			}
		}
	}
	delete(raw, "choices")

	// Passthrough remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}
	return prog, nil
}
