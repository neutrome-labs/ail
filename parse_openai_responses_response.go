package ail

import (
	"encoding/json"
	"fmt"
)

func (p *ResponsesParser) ParseResponse(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse responses response: %w", err)
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
		// Responses API usage has input_tokens/output_tokens/total_tokens
		// Convert to standard format
		var respUsage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		}
		if json.Unmarshal(usageRaw, &respUsage) == nil {
			stdUsage, _ := json.Marshal(map[string]int{
				"prompt_tokens":     respUsage.InputTokens,
				"completion_tokens": respUsage.OutputTokens,
				"total_tokens":      respUsage.TotalTokens,
			})
			prog.EmitJSON(USAGE, stdUsage)
		}
		delete(raw, "usage")
	}

	// Output items → messages
	if outputRaw, ok := raw["output"]; ok {
		var items []struct {
			Type      string          `json:"type"`
			ID        string          `json:"id,omitempty"`
			Status    string          `json:"status,omitempty"`
			Role      string          `json:"role,omitempty"`
			Content   json.RawMessage `json:"content,omitempty"`
			CallID    string          `json:"call_id,omitempty"`
			Name      string          `json:"name,omitempty"`
			Arguments string          `json:"arguments,omitempty"`
		}
		if json.Unmarshal(outputRaw, &items) == nil {
			for _, item := range items {
				switch item.Type {
				case "message":
					prog.Emit(MSG_START)
					prog.Emit(ROLE_AST)
					// Content is an array of content parts
					if item.Content != nil {
						var parts []struct {
							Type string `json:"type"`
							Text string `json:"text,omitempty"`
						}
						if json.Unmarshal(item.Content, &parts) == nil {
							for _, part := range parts {
								if part.Type == "output_text" || part.Type == "text" {
									prog.EmitString(TXT_CHUNK, part.Text)
								}
							}
						}
					}
					prog.EmitString(RESP_DONE, "stop")
					prog.Emit(MSG_END)

				case "function_call":
					prog.Emit(MSG_START)
					prog.Emit(ROLE_AST)
					prog.EmitString(CALL_START, item.CallID)
					prog.EmitString(CALL_NAME, item.Name)
					if item.Arguments != "" {
						prog.EmitJSON(CALL_ARGS, json.RawMessage(item.Arguments))
					}
					prog.Emit(CALL_END)
					prog.EmitString(RESP_DONE, "tool_calls")
					prog.Emit(MSG_END)
				}
			}
		}
	}
	delete(raw, "output")

	// Passthrough remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}

// ParseStreamChunk parses an OpenAI Responses API streaming event into AIL.
