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
		var rawItems []json.RawMessage
		if json.Unmarshal(outputRaw, &rawItems) == nil {
			for _, ri := range rawItems {
				var itemMap map[string]json.RawMessage
				if json.Unmarshal(ri, &itemMap) != nil {
					continue
				}

				var itemType string
				if typeRaw, ok := itemMap["type"]; ok {
					json.Unmarshal(typeRaw, &itemType)
				}

				switch itemType {
				case "message":
					prog.Emit(MSG_START)
					prog.Emit(ROLE_AST)
					// Content is an array of content parts
					if contentRaw, ok := itemMap["content"]; ok {
						var parts []struct {
							Type string `json:"type"`
							Text string `json:"text,omitempty"`
						}
						if json.Unmarshal(contentRaw, &parts) == nil {
							for _, part := range parts {
								if part.Type == "output_text" || part.Type == "text" {
									prog.EmitString(TXT_CHUNK, part.Text)
								}
							}
						}
						delete(itemMap, "content")
					}
					prog.EmitString(RESP_DONE, "stop")
					// Remaining item-level fields as EXT_DATA
					delete(itemMap, "type")
					delete(itemMap, "role")
					delete(itemMap, "status")
					for key, val := range itemMap {
						prog.EmitKeyJSON(EXT_DATA, key, val)
					}
					prog.Emit(MSG_END)

				case "function_call":
					prog.Emit(MSG_START)
					prog.Emit(ROLE_AST)
					var callID, name, arguments string
					if cidRaw, ok := itemMap["call_id"]; ok {
						json.Unmarshal(cidRaw, &callID)
						delete(itemMap, "call_id")
					}
					if nameRaw, ok := itemMap["name"]; ok {
						json.Unmarshal(nameRaw, &name)
						delete(itemMap, "name")
					}
					if argsRaw, ok := itemMap["arguments"]; ok {
						json.Unmarshal(argsRaw, &arguments)
						delete(itemMap, "arguments")
					}
					prog.EmitString(CALL_START, callID)
					prog.EmitString(CALL_NAME, name)
					if arguments != "" {
						prog.EmitJSON(CALL_ARGS, json.RawMessage(arguments))
					}
					// Remaining item-level fields as EXT_DATA inside CALL
					delete(itemMap, "type")
					delete(itemMap, "status")
					delete(itemMap, "id")
					for key, val := range itemMap {
						prog.EmitKeyJSON(EXT_DATA, key, val)
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
