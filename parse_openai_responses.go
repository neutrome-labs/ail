package ail

import (
	"encoding/json"
	"fmt"
)

// ─── OpenAI Responses API Parser ─────────────────────────────────────────────

// ResponsesParser parses OpenAI Responses API JSON into AIL.
type ResponsesParser struct{}

func (p *ResponsesParser) ParseRequest(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse responses request: %w", err)
	}

	prog := NewProgram()

	// Model
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(SET_MODEL, model)
		}
		delete(raw, "model")
	}

	// Temperature
	if tempRaw, ok := raw["temperature"]; ok {
		var temp float64
		if json.Unmarshal(tempRaw, &temp) == nil {
			prog.EmitFloat(SET_TEMP, temp)
		}
		delete(raw, "temperature")
	}

	// top_p
	if tpRaw, ok := raw["top_p"]; ok {
		var tp float64
		if json.Unmarshal(tpRaw, &tp) == nil {
			prog.EmitFloat(SET_TOPP, tp)
		}
		delete(raw, "top_p")
	}

	// max_output_tokens
	if maxRaw, ok := raw["max_output_tokens"]; ok {
		var max int32
		if json.Unmarshal(maxRaw, &max) == nil {
			prog.EmitInt(SET_MAX, max)
		}
		delete(raw, "max_output_tokens")
	}

	// Stream
	if streamRaw, ok := raw["stream"]; ok {
		var stream bool
		if json.Unmarshal(streamRaw, &stream) == nil && stream {
			prog.Emit(SET_STREAM)
		}
		delete(raw, "stream")
	}

	// Instructions → system message
	if instrRaw, ok := raw["instructions"]; ok {
		var instructions string
		if json.Unmarshal(instrRaw, &instructions) == nil && instructions != "" {
			prog.Emit(MSG_START)
			prog.Emit(ROLE_SYS)
			prog.EmitString(TXT_CHUNK, instructions)
			prog.Emit(MSG_END)
		}
		delete(raw, "instructions")
	}

	// Tools
	if toolsRaw, ok := raw["tools"]; ok {
		var tools []struct {
			Type        string          `json:"type"`
			Name        string          `json:"name,omitempty"`
			Description string          `json:"description,omitempty"`
			Parameters  json.RawMessage `json:"parameters,omitempty"`
		}
		if json.Unmarshal(toolsRaw, &tools) == nil {
			prog.Emit(DEF_START)
			for _, tool := range tools {
				if tool.Name != "" {
					prog.EmitString(DEF_NAME, tool.Name)
				}
				if tool.Description != "" {
					prog.EmitString(DEF_DESC, tool.Description)
				}
				if len(tool.Parameters) > 0 {
					prog.EmitJSON(DEF_SCHEMA, tool.Parameters)
				}
			}
			prog.Emit(DEF_END)
		}
		delete(raw, "tools")
	}

	// Input → messages
	if inputRaw, ok := raw["input"]; ok {
		// Input can be string, or array of messages
		var inputStr string
		if json.Unmarshal(inputRaw, &inputStr) == nil {
			prog.Emit(MSG_START)
			prog.Emit(ROLE_USR)
			prog.EmitString(TXT_CHUNK, inputStr)
			prog.Emit(MSG_END)
		} else {
			// Array of message objects
			var inputMsgs []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			}
			if json.Unmarshal(inputRaw, &inputMsgs) == nil {
				for _, msg := range inputMsgs {
					prog.Emit(MSG_START)
					switch msg.Role {
					case "system", "developer":
						prog.Emit(ROLE_SYS)
					case "user":
						prog.Emit(ROLE_USR)
					case "assistant":
						prog.Emit(ROLE_AST)
					}
					if msg.Content != nil {
						var contentStr string
						if json.Unmarshal(msg.Content, &contentStr) == nil {
							prog.EmitString(TXT_CHUNK, contentStr)
						}
					}
					prog.Emit(MSG_END)
				}
			}
		}
		delete(raw, "input")
	}

	// Remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}

// ParseResponse parses an OpenAI Responses API response into AIL.
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
	}

	// Model
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(RESP_MODEL, model)
		}
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

	return prog, nil
}

// ParseStreamChunk parses an OpenAI Responses API streaming event into AIL.
// Responses API uses typed events rather than uniform chunks.
func (p *ResponsesParser) ParseStreamChunk(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse responses stream chunk: %w", err)
	}

	prog := NewProgram()

	eventType := ""
	if typeRaw, ok := raw["type"]; ok {
		json.Unmarshal(typeRaw, &eventType)
	}

	switch eventType {
	case "response.created", "response.in_progress":
		prog.Emit(STREAM_START)
		// Extract response ID if present
		if respRaw, ok := raw["response"]; ok {
			var resp struct {
				ID    string `json:"id"`
				Model string `json:"model"`
			}
			if json.Unmarshal(respRaw, &resp) == nil {
				if resp.ID != "" {
					prog.EmitString(RESP_ID, resp.ID)
				}
				if resp.Model != "" {
					prog.EmitString(RESP_MODEL, resp.Model)
				}
			}
		}

	case "response.output_text.delta":
		delta := ""
		if deltaRaw, ok := raw["delta"]; ok {
			json.Unmarshal(deltaRaw, &delta)
		}
		if delta != "" {
			prog.EmitString(STREAM_DELTA, delta)
		}

	case "response.function_call_arguments.delta":
		delta := ""
		if deltaRaw, ok := raw["delta"]; ok {
			json.Unmarshal(deltaRaw, &delta)
		}
		outputIndex := 0
		if idxRaw, ok := raw["output_index"]; ok {
			json.Unmarshal(idxRaw, &outputIndex)
		}
		itemID := ""
		if idRaw, ok := raw["item_id"]; ok {
			json.Unmarshal(idRaw, &itemID)
		}
		toolDelta := map[string]any{
			"index":     outputIndex,
			"arguments": delta,
		}
		if itemID != "" {
			toolDelta["id"] = itemID
		}
		j, _ := json.Marshal(toolDelta)
		prog.EmitJSON(STREAM_TOOL_DELTA, j)

	case "response.output_item.added":
		// New output item (message or function call)
		if itemRaw, ok := raw["item"]; ok {
			var item struct {
				Type   string `json:"type"`
				ID     string `json:"id"`
				CallID string `json:"call_id,omitempty"`
				Name   string `json:"name,omitempty"`
			}
			if json.Unmarshal(itemRaw, &item) == nil {
				if item.Type == "function_call" {
					td := map[string]any{"index": 0, "id": item.CallID, "name": item.Name}
					j, _ := json.Marshal(td)
					prog.EmitJSON(STREAM_TOOL_DELTA, j)
				}
			}
		}

	case "response.output_item.done":
		if itemRaw, ok := raw["item"]; ok {
			var item struct {
				Type   string `json:"type"`
				Status string `json:"status"`
			}
			if json.Unmarshal(itemRaw, &item) == nil {
				if item.Status == "completed" {
					switch item.Type {
					case "message":
						prog.EmitString(RESP_DONE, "stop")
					case "function_call":
						prog.EmitString(RESP_DONE, "tool_calls")
					}
				}
			}
		}

	case "response.completed", "response.done":
		if respRaw, ok := raw["response"]; ok {
			var resp struct {
				Usage *struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
					TotalTokens  int `json:"total_tokens"`
				} `json:"usage,omitempty"`
			}
			if json.Unmarshal(respRaw, &resp) == nil && resp.Usage != nil {
				stdUsage, _ := json.Marshal(map[string]int{
					"prompt_tokens":     resp.Usage.InputTokens,
					"completion_tokens": resp.Usage.OutputTokens,
					"total_tokens":      resp.Usage.TotalTokens,
				})
				prog.EmitJSON(USAGE, stdUsage)
			}
		}
		prog.Emit(STREAM_END)
	}

	return prog, nil
}
