package ail

import (
	"encoding/json"
	"fmt"
)

// ─── Anthropic Messages Parser ───────────────────────────────────────────────

// AnthropicParser parses Anthropic Messages API JSON into AIL.
type AnthropicParser struct{}

func (p *AnthropicParser) ParseRequest(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse anthropic request: %w", err)
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

	// max_tokens (required in Anthropic)
	if maxRaw, ok := raw["max_tokens"]; ok {
		var max int32
		if json.Unmarshal(maxRaw, &max) == nil {
			prog.EmitInt(SET_MAX, max)
		}
		delete(raw, "max_tokens")
	}

	// stop_sequences
	if stopRaw, ok := raw["stop_sequences"]; ok {
		var stops []string
		if json.Unmarshal(stopRaw, &stops) == nil {
			for _, s := range stops {
				prog.EmitString(SET_STOP, s)
			}
		}
		delete(raw, "stop_sequences")
	}

	// Stream
	if streamRaw, ok := raw["stream"]; ok {
		var stream bool
		if json.Unmarshal(streamRaw, &stream) == nil && stream {
			prog.Emit(SET_STREAM)
		}
		delete(raw, "stream")
	}

	// System (top-level in Anthropic, not in messages)
	if sysRaw, ok := raw["system"]; ok {
		var sysStr string
		if json.Unmarshal(sysRaw, &sysStr) == nil && sysStr != "" {
			prog.Emit(MSG_START)
			prog.Emit(ROLE_SYS)
			prog.EmitString(TXT_CHUNK, sysStr)
			prog.Emit(MSG_END)
		}
		delete(raw, "system")
	}

	// Tools
	if toolsRaw, ok := raw["tools"]; ok {
		var tools []struct {
			Name        string          `json:"name"`
			Description string          `json:"description,omitempty"`
			InputSchema json.RawMessage `json:"input_schema,omitempty"`
		}
		if json.Unmarshal(toolsRaw, &tools) == nil {
			prog.Emit(DEF_START)
			for _, tool := range tools {
				prog.EmitString(DEF_NAME, tool.Name)
				if tool.Description != "" {
					prog.EmitString(DEF_DESC, tool.Description)
				}
				if len(tool.InputSchema) > 0 {
					prog.EmitJSON(DEF_SCHEMA, tool.InputSchema)
				}
			}
			prog.Emit(DEF_END)
		}
		delete(raw, "tools")
	}

	// Messages
	if msgsRaw, ok := raw["messages"]; ok {
		var messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(msgsRaw, &messages) == nil {
			for _, msg := range messages {
				prog.Emit(MSG_START)

				switch msg.Role {
				case "user":
					prog.Emit(ROLE_USR)
				case "assistant":
					prog.Emit(ROLE_AST)
				}

				// Content can be string or array of content blocks
				if msg.Content != nil {
					var contentStr string
					if json.Unmarshal(msg.Content, &contentStr) == nil {
						prog.EmitString(TXT_CHUNK, contentStr)
					} else {
						var blocks []struct {
							Type      string          `json:"type"`
							Text      string          `json:"text,omitempty"`
							ID        string          `json:"id,omitempty"`
							Name      string          `json:"name,omitempty"`
							Input     json.RawMessage `json:"input,omitempty"`
							ToolUseID string          `json:"tool_use_id,omitempty"`
							Content   json.RawMessage `json:"content,omitempty"`
							Source    *struct {
								Type      string `json:"type"`
								MediaType string `json:"media_type"`
								Data      string `json:"data"`
							} `json:"source,omitempty"`
						}
						if json.Unmarshal(msg.Content, &blocks) == nil {
							for _, block := range blocks {
								switch block.Type {
								case "text":
									prog.EmitString(TXT_CHUNK, block.Text)
								case "image":
									if block.Source != nil {
										ref := prog.AddBuffer([]byte(block.Source.Data))
										if block.Source.MediaType != "" {
											prog.EmitKeyVal(SET_META, "media_type", block.Source.MediaType)
										}
										prog.EmitRef(IMG_REF, ref)
									}
								case "tool_use":
									prog.EmitString(CALL_START, block.ID)
									prog.EmitString(CALL_NAME, block.Name)
									if len(block.Input) > 0 {
										prog.EmitJSON(CALL_ARGS, block.Input)
									}
									prog.Emit(CALL_END)
								case "tool_result":
									prog.EmitString(RESULT_START, block.ToolUseID)
									if block.Content != nil {
										var resultStr string
										if json.Unmarshal(block.Content, &resultStr) == nil {
											prog.EmitString(RESULT_DATA, resultStr)
										}
									}
									prog.Emit(RESULT_END)
								}
							}
						}
					}
				}

				prog.Emit(MSG_END)
			}
		}
		delete(raw, "messages")
	}

	// Remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}

// ParseResponse parses an Anthropic Messages API response into AIL.
func (p *AnthropicParser) ParseResponse(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse anthropic response: %w", err)
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
		var u struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		}
		if json.Unmarshal(usageRaw, &u) == nil {
			stdUsage, _ := json.Marshal(map[string]int{
				"prompt_tokens":     u.InputTokens,
				"completion_tokens": u.OutputTokens,
				"total_tokens":      u.InputTokens + u.OutputTokens,
			})
			prog.EmitJSON(USAGE, stdUsage)
		}
	}

	// Content → message
	prog.Emit(MSG_START)
	prog.Emit(ROLE_AST)

	if contentRaw, ok := raw["content"]; ok {
		var blocks []struct {
			Type  string          `json:"type"`
			Text  string          `json:"text,omitempty"`
			ID    string          `json:"id,omitempty"`
			Name  string          `json:"name,omitempty"`
			Input json.RawMessage `json:"input,omitempty"`
		}
		if json.Unmarshal(contentRaw, &blocks) == nil {
			for _, block := range blocks {
				switch block.Type {
				case "text":
					prog.EmitString(TXT_CHUNK, block.Text)
				case "tool_use":
					prog.EmitString(CALL_START, block.ID)
					prog.EmitString(CALL_NAME, block.Name)
					if len(block.Input) > 0 {
						prog.EmitJSON(CALL_ARGS, block.Input)
					}
					prog.Emit(CALL_END)
				}
			}
		}
	}

	// Stop reason → finish reason
	if srRaw, ok := raw["stop_reason"]; ok {
		var sr string
		if json.Unmarshal(srRaw, &sr) == nil {
			switch sr {
			case "end_turn":
				prog.EmitString(RESP_DONE, "stop")
			case "tool_use":
				prog.EmitString(RESP_DONE, "tool_calls")
			case "max_tokens":
				prog.EmitString(RESP_DONE, "length")
			default:
				prog.EmitString(RESP_DONE, sr)
			}
		}
	}

	prog.Emit(MSG_END)

	return prog, nil
}

// ParseStreamChunk parses an Anthropic streaming event into AIL.
func (p *AnthropicParser) ParseStreamChunk(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse anthropic stream event: %w", err)
	}

	prog := NewProgram()

	eventType := ""
	if typeRaw, ok := raw["type"]; ok {
		json.Unmarshal(typeRaw, &eventType)
	}

	switch eventType {
	case "message_start":
		prog.Emit(STREAM_START)
		if msgRaw, ok := raw["message"]; ok {
			var msg struct {
				ID    string `json:"id"`
				Model string `json:"model"`
			}
			if json.Unmarshal(msgRaw, &msg) == nil {
				if msg.ID != "" {
					prog.EmitString(RESP_ID, msg.ID)
				}
				if msg.Model != "" {
					prog.EmitString(RESP_MODEL, msg.Model)
				}
			}
		}

	case "content_block_start":
		if cbRaw, ok := raw["content_block"]; ok {
			var cb struct {
				Type string `json:"type"`
				ID   string `json:"id,omitempty"`
				Name string `json:"name,omitempty"`
			}
			if json.Unmarshal(cbRaw, &cb) == nil && cb.Type == "tool_use" {
				idx := 0
				if idxRaw, ok := raw["index"]; ok {
					json.Unmarshal(idxRaw, &idx)
				}
				td := map[string]any{"index": idx, "id": cb.ID, "name": cb.Name}
				j, _ := json.Marshal(td)
				prog.EmitJSON(STREAM_TOOL_DELTA, j)
			}
		}

	case "content_block_delta":
		if deltaRaw, ok := raw["delta"]; ok {
			var delta struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				PartialJSON string `json:"partial_json,omitempty"`
			}
			if json.Unmarshal(deltaRaw, &delta) == nil {
				switch delta.Type {
				case "text_delta":
					prog.EmitString(STREAM_DELTA, delta.Text)
				case "input_json_delta":
					idx := 0
					if idxRaw, ok := raw["index"]; ok {
						json.Unmarshal(idxRaw, &idx)
					}
					td := map[string]any{"index": idx, "arguments": delta.PartialJSON}
					j, _ := json.Marshal(td)
					prog.EmitJSON(STREAM_TOOL_DELTA, j)
				}
			}
		}

	case "message_delta":
		if deltaRaw, ok := raw["delta"]; ok {
			var delta struct {
				StopReason string `json:"stop_reason,omitempty"`
			}
			if json.Unmarshal(deltaRaw, &delta) == nil && delta.StopReason != "" {
				switch delta.StopReason {
				case "end_turn":
					prog.EmitString(RESP_DONE, "stop")
				case "tool_use":
					prog.EmitString(RESP_DONE, "tool_calls")
				case "max_tokens":
					prog.EmitString(RESP_DONE, "length")
				default:
					prog.EmitString(RESP_DONE, delta.StopReason)
				}
			}
		}
		if usageRaw, ok := raw["usage"]; ok {
			var u struct {
				OutputTokens int `json:"output_tokens"`
			}
			if json.Unmarshal(usageRaw, &u) == nil {
				stdUsage, _ := json.Marshal(map[string]int{
					"completion_tokens": u.OutputTokens,
				})
				prog.EmitJSON(USAGE, stdUsage)
			}
		}

	case "message_stop":
		prog.Emit(STREAM_END)
	}

	return prog, nil
}
