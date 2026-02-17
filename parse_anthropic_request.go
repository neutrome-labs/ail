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
