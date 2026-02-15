package ail

import (
	"encoding/json"
	"fmt"
)

// ─── OpenAI Chat Completions Parser ──────────────────────────────────────────

// ChatCompletionsParser parses OpenAI Chat Completions JSON into AIL.
type ChatCompletionsParser struct{}

func (p *ChatCompletionsParser) ParseRequest(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse chat completions request: %w", err)
	}

	prog := NewProgram()

	// Config: model
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if err := json.Unmarshal(modelRaw, &model); err == nil {
			prog.EmitString(SET_MODEL, model)
		}
		delete(raw, "model")
	}

	// Config: temperature
	if tempRaw, ok := raw["temperature"]; ok {
		var temp float64
		if err := json.Unmarshal(tempRaw, &temp); err == nil {
			prog.EmitFloat(SET_TEMP, temp)
		}
		delete(raw, "temperature")
	}

	// Config: top_p
	if tpRaw, ok := raw["top_p"]; ok {
		var tp float64
		if err := json.Unmarshal(tpRaw, &tp); err == nil {
			prog.EmitFloat(SET_TOPP, tp)
		}
		delete(raw, "top_p")
	}

	// Config: max_tokens / max_completion_tokens
	if mtRaw, ok := raw["max_tokens"]; ok {
		var mt int32
		if err := json.Unmarshal(mtRaw, &mt); err == nil {
			prog.EmitInt(SET_MAX, mt)
		}
		delete(raw, "max_tokens")
	} else if mctRaw, ok := raw["max_completion_tokens"]; ok {
		var mct int32
		if err := json.Unmarshal(mctRaw, &mct); err == nil {
			prog.EmitInt(SET_MAX, mct)
		}
		delete(raw, "max_completion_tokens")
	}

	// Config: stop
	if stopRaw, ok := raw["stop"]; ok {
		// stop can be string or []string
		var stopStr string
		if err := json.Unmarshal(stopRaw, &stopStr); err == nil {
			prog.EmitString(SET_STOP, stopStr)
		} else {
			var stopArr []string
			if err := json.Unmarshal(stopRaw, &stopArr); err == nil {
				for _, s := range stopArr {
					prog.EmitString(SET_STOP, s)
				}
			}
		}
		delete(raw, "stop")
	}

	// Config: stream
	if streamRaw, ok := raw["stream"]; ok {
		var stream bool
		if err := json.Unmarshal(streamRaw, &stream); err == nil && stream {
			prog.Emit(SET_STREAM)
		}
		delete(raw, "stream")
	}

	// Tool definitions
	if toolsRaw, ok := raw["tools"]; ok {
		var tools []struct {
			Type     string `json:"type"`
			Function *struct {
				Name        string          `json:"name"`
				Description string          `json:"description,omitempty"`
				Parameters  json.RawMessage `json:"parameters,omitempty"`
			} `json:"function,omitempty"`
		}
		if err := json.Unmarshal(toolsRaw, &tools); err == nil {
			prog.Emit(DEF_START)
			for _, tool := range tools {
				if tool.Function != nil {
					prog.EmitString(DEF_NAME, tool.Function.Name)
					if tool.Function.Description != "" {
						prog.EmitString(DEF_DESC, tool.Function.Description)
					}
					if len(tool.Function.Parameters) > 0 {
						prog.EmitJSON(DEF_SCHEMA, tool.Function.Parameters)
					}
				}
			}
			prog.Emit(DEF_END)
		}
		delete(raw, "tools")
	}

	// Messages
	if msgsRaw, ok := raw["messages"]; ok {
		var messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			Name       string          `json:"name,omitempty"`
			ToolCallID string          `json:"tool_call_id,omitempty"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function *struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		}
		if err := json.Unmarshal(msgsRaw, &messages); err != nil {
			return nil, fmt.Errorf("ail: parse messages: %w", err)
		}

		for _, msg := range messages {
			prog.Emit(MSG_START)

			// Role
			switch msg.Role {
			case "system", "developer":
				prog.Emit(ROLE_SYS)
			case "user":
				prog.Emit(ROLE_USR)
			case "assistant":
				prog.Emit(ROLE_AST)
			case "tool":
				prog.Emit(ROLE_TOOL)
				// Tool result
				if msg.ToolCallID != "" {
					prog.EmitString(RESULT_START, msg.ToolCallID)
				}
			}

			// Content: can be string or array of content parts
			if msg.Content != nil {
				var contentStr string
				if err := json.Unmarshal(msg.Content, &contentStr); err == nil {
					// Simple string content
					if msg.Role == "tool" {
						prog.EmitString(RESULT_DATA, contentStr)
					} else {
						prog.EmitString(TXT_CHUNK, contentStr)
					}
				} else {
					// Array of content parts
					var parts []struct {
						Type     string `json:"type"`
						Text     string `json:"text,omitempty"`
						ImageURL *struct {
							URL    string `json:"url"`
							Detail string `json:"detail,omitempty"`
						} `json:"image_url,omitempty"`
						InputAudio *struct {
							Data   string `json:"data"`
							Format string `json:"format"`
						} `json:"input_audio,omitempty"`
					}
					if err2 := json.Unmarshal(msg.Content, &parts); err2 == nil {
						for _, part := range parts {
							switch part.Type {
							case "text":
								prog.EmitString(TXT_CHUNK, part.Text)
							case "image_url":
								if part.ImageURL != nil {
									// Store URL as buffer, emit reference
									ref := prog.AddBuffer([]byte(part.ImageURL.URL))
									prog.EmitRef(IMG_REF, ref)
								}
							case "input_audio":
								if part.InputAudio != nil {
									ref := prog.AddBuffer([]byte(part.InputAudio.Data))
									if part.InputAudio.Format != "" {
										prog.EmitKeyVal(SET_META, "media_type", "audio/"+part.InputAudio.Format)
									}
									prog.EmitRef(AUD_REF, ref)
								}
							}
						}
					}
				}
			}

			// Tool calls in assistant messages
			for _, tc := range msg.ToolCalls {
				prog.EmitString(CALL_START, tc.ID)
				if tc.Function != nil {
					prog.EmitString(CALL_NAME, tc.Function.Name)
					if tc.Function.Arguments != "" {
						prog.EmitJSON(CALL_ARGS, json.RawMessage(tc.Function.Arguments))
					}
				}
				prog.Emit(CALL_END)
			}

			// Close tool result
			if msg.Role == "tool" && msg.ToolCallID != "" {
				prog.Emit(RESULT_END)
			}

			prog.Emit(MSG_END)
		}
		delete(raw, "messages")
	}

	// Passthrough remaining fields as EXT_DATA
	delete(raw, "stream_options") // handled implicitly by SET_STREAM
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}

// ParseResponse parses an OpenAI Chat Completions response into AIL.
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
		prog.EmitJSON(USAGE, usageRaw)
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

	return prog, nil
}

// ParseStreamChunk parses an OpenAI Chat Completions streaming chunk into AIL.
func (p *ChatCompletionsParser) ParseStreamChunk(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse chat completions stream chunk: %w", err)
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

	// Usage (may appear in final chunk with stream_options)
	if usageRaw, ok := raw["usage"]; ok {
		prog.EmitJSON(USAGE, usageRaw)
	}

	// Choices (each with a delta)
	if choicesRaw, ok := raw["choices"]; ok {
		var choices []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Delta        *struct {
				Role      string          `json:"role,omitempty"`
				Content   json.RawMessage `json:"content,omitempty"`
				ToolCalls []struct {
					Index    int    `json:"index"`
					ID       string `json:"id,omitempty"`
					Type     string `json:"type,omitempty"`
					Function *struct {
						Name      string `json:"name,omitempty"`
						Arguments string `json:"arguments,omitempty"`
					} `json:"function,omitempty"`
				} `json:"tool_calls,omitempty"`
			} `json:"delta,omitempty"`
		}
		if json.Unmarshal(choicesRaw, &choices) == nil {
			for _, choice := range choices {
				if choice.Delta != nil {
					if choice.Delta.Role != "" {
						prog.Emit(STREAM_START)
					}
					if choice.Delta.Content != nil {
						var content string
						if json.Unmarshal(choice.Delta.Content, &content) == nil && content != "" {
							prog.EmitString(STREAM_DELTA, content)
						}
					}
					for _, tc := range choice.Delta.ToolCalls {
						delta := map[string]any{
							"index": tc.Index,
						}
						if tc.ID != "" {
							delta["id"] = tc.ID
						}
						if tc.Function != nil {
							if tc.Function.Name != "" {
								delta["name"] = tc.Function.Name
							}
							if tc.Function.Arguments != "" {
								delta["arguments"] = tc.Function.Arguments
							}
						}
						j, _ := json.Marshal(delta)
						prog.EmitJSON(STREAM_TOOL_DELTA, j)
					}
				}
				if choice.FinishReason != "" {
					prog.EmitString(RESP_DONE, choice.FinishReason)
					prog.Emit(STREAM_END)
				}
			}
		}
	}

	return prog, nil
}
