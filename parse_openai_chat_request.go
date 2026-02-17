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
