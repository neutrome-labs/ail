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
