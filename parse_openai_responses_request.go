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
		var rawTools []json.RawMessage
		if json.Unmarshal(toolsRaw, &rawTools) == nil && len(rawTools) > 0 {
			prog.Emit(DEF_START)
			for _, rt := range rawTools {
				var toolMap map[string]json.RawMessage
				if json.Unmarshal(rt, &toolMap) != nil {
					continue
				}

				if nameRaw, ok := toolMap["name"]; ok {
					var name string
					if json.Unmarshal(nameRaw, &name) == nil && name != "" {
						prog.EmitString(DEF_NAME, name)
					}
					delete(toolMap, "name")
				}
				if descRaw, ok := toolMap["description"]; ok {
					var desc string
					if json.Unmarshal(descRaw, &desc) == nil && desc != "" {
						prog.EmitString(DEF_DESC, desc)
					}
					delete(toolMap, "description")
				}
				if paramsRaw, ok := toolMap["parameters"]; ok {
					prog.EmitJSON(DEF_SCHEMA, paramsRaw)
					delete(toolMap, "parameters")
				}
				delete(toolMap, "type") // always "function", reconstructed by emitter

				// Remaining tool-level fields as EXT_DATA (e.g., strict)
				for key, val := range toolMap {
					prog.EmitKeyJSON(EXT_DATA, key, val)
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
			var rawMsgs []json.RawMessage
			if json.Unmarshal(inputRaw, &rawMsgs) == nil {
				for _, rm := range rawMsgs {
					var msgMap map[string]json.RawMessage
					if json.Unmarshal(rm, &msgMap) != nil {
						continue
					}

					prog.Emit(MSG_START)

					if roleRaw, ok := msgMap["role"]; ok {
						var role string
						if json.Unmarshal(roleRaw, &role) == nil {
							switch role {
							case "system", "developer":
								prog.Emit(ROLE_SYS)
							case "user":
								prog.Emit(ROLE_USR)
							case "assistant":
								prog.Emit(ROLE_AST)
							}
						}
						delete(msgMap, "role")
					}

					if contentRaw, ok := msgMap["content"]; ok {
						var contentStr string
						if json.Unmarshal(contentRaw, &contentStr) == nil {
							prog.EmitString(TXT_CHUNK, contentStr)
						}
						delete(msgMap, "content")
					}

					// Remaining per-message fields as EXT_DATA
					for key, val := range msgMap {
						prog.EmitKeyJSON(EXT_DATA, key, val)
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
