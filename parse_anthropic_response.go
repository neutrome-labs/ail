package ail

import (
	"encoding/json"
	"fmt"
)

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
		delete(raw, "usage")
	}

	// Content → message
	prog.Emit(MSG_START)
	prog.Emit(ROLE_AST)

	if contentRaw, ok := raw["content"]; ok {
		var rawBlocks []json.RawMessage
		if json.Unmarshal(contentRaw, &rawBlocks) == nil {
			for _, rb := range rawBlocks {
				var blockMap map[string]json.RawMessage
				if json.Unmarshal(rb, &blockMap) != nil {
					continue
				}

				var blockType string
				if typeRaw, ok := blockMap["type"]; ok {
					json.Unmarshal(typeRaw, &blockType)
				}

				switch blockType {
				case "text":
					var text string
					if textRaw, ok := blockMap["text"]; ok {
						json.Unmarshal(textRaw, &text)
					}
					prog.EmitString(TXT_CHUNK, text)
				case "thinking":
					prog.Emit(THINK_START)
					var thinking string
					if thinkRaw, ok := blockMap["thinking"]; ok {
						json.Unmarshal(thinkRaw, &thinking)
					}
					if thinking != "" {
						prog.EmitString(THINK_CHUNK, thinking)
					}
					if sigRaw, ok := blockMap["signature"]; ok {
						var sig string
						if json.Unmarshal(sigRaw, &sig) == nil && sig != "" {
							ref := prog.AddBuffer([]byte(sig))
							prog.EmitRef(THINK_REF, ref)
						}
					}
					prog.Emit(THINK_END)
				case "tool_use":
					var id, name string
					if idRaw, ok := blockMap["id"]; ok {
						json.Unmarshal(idRaw, &id)
						delete(blockMap, "id")
					}
					if nameRaw, ok := blockMap["name"]; ok {
						json.Unmarshal(nameRaw, &name)
						delete(blockMap, "name")
					}
					prog.EmitString(CALL_START, id)
					prog.EmitString(CALL_NAME, name)
					if inputRaw, ok := blockMap["input"]; ok {
						if len(inputRaw) > 0 {
							prog.EmitJSON(CALL_ARGS, inputRaw)
						}
						delete(blockMap, "input")
					}
					// Remaining block-level fields as EXT_DATA
					delete(blockMap, "type")
					for key, val := range blockMap {
						prog.EmitKeyJSON(EXT_DATA, key, val)
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

	delete(raw, "content")
	delete(raw, "stop_reason")
	delete(raw, "type")
	delete(raw, "role")

	// Passthrough remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}
