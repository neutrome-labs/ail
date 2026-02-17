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
