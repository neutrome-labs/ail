package ail

import (
	"encoding/json"
	"fmt"
)

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
			if json.Unmarshal(cbRaw, &cb) == nil {
				switch cb.Type {
				case "tool_use":
					idx := 0
					if idxRaw, ok := raw["index"]; ok {
						json.Unmarshal(idxRaw, &idx)
					}
					td := map[string]any{"index": idx, "id": cb.ID, "name": cb.Name}
					j, _ := json.Marshal(td)
					prog.EmitJSON(STREAM_TOOL_DELTA, j)
				case "thinking":
					// Thinking block started — emit nothing, deltas carry content
				}
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
				case "thinking_delta":
					var thinkDelta struct {
						Thinking string `json:"thinking"`
					}
					if json.Unmarshal(deltaRaw, &thinkDelta) == nil && thinkDelta.Thinking != "" {
						prog.EmitString(STREAM_THINK_DELTA, thinkDelta.Thinking)
					}
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
