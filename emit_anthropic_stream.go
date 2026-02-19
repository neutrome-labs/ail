package ail

import (
	"encoding/json"
)

func (e *AnthropicEmitter) EmitStreamChunk(prog *Program) ([]byte, error) {
	// Anthropic streaming uses typed events; emit the appropriate type
	for _, inst := range prog.Code {
		switch inst.Op {
		case STREAM_START:
			event := map[string]any{"type": "message_start"}
			msgObj := map[string]any{"role": "assistant"}
			// Look ahead for RESP_ID and RESP_MODEL in same chunk
			for _, ahead := range prog.Code {
				if ahead.Op == RESP_ID {
					msgObj["id"] = ahead.Str
				}
				if ahead.Op == RESP_MODEL {
					msgObj["model"] = ahead.Str
				}
			}
			event["message"] = msgObj
			return json.Marshal(event)

		case STREAM_DELTA:
			event := map[string]any{
				"type": "content_block_delta",
				"delta": map[string]any{
					"type": "text_delta",
					"text": inst.Str,
				},
			}
			return json.Marshal(event)

		case STREAM_THINK_DELTA:
			event := map[string]any{
				"type": "content_block_delta",
				"delta": map[string]any{
					"type":     "thinking_delta",
					"thinking": inst.Str,
				},
			}
			return json.Marshal(event)

		case STREAM_TOOL_DELTA:
			var td map[string]any
			if json.Unmarshal(inst.JSON, &td) == nil {
				if _, hasName := td["name"]; hasName {
					// Tool start
					event := map[string]any{
						"type":  "content_block_start",
						"index": td["index"],
						"content_block": map[string]any{
							"type": "tool_use",
							"id":   td["id"],
							"name": td["name"],
						},
					}
					return json.Marshal(event)
				}
				if args, ok := td["arguments"]; ok {
					event := map[string]any{
						"type":  "content_block_delta",
						"index": td["index"],
						"delta": map[string]any{
							"type":         "input_json_delta",
							"partial_json": args,
						},
					}
					return json.Marshal(event)
				}
			}

		case RESP_DONE:
			stopReason := "end_turn"
			switch inst.Str {
			case "stop":
				stopReason = "end_turn"
			case "tool_calls":
				stopReason = "tool_use"
			case "length":
				stopReason = "max_tokens"
			default:
				stopReason = inst.Str
			}
			event := map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{"stop_reason": stopReason},
			}
			// Look ahead for USAGE in the same chunk (Anthropic puts
			// usage alongside stop_reason in message_delta).
			for _, ahead := range prog.Code {
				if ahead.Op == USAGE {
					event["usage"] = json.RawMessage(ahead.JSON)
				}
			}
			return json.Marshal(event)

		case STREAM_END:
			return json.Marshal(map[string]any{"type": "message_stop"})

		case EXT_DATA:
			// Stream events are typed — top-level EXT_DATA is merged
			// into the next event if any, but for simplicity we skip
			// since stream chunks are small single-events.
		}
	}

	// Empty chunk
	return json.Marshal(map[string]any{"type": "ping"})
}
