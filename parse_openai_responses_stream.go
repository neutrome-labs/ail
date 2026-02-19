package ail

import (
	"encoding/json"
	"fmt"
)

func (p *ResponsesParser) ParseStreamChunk(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse responses stream chunk: %w", err)
	}

	prog := NewProgram()

	eventType := ""
	if typeRaw, ok := raw["type"]; ok {
		json.Unmarshal(typeRaw, &eventType)
	}

	switch eventType {
	case "response.created", "response.in_progress":
		prog.Emit(STREAM_START)
		// Extract response ID if present
		if respRaw, ok := raw["response"]; ok {
			var resp struct {
				ID    string `json:"id"`
				Model string `json:"model"`
			}
			if json.Unmarshal(respRaw, &resp) == nil {
				if resp.ID != "" {
					prog.EmitString(RESP_ID, resp.ID)
				}
				if resp.Model != "" {
					prog.EmitString(RESP_MODEL, resp.Model)
				}
			}
		}

	case "response.output_text.delta":
		delta := ""
		if deltaRaw, ok := raw["delta"]; ok {
			json.Unmarshal(deltaRaw, &delta)
		}
		if delta != "" {
			prog.EmitString(STREAM_DELTA, delta)
		}

	case "response.reasoning_summary_text.delta":
		delta := ""
		if deltaRaw, ok := raw["delta"]; ok {
			json.Unmarshal(deltaRaw, &delta)
		}
		if delta != "" {
			prog.EmitString(STREAM_THINK_DELTA, delta)
		}

	case "response.function_call_arguments.delta":
		delta := ""
		if deltaRaw, ok := raw["delta"]; ok {
			json.Unmarshal(deltaRaw, &delta)
		}
		outputIndex := 0
		if idxRaw, ok := raw["output_index"]; ok {
			json.Unmarshal(idxRaw, &outputIndex)
		}
		itemID := ""
		if idRaw, ok := raw["item_id"]; ok {
			json.Unmarshal(idRaw, &itemID)
		}
		toolDelta := map[string]any{
			"index":     outputIndex,
			"arguments": delta,
		}
		if itemID != "" {
			toolDelta["id"] = itemID
		}
		j, _ := json.Marshal(toolDelta)
		prog.EmitJSON(STREAM_TOOL_DELTA, j)

	case "response.output_item.added":
		// New output item (message or function call)
		if itemRaw, ok := raw["item"]; ok {
			var item struct {
				Type   string `json:"type"`
				ID     string `json:"id"`
				CallID string `json:"call_id,omitempty"`
				Name   string `json:"name,omitempty"`
			}
			if json.Unmarshal(itemRaw, &item) == nil {
				if item.Type == "function_call" {
					td := map[string]any{"index": 0, "id": item.CallID, "name": item.Name}
					j, _ := json.Marshal(td)
					prog.EmitJSON(STREAM_TOOL_DELTA, j)
				}
			}
		}

	case "response.output_item.done":
		if itemRaw, ok := raw["item"]; ok {
			var item struct {
				Type   string `json:"type"`
				Status string `json:"status"`
			}
			if json.Unmarshal(itemRaw, &item) == nil {
				if item.Status == "completed" {
					switch item.Type {
					case "message":
						prog.EmitString(RESP_DONE, "stop")
					case "function_call":
						prog.EmitString(RESP_DONE, "tool_calls")
					}
				}
			}
		}

	case "response.completed", "response.done":
		if respRaw, ok := raw["response"]; ok {
			var resp struct {
				Usage *struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
					TotalTokens  int `json:"total_tokens"`
				} `json:"usage,omitempty"`
			}
			if json.Unmarshal(respRaw, &resp) == nil && resp.Usage != nil {
				stdUsage, _ := json.Marshal(map[string]int{
					"prompt_tokens":     resp.Usage.InputTokens,
					"completion_tokens": resp.Usage.OutputTokens,
					"total_tokens":      resp.Usage.TotalTokens,
				})
				prog.EmitJSON(USAGE, stdUsage)
			}
		}
		prog.Emit(STREAM_END)
	}

	return prog, nil
}
