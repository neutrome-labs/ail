package ail

import (
	"encoding/json"
	"fmt"
)

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

	// Usage (may appear in final chunk with stream_options)
	if usageRaw, ok := raw["usage"]; ok {
		prog.EmitJSON(USAGE, usageRaw)
		delete(raw, "usage")
	}

	// Choices (each with a delta)
	if choicesRaw, ok := raw["choices"]; ok {
		var choices []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Delta        *struct {
				Role             string          `json:"role,omitempty"`
				Content          json.RawMessage `json:"content,omitempty"`
				ReasoningContent json.RawMessage `json:"reasoning_content,omitempty"`
				ToolCalls        []struct {
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
					if choice.Delta.ReasoningContent != nil {
						var rc string
						if json.Unmarshal(choice.Delta.ReasoningContent, &rc) == nil && rc != "" {
							prog.EmitString(STREAM_THINK_DELTA, rc)
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
	delete(raw, "choices")

	// Passthrough remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}
