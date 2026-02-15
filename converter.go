package ail

import (
	"fmt"
)

// ─── Converter: any-to-any via AIL ──────────────────────────────────────────

// GetParser returns the appropriate parser for the given style.
func GetParser(style Style) (Parser, error) {
	switch style {
	case StyleChatCompletions:
		return &ChatCompletionsParser{}, nil
	case StyleResponses:
		return &ResponsesParser{}, nil
	case StyleAnthropic:
		return &AnthropicParser{}, nil
	case StyleGoogleGenAI:
		return &GoogleGenAIParser{}, nil
	default:
		return nil, fmt.Errorf("ail: no parser for style %q", style)
	}
}

// GetEmitter returns the appropriate emitter for the given style.
func GetEmitter(style Style) (Emitter, error) {
	switch style {
	case StyleChatCompletions:
		return &ChatCompletionsEmitter{}, nil
	case StyleResponses:
		return &ResponsesEmitter{}, nil
	case StyleAnthropic:
		return &AnthropicEmitter{}, nil
	case StyleGoogleGenAI:
		return &GoogleGenAIEmitter{}, nil
	default:
		return nil, fmt.Errorf("ail: no emitter for style %q", style)
	}
}

// GetResponseParser returns the appropriate response parser for a style.
func GetResponseParser(style Style) (ResponseParser, error) {
	switch style {
	case StyleChatCompletions:
		return &ChatCompletionsParser{}, nil
	case StyleResponses:
		return &ResponsesParser{}, nil
	case StyleAnthropic:
		return &AnthropicParser{}, nil
	case StyleGoogleGenAI:
		return &GoogleGenAIParser{}, nil
	default:
		return nil, fmt.Errorf("ail: no response parser for style %q", style)
	}
}

// GetResponseEmitter returns the appropriate response emitter for a style.
func GetResponseEmitter(style Style) (ResponseEmitter, error) {
	switch style {
	case StyleChatCompletions:
		return &ChatCompletionsEmitter{}, nil
	case StyleAnthropic:
		return &AnthropicEmitter{}, nil
	case StyleGoogleGenAI:
		return &GoogleGenAIEmitter{}, nil
	default:
		return nil, fmt.Errorf("ail: no response emitter for style %q", style)
	}
}

// GetStreamChunkParser returns the appropriate stream chunk parser.
func GetStreamChunkParser(style Style) (StreamChunkParser, error) {
	switch style {
	case StyleChatCompletions:
		return &ChatCompletionsParser{}, nil
	case StyleResponses:
		return &ResponsesParser{}, nil
	case StyleAnthropic:
		return &AnthropicParser{}, nil
	case StyleGoogleGenAI:
		return &GoogleGenAIParser{}, nil
	default:
		return nil, fmt.Errorf("ail: no stream chunk parser for style %q", style)
	}
}

// GetStreamChunkEmitter returns the appropriate stream chunk emitter.
func GetStreamChunkEmitter(style Style) (StreamChunkEmitter, error) {
	switch style {
	case StyleChatCompletions:
		return &ChatCompletionsEmitter{}, nil
	case StyleAnthropic:
		return &AnthropicEmitter{}, nil
	case StyleGoogleGenAI:
		return &GoogleGenAIEmitter{}, nil
	default:
		return nil, fmt.Errorf("ail: no stream chunk emitter for style %q", style)
	}
}

// ─── Convenience: Convert request from one style to another ──────────────────

// ConvertRequest converts a request body from one style to another via AIL.
// If from == to, it's a passthrough (still parses/emits for normalization).
func ConvertRequest(body []byte, from, to Style) ([]byte, error) {
	parser, err := GetParser(from)
	if err != nil {
		return nil, err
	}

	prog, err := parser.ParseRequest(body)
	if err != nil {
		return nil, err
	}

	emitter, err := GetEmitter(to)
	if err != nil {
		return nil, err
	}

	return emitter.EmitRequest(prog)
}

// ConvertRequestProgram parses a request into AIL, then emits for the target style.
// Returns both the program (for plugin inspection) and the emitted bytes.
func ConvertRequestProgram(body []byte, from, to Style) (*Program, []byte, error) {
	parser, err := GetParser(from)
	if err != nil {
		return nil, nil, err
	}

	prog, err := parser.ParseRequest(body)
	if err != nil {
		return nil, nil, err
	}

	emitter, err := GetEmitter(to)
	if err != nil {
		return nil, nil, err
	}

	out, err := emitter.EmitRequest(prog)
	if err != nil {
		return nil, nil, err
	}

	return prog, out, nil
}
