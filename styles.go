package ail

// Style represents a provider's API style.
// This is defined in the ail package so it has zero external dependencies,
// making it safe to extract into a standalone module.
type Style string

// Well-known API styles.
const (
	StyleChatCompletions Style = "openai-chat-completions"
	StyleResponses       Style = "openai-responses"
	StyleAnthropic       Style = "anthropic-messages"
	StyleGoogleGenAI     Style = "google-genai"
	StyleCfAiGateway     Style = "cloudflare-ai-gateway"
	StyleCfWorkersAi     Style = "cloudflare-workers-ai"
)
