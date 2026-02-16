// Package main implements the llm2llm web app — a single-page application
// for converting between AI provider API formats using the AIL intermediate language.
//
// Run:
//
//	go run ./cmd/demo
//
// Then open http://localhost:8080 in your browser.
package main

import (
	"embed"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/neutrome-labs/ail"
)

//go:embed static
var staticFiles embed.FS

// ─── Slug ↔ Style mapping ─────────────────────────────────────────

var slugToStyle = map[string]string{
	"chat":      "openai-chat-completions",
	"responses": "openai-responses",
	"anthropic": "anthropic-messages",
	"genai":     "google-genai",
	"ail":       "ail",
}

var styleToSlug = map[string]string{
	"openai-chat-completions": "chat",
	"openai-responses":        "responses",
	"anthropic-messages":      "anthropic",
	"google-genai":            "genai",
	"ail":                     "ail",
}

var styleDisplayName = map[string]string{
	"openai-chat-completions": "OpenAI Chat Completions",
	"openai-responses":        "OpenAI Responses",
	"anthropic-messages":      "Anthropic Messages",
	"google-genai":            "Google GenAI",
	"ail":                     "AIL Assembly",
}

// slugOrder determines canonical ordering for sitemap generation.
var slugOrder = []string{"chat", "responses", "anthropic", "genai", "ail"}

// ─── Template data ────────────────────────────────────────────────

type pageData struct {
	Title       string
	Description string
	Canonical   string
	OGTitle     string
	OGDesc      string
	OGURL       string
	FromStyle   string
	ToStyle     string
	FromSlug    string
	ToSlug      string
	H2          string
	Intro       string
	Keywords    string
	NavLinks    []navLink
	BaseDomain  string
}

// navLink represents a single internal link in the conversion grid.
type navLink struct {
	Href  string
	Label string
}

func allNavLinks() []navLink {
	var links []navLink
	for _, from := range slugOrder {
		for _, to := range slugOrder {
			if from == to {
				continue
			}
			fromName := styleDisplayName[slugToStyle[from]]
			toName := styleDisplayName[slugToStyle[to]]
			links = append(links, navLink{
				Href:  "/" + from + "/" + to,
				Label: fromName + " to " + toName,
			})
		}
	}
	return links
}

// baseDomain can be overridden via the BASE_DOMAIN env var.
func baseDomain() string {
	if d := os.Getenv("BASE_DOMAIN"); d != "" {
		return strings.TrimRight(d, "/")
	}
	return "https://example.com"
}

func homePage() pageData {
	base := baseDomain()
	return pageData{
		Title:       "llm2llm — Convert between LLM API formats online",
		Description: "Free online tool to convert between OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Google GenAI and AIL assembly formats. Convert LLM API request and response JSON instantly.",
		Canonical:   base + "/",
		OGTitle:     "llm2llm — Convert between LLM API formats online",
		OGDesc:      "Convert OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Google GenAI and AIL assembly formats to each other. Supports requests, responses and streaming chunks.",
		OGURL:       base + "/",
		FromStyle:   "openai-chat-completions",
		ToStyle:     "anthropic-messages",
		FromSlug:    "chat",
		ToSlug:      "anthropic",
		H2:          "Convert any LLM API format to any other",
		Intro:       "Paste an OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, or Google GenAI JSON payload and convert it to any other LLM provider format instantly. Supports request bodies, response bodies, and streaming chunks.",
		Keywords:    "convert chat completions to anthropic, convert openai to google genai, LLM API format converter, openai responses to chat completions, convert LLM JSON, AI API translator",
		NavLinks:    allNavLinks(),
		BaseDomain:  base,
	}
}

func routePage(fromSlug, toSlug string) pageData {
	base := baseDomain()
	fromStyle := slugToStyle[fromSlug]
	toStyle := slugToStyle[toSlug]
	fromName := styleDisplayName[fromStyle]
	toName := styleDisplayName[toStyle]
	path := "/" + fromSlug + "/" + toSlug

	h2 := fmt.Sprintf("Convert %s to %s JSON", fromName, toName)
	intro := fmt.Sprintf(
		"Instantly convert %s API format to %s format online. "+
			"Paste your %s request or response JSON and get the equivalent %s JSON in one click. "+
			"Supports request bodies, response bodies, and streaming chunks.",
		fromName, toName, fromName, toName,
	)
	keywords := fmt.Sprintf(
		"convert %s to %s, %s to %s JSON, %s to %s API, %s converter, %s converter, LLM API converter",
		fromName, toName, fromName, toName, fromName, toName, fromName, toName,
	)

	return pageData{
		Title:       fmt.Sprintf("Convert %s to %s JSON online — llm2llm", fromName, toName),
		Description: fmt.Sprintf("Free online tool to convert %s API JSON to %s format. Paste your %s request, response, or streaming chunk and get the %s equivalent instantly.", fromName, toName, fromName, toName),
		Canonical:   base + path,
		OGTitle:     fmt.Sprintf("Convert %s to %s — llm2llm", fromName, toName),
		OGDesc:      fmt.Sprintf("Free online converter from %s to %s format. Supports requests, responses and streaming chunks.", fromName, toName),
		OGURL:       base + path,
		FromStyle:   fromStyle,
		ToStyle:     toStyle,
		FromSlug:    fromSlug,
		ToSlug:      toSlug,
		H2:          h2,
		Intro:       intro,
		Keywords:    keywords,
		NavLinks:    allNavLinks(),
		BaseDomain:  base,
	}
}

// ─── Sitemap ──────────────────────────────────────────────────────

type sitemapURL struct {
	XMLName    xml.Name `xml:"url"`
	Loc        string   `xml:"loc"`
	ChangeFreq string   `xml:"changefreq,omitempty"`
	Priority   string   `xml:"priority,omitempty"`
	LastMod    string   `xml:"lastmod,omitempty"`
}

type sitemapIndex struct {
	XMLName xml.Name     `xml:"urlset"`
	XMLNS   string       `xml:"xmlns,attr"`
	URLs    []sitemapURL `xml:"url"`
}

func generateSitemap() []byte {
	base := baseDomain()
	now := time.Now().Format("2006-01-02")

	sm := sitemapIndex{XMLNS: "http://www.sitemaps.org/schemas/sitemap/0.9"}

	// Homepage
	sm.URLs = append(sm.URLs, sitemapURL{
		Loc:        base + "/",
		ChangeFreq: "weekly",
		Priority:   "1.0",
		LastMod:    now,
	})

	// All from/to variants
	for _, from := range slugOrder {
		for _, to := range slugOrder {
			if from == to {
				continue
			}
			sm.URLs = append(sm.URLs, sitemapURL{
				Loc:        base + "/" + from + "/" + to,
				ChangeFreq: "monthly",
				Priority:   "0.8",
				LastMod:    now,
			})
		}
	}

	out, _ := xml.MarshalIndent(sm, "", "  ")
	return append([]byte(xml.Header), out...)
}

// ─── API types (unchanged) ────────────────────────────────────────

// convertRequest is the JSON body sent by the frontend.
type convertRequest struct {
	Input     string `json:"input"`
	FromStyle string `json:"fromStyle"`
	ToStyle   string `json:"toStyle"`
	Type      string `json:"type"` // "request", "response", "stream_chunk"
}

// convertResponse is what we reply with.
type convertResponse struct {
	Output string `json:"output"`
	Disasm string `json:"disasm"`
	Error  string `json:"error,omitempty"`
}

const styleAIL = "ail"

func handleConvert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req convertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, convertResponse{Error: "invalid JSON body: " + err.Error()})
		return
	}

	from := req.FromStyle
	to := req.ToStyle
	input := []byte(req.Input)

	// ─── Parse phase ────────────────────────────────────────────────────
	var (
		prog *ail.Program
		err  error
	)

	if from == styleAIL {
		// Input is AIL assembly text — parse with Asm()
		prog, err = ail.Asm(req.Input)
		if err != nil {
			writeJSON(w, http.StatusUnprocessableEntity, convertResponse{Error: fmt.Sprintf("AIL parse error: %v", err)})
			return
		}
	} else {
		switch req.Type {
		case "request":
			parser, e := ail.GetParser(ail.Style(from))
			if e != nil {
				err = e
			} else {
				prog, err = parser.ParseRequest(input)
			}
		case "response":
			parser, e := ail.GetResponseParser(ail.Style(from))
			if e != nil {
				err = e
			} else {
				prog, err = parser.ParseResponse(input)
			}
		case "stream_chunk":
			parser, e := ail.GetStreamChunkParser(ail.Style(from))
			if e != nil {
				err = e
			} else {
				prog, err = parser.ParseStreamChunk(input)
			}
		default:
			writeJSON(w, http.StatusBadRequest, convertResponse{Error: fmt.Sprintf("unknown type %q", req.Type)})
			return
		}
	}

	if err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, convertResponse{Error: err.Error()})
		return
	}

	// ─── Emit phase ─────────────────────────────────────────────────────
	var out []byte

	if to == styleAIL {
		// Output is AIL disassembly text
		out = []byte(prog.Disasm())
	} else {
		switch req.Type {
		case "request":
			emitter, e := ail.GetEmitter(ail.Style(to))
			if e != nil {
				err = e
			} else {
				out, err = emitter.EmitRequest(prog)
			}
		case "response":
			emitter, e := ail.GetResponseEmitter(ail.Style(to))
			if e != nil {
				err = e
			} else {
				out, err = emitter.EmitResponse(prog)
			}
		case "stream_chunk":
			emitter, e := ail.GetStreamChunkEmitter(ail.Style(to))
			if e != nil {
				err = e
			} else {
				out, err = emitter.EmitStreamChunk(prog)
			}
		}
	}

	if err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, convertResponse{Error: err.Error()})
		return
	}

	// Pretty-print if JSON output
	if to != styleAIL {
		var pretty json.RawMessage
		if json.Unmarshal(out, &pretty) == nil {
			if p, err := json.MarshalIndent(pretty, "", "  "); err == nil {
				out = p
			}
		}
	}

	resp := convertResponse{
		Output: string(out),
	}
	if prog != nil {
		resp.Disasm = prog.Disasm()
	}

	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func main() {
	port := "8080"
	if p := os.Getenv("PORT"); p != "" {
		port = p
	}

	// Parse the embedded HTML template.
	tmplBytes, err := staticFiles.ReadFile("static/index.html")
	if err != nil {
		log.Fatalf("failed to read template: %v", err)
	}
	pageTmpl, err := template.New("page").Parse(string(tmplBytes))
	if err != nil {
		log.Fatalf("failed to parse template: %v", err)
	}

	// Pre-generate sitemap.
	sitemapBytes := generateSitemap()

	mux := http.NewServeMux()

	// API endpoints.
	mux.HandleFunc("POST /api/convert", handleConvert)

	// Sitemap.
	mux.HandleFunc("GET /sitemap.xml", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml; charset=utf-8")
		w.Header().Set("Cache-Control", "public, max-age=86400")
		w.Write(sitemapBytes)
	})

	// robots.txt
	mux.HandleFunc("GET /robots.txt", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprintf(w, "User-agent: *\nAllow: /\nSitemap: %s/sitemap.xml\n", baseDomain())
	})

	// Favicon.
	mux.HandleFunc("GET /favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		data, err := staticFiles.ReadFile("static/favicon.ico")
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "image/x-icon")
		w.Header().Set("Cache-Control", "public, max-age=604800")
		w.Write(data)
	})

	// Serve the page template — handles homepage and /{from}/{to} routes.
	mux.HandleFunc("GET /{path...}", func(w http.ResponseWriter, r *http.Request) {
		path := strings.Trim(r.URL.Path, "/")

		var data pageData

		switch {
		case path == "":
			data = homePage()
		default:
			parts := strings.SplitN(path, "/", 3)
			if len(parts) == 2 {
				fromSlug, toSlug := parts[0], parts[1]
				_, fromOK := slugToStyle[fromSlug]
				_, toOK := slugToStyle[toSlug]
				if fromOK && toOK && fromSlug != toSlug {
					data = routePage(fromSlug, toSlug)
				} else {
					http.Redirect(w, r, "/", http.StatusFound)
					return
				}
			} else {
				http.Redirect(w, r, "/", http.StatusFound)
				return
			}
		}

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := pageTmpl.Execute(w, data); err != nil {
			http.Error(w, "template error", http.StatusInternalServerError)
			log.Printf("template error: %v", err)
		}
	})

	fmt.Printf("🚀 llm2llm running at http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
