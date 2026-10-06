import type { ExtensionContext, AgentToolUpdateCallback } from "@earendil-works/pi-coding-agent";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getAuth, getProviderSessionHeaders } from "./auth.ts";
import { readSseEvents } from "./sse.ts";
import {
    applyTextCitations,
    deriveSources,
    mergeSearchResultMetadata,
    normalizeCitedSources,
    pushNativeSearchEvent,
    pushUniqueSearchResult,
    sanitizeSearchResults,
    titleFromUrl,
} from "./results.ts";
import type { NativeSearchCallDetail, SearchResultDetail, StreamResult } from "./types.ts";

// Z.AI (Zhipu GLM) exposes server-side web search on its Anthropic-compatible
// route only (api.z.ai/api/anthropic), not on the OpenAI-compatible coding/paas
// route that pi configures as the model baseUrl. Known hosts map to their
// Anthropic-compatible endpoint; custom proxies get a best-effort /anthropic
// suffix, mirroring the DeepSeek transport.
export function resolveZaiBaseUrl(baseUrl: string): string {
    const base = baseUrl.replace(/\/+$/, "");
    if (/\/anthropic(?:\/v1)?$/.test(base)) return base;
    let host = "";
    try {
        host = new URL(base).hostname;
    } catch {
        return `${base.replace(/\/v\d+$/, "")}/anthropic`;
    }
    if (/(^|\.)z\.ai$/.test(host)) return "https://api.z.ai/api/anthropic";
    if (/(^|\.)bigmodel\.cn$/.test(host) || /(^|\.)zhipuai\.cn$/.test(host)) return "https://open.bigmodel.cn/api/anthropic";
    return `${base.replace(/\/v\d+$/, "")}/anthropic`;
}

function resolveZaiMessagesUrl(baseUrl: string): string {
    const base = resolveZaiBaseUrl(baseUrl);
    return base.endsWith("/v1") ? `${base}/messages` : `${base}/v1/messages`;
}

interface ZaiSearchHit {
    title?: string;
    link?: string;
    url?: string;
    content?: string;
}

/** Z.AI tool_result payloads are usually JSON, sometimes a Python-style repr
 *  with single quotes, nested inside arrays. Walk whatever parses; fall back
 *  to a lenient regex so citations survive format drift. */
function parseZaiToolResult(content: unknown): ZaiSearchHit[] {
    const text = typeof content === "string" ? content : JSON.stringify(content);
    const hits: ZaiSearchHit[] = [];

    let parsed: unknown;
    try {
        parsed = JSON.parse(text);
    } catch {
        try {
            parsed = JSON.parse(text.replace(/'/g, '"'));
        } catch {
            parsed = undefined;
        }
    }
    const visit = (node: any) => {
        if (!node || typeof node !== "object") return;
        if (Array.isArray(node)) {
            for (const child of node) visit(child);
            return;
        }
        const url = node.link ?? node.url;
        if (typeof url === "string" && url) hits.push(node as ZaiSearchHit);
    };
    if (parsed !== undefined) visit(parsed);
    if (hits.length > 0) return hits;

    const re = /["']title["']\s*:\s*["']([^"']*)["'][^{}]*?["'](?:link|url)["']\s*:\s*["']([^"']+)["']/g;
    for (const match of text.matchAll(re)) hits.push({ title: match[1], link: match[2] });
    return hits;
}

function hasApiKeyHeader(headers: Record<string, string>): boolean {
    return Object.entries(headers).some(([name, value]) => {
        if (!value) return false;
        const normalized = name.toLowerCase();
        return normalized === "authorization" || normalized === "x-api-key";
    });
}

export async function callZaiStream(
    ctx: ExtensionContext,
    model: Model<Api>,
    prompt: string,
    onUpdate?: AgentToolUpdateCallback,
    signal?: AbortSignal
): Promise<StreamResult> {
    const auth = await getAuth(ctx, model);
    if (!auth.ok) {
        throw new Error(auth.error || "Failed to get API key and headers");
    }

    const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "anthropic-version": "2023-06-01",
        ...(getProviderSessionHeaders(model, ctx) || {}),
        ...(model.headers || {}),
        ...(auth.headers || {}),
    };
    if (auth.apiKey && !hasApiKeyHeader(headers)) {
        headers["x-api-key"] = auth.apiKey;
    }
    if (!hasApiKeyHeader(headers)) {
        throw new Error("No Z.AI API key found. Run /login in pi and select Z.AI, or set ZAI_API_KEY.");
    }

    const maxTokens = Math.min(Math.max(1024, Math.floor(model.maxTokens / 3) || 4096), 8192);
    // Verified against api.z.ai: the dated Anthropic server-tool type is
    // accepted and Z.AI executes its own web_search engine server-side.
    const requestBody = {
        model: model.id,
        max_tokens: maxTokens,
        messages: [{ role: "user", content: prompt }],
        tools: [{
            type: "web_search_20260209",
            name: "web_search",
            max_uses: 10,
        }],
        stream: true,
    };

    const response = await fetch(resolveZaiMessagesUrl(auth.baseUrl || model.baseUrl), {
        method: "POST",
        headers,
        body: JSON.stringify(requestBody),
        signal
    });

    if (!response.ok) {
        throw new Error(`Z.AI API error (${response.status}): ${await response.text()}`);
    }

    // Stream shape (verified): text(tool rendering) -> server_tool_use ->
    // text(results summary) -> tool_result(structured hits) -> text(answer).
    // The answer is the last non-empty text block; earlier text blocks are
    // server-side tool chatter kept out of the returned answer.
    const blockTexts: Map<number, string> = new Map();
    let sawServerSearch = false;
    let sawToolResult = false;
    const citations: Array<{ citedText?: string; title: string; url: string }> = [];
    const nativeSearchEvents: string[] = [];
    const nativeSearchCalls: NativeSearchCallDetail[] = [];
    const searchResults: SearchResultDetail[] = [];

    const collectToolResult = (block: any) => {
        pushNativeSearchEvent(nativeSearchEvents, "zai.content_block_start.tool_result");
        const call = nativeSearchCalls.find((item) => item.id === block.tool_use_id);
        if (call) call.status = "completed";
        else nativeSearchCalls.push({ id: block.tool_use_id, provider: "zai", status: "completed", actionType: "web_search", raw: block });
        for (const hit of parseZaiToolResult(block.content)) {
            const url = hit.link ?? hit.url ?? "";
            if (!url) continue;
            const title = hit.title || titleFromUrl(url);
            citations.push({ citedText: hit.content, title, url });
            pushUniqueSearchResult(searchResults, {
                title,
                url,
                citedText: hit.content,
                source: "zai.tool_result",
                type: "web_search_result",
                raw: hit,
            });
        }
    };

    const latestText = (): string => {
        let text = "";
        for (const value of blockTexts.values()) if (value.trim()) text = value;
        return text;
    };

    await readSseEvents(response, signal, ({ data: event }) => {
        if (event.type === "content_block_start") {
            const block = event.content_block;
            if (block?.type === "text" && block.text) {
                blockTexts.set(event.index, block.text);
            } else if (block?.type === "server_tool_use" && /^web_search/.test(block.name || "")) {
                sawServerSearch = true;
                pushNativeSearchEvent(nativeSearchEvents, "zai.content_block_start.server_tool_use");
                nativeSearchCalls.push({
                    id: block.id,
                    provider: "zai",
                    status: "in_progress",
                    actionType: block.name,
                    queries: typeof block.input?.search_query === "string" ? [block.input.search_query] : undefined,
                    raw: block,
                });
                onUpdate?.({
                    content: [{ type: "text", text: `Searching the web with Z.AI...` }],
                    details: { streaming: true, searching: true }
                });
            } else if (block?.type === "tool_result" && block.tool_use_id) {
                sawToolResult = true;
                collectToolResult(block);
            }
        } else if (event.type === "content_block_delta") {
            const delta = event.delta;
            if (delta?.type === "text_delta") {
                blockTexts.set(event.index, (blockTexts.get(event.index) || "") + (delta.text || ""));
                // Stream only answer-shaped text: the tool-call rendering and
                // the raw results summary arrive before tool_result, the answer
                // after it. Without a search, the single text block is the answer.
                const isAnswerBlock = sawToolResult || !sawServerSearch;
                if (isAnswerBlock) {
                    onUpdate?.({
                        content: [{ type: "text", text: blockTexts.get(event.index) || "" }],
                        details: { streaming: true }
                    });
                }
            }
        } else if (event.type === "error") {
            throw new Error(event.error?.message || JSON.stringify(event.error || event));
        }
    });

    const answer = latestText().trim() || "No answer available.";
    const cited = applyTextCitations(answer, citations);
    const citationDetails = citations.map((citation) => ({
        title: citation.title,
        url: citation.url,
        citedText: citation.citedText,
        source: "zai.citation",
        type: "citation",
        raw: citation,
    }));
    mergeSearchResultMetadata(searchResults, citationDetails);
    const sanitizedSearchResults = sanitizeSearchResults(searchResults);
    const sanitizedCitations = sanitizeSearchResults(citationDetails);
    mergeSearchResultMetadata(sanitizedSearchResults, sanitizedCitations);
    const derivedSources = deriveSources(sanitizedSearchResults, sanitizedCitations);

    return {
        text: cited.text,
        sources: cited.sources.length ? normalizeCitedSources(cited.sources) : derivedSources,
        providerKind: "zai",
        nativeSearchUsed: nativeSearchEvents.length > 0 || nativeSearchCalls.length > 0 || sanitizedSearchResults.length > 0,
        nativeSearchEvents,
        nativeSearchCalls,
        searchQueries: nativeSearchCalls.flatMap((call) => call.queries || []),
        searchResults: sanitizedSearchResults,
        citations: sanitizedCitations,
    };
}
