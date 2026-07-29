import DOMPurify from "dompurify";
import { marked } from "marked";

import { conversationLocation } from "./conversationLocator";

export function normalizeBackendText(text: string): string {
  return (text || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t");
}

export function markdownHtml(source: string, conversationId: string | null): string {
  marked.setOptions({ gfm: true, breaks: true });
  const raw = marked.parse(displayAssistantSource(source || ""), { async: false });
  const clean = DOMPurify.sanitize(raw, { ADD_ATTR: ["target"] });
  return rewriteLocalImages(clean, conversationId);
}

function displayAssistantSource(source: string): string {
  let text = normalizeBackendText(source || "");
  text = text.replace(
    /<details\s+class=["']role-output orchestrator["'][\s\S]*?<\/details>\s*/gi,
    "",
  );
  text = stripLegacyMarkdownOrchestrator(text);
  if (text.startsWith("### Message\n\n")) {
    return text.slice("### Message\n\n".length).trimStart();
  }
  return text.trimStart();
}

function stripLegacyMarkdownOrchestrator(source: string): string {
  const text = source.trimStart();
  if (!text.startsWith("### Orchestrator")) {
    return source;
  }

  const markers = [
    { marker: "\n\n### Implementer Summary\n\n", keepHeading: true },
    { marker: "\n\n### Message\n\n", keepHeading: false },
    { marker: "\n\n### Error\n\n", keepHeading: true },
  ];
  const match = markers
    .map((candidate) => ({ ...candidate, index: text.indexOf(candidate.marker) }))
    .filter((candidate) => candidate.index >= 0)
    .sort((left, right) => left.index - right.index)[0];
  if (!match) {
    return source;
  }

  if (match.keepHeading) {
    return text.slice(match.index + 2);
  }
  return text.slice(match.index + match.marker.length);
}

function rewriteLocalImages(html: string, conversationId: string | null): string {
  const template = document.createElement("template");
  template.innerHTML = html;
  template.content.querySelectorAll("img").forEach((image) => {
    const src = image.getAttribute("src") || "";
    if (!/^(https?:|data:|\/api\/file)/i.test(src)) {
      const workspaceId = conversationId
        ? conversationLocation(conversationId).workspaceId
        : "w_main";
      image.setAttribute(
        "src",
        `/api/file?path=${encodeURIComponent(src)}&workspace_id=${encodeURIComponent(workspaceId)}`,
      );
    }
    image.setAttribute("loading", "lazy");
  });
  template.content.querySelectorAll("a").forEach((anchor) => {
    anchor.setAttribute("target", "_blank");
    anchor.setAttribute("rel", "noopener");
  });
  return template.innerHTML;
}
