import DOMPurify from "dompurify";
import { marked } from "marked";

export function normalizeBackendText(text: string): string {
  return (text || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t");
}

export function markdownHtml(source: string, conversationId: string | null): string {
  marked.setOptions({ gfm: true, breaks: true });
  const raw = marked.parse(normalizeBackendText(source || ""), { async: false });
  const clean = DOMPurify.sanitize(raw, { ADD_ATTR: ["target"] });
  return rewriteLocalImages(clean, conversationId);
}

function rewriteLocalImages(html: string, conversationId: string | null): string {
  const template = document.createElement("template");
  template.innerHTML = html;
  template.content.querySelectorAll("img").forEach((image) => {
    const src = image.getAttribute("src") || "";
    if (!/^(https?:|data:|\/api\/file)/i.test(src)) {
      const cid = conversationId ? `&cid=${encodeURIComponent(conversationId)}` : "";
      image.setAttribute("src", `/api/file?path=${encodeURIComponent(src)}${cid}`);
    }
    image.setAttribute("loading", "lazy");
  });
  template.content.querySelectorAll("a").forEach((anchor) => {
    anchor.setAttribute("target", "_blank");
    anchor.setAttribute("rel", "noopener");
  });
  return template.innerHTML;
}
