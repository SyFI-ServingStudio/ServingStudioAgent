const LOCATOR_PREFIX = "workspace:";
const MAIN_WORKSPACE_ID = "w_main";

export interface ConversationLocation {
  workspaceId: string;
  conversationId: string;
}

/**
 * The standalone shell predates workspace identity and stores one opaque
 * `currentId`. Keep that component contract while carrying both identities;
 * every network request decodes the locator before addressing the backend.
 */
export function conversationLocator(workspaceId: string, conversationId: string): string {
  return `${LOCATOR_PREFIX}${encodeURIComponent(workspaceId)}:${encodeURIComponent(conversationId)}`;
}

export function conversationLocation(locator: string): ConversationLocation {
  if (!locator.startsWith(LOCATOR_PREFIX)) {
    return { workspaceId: MAIN_WORKSPACE_ID, conversationId: locator };
  }
  const separator = locator.indexOf(":", LOCATOR_PREFIX.length);
  if (separator < 0) {
    return { workspaceId: MAIN_WORKSPACE_ID, conversationId: locator };
  }
  return {
    workspaceId: decodeURIComponent(locator.slice(LOCATOR_PREFIX.length, separator)),
    conversationId: decodeURIComponent(locator.slice(separator + 1)),
  };
}
