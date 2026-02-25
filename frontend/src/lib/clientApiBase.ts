const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"]);

function normalizeApiBase(apiBase: string): string {
  return apiBase.replace(/\/+$/, "");
}

/**
 * Resolve the API base URL that is safe to use from client-side code.
 *
 * If NEXT_PUBLIC_API_URL points to loopback (127.0.0.1/localhost) but the
 * browser is not on loopback, we fall back to same-origin (empty base) so
 * requests continue to route through Next.js rewrites/proxies.
 */
export function resolveClientApiBaseUrl(
  configuredApiBase: string | undefined,
  browserHostname?: string
): string {
  const raw = configuredApiBase?.trim();
  if (!raw) return "";

  if (!browserHostname) {
    return normalizeApiBase(raw);
  }

  try {
    const parsed = new URL(raw);
    const targetHost = parsed.hostname.toLowerCase();
    const browserHost = browserHostname.toLowerCase();
    const targetIsLoopback = LOOPBACK_HOSTS.has(targetHost);
    const browserIsLoopback = LOOPBACK_HOSTS.has(browserHost);

    if (targetIsLoopback && !browserIsLoopback) {
      return "";
    }
  } catch {
    // Relative paths are valid and should remain same-origin.
  }

  return normalizeApiBase(raw);
}

export function getClientApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return resolveClientApiBaseUrl(process.env.NEXT_PUBLIC_API_URL);
  }

  return resolveClientApiBaseUrl(
    process.env.NEXT_PUBLIC_API_URL,
    window.location.hostname
  );
}
