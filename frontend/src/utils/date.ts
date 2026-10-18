/**
 * Converts ISO/UTC timestamps to user's local device time (e.g. YYYY-MM-DD HH:mm:ss).
 * Ensures consistency with the local OS system clock.
 */
export function formatToLocalTime(isoString?: string | null): string {
  if (!isoString) return 'N/A';
  try {
    const s = isoString.trim();
    if (!s) return 'N/A';
    // If string does not have timezone information ('Z' or '+/-'), treat as UTC
    const normalized = (s.endsWith('Z') || s.includes('+') || (s.includes('-') && s.length > 19))
      ? s
      : s + 'Z';
    const d = new Date(normalized);
    if (isNaN(d.getTime())) {
      return s.substring(0, 19).replace('T', ' ');
    }
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  } catch {
    return isoString.substring(0, 19).replace('T', ' ');
  }
}
