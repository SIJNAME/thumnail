import { ValidationError } from "./errors";

export function sanitizeUrl(urlStr: string): string {
  const url = new URL(urlStr);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new ValidationError("Only http/https URLs are allowed");
  }
  return url.toString();
}
