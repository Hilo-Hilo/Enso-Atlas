import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// Utility function to merge Tailwind classes safely
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Format probability as percentage
export function formatProbability(probability: number | undefined): string {
  if (probability === undefined || probability === null) return "N/A";
  return `${(probability * 100).toFixed(1)}%`;
}

// Format confidence level with appropriate styling class
export function getConfidenceClass(confidence: "high" | "moderate" | "low"): string {
  switch (confidence) {
    case "high":
      return "text-status-positive bg-green-50 border-green-200";
    case "moderate":
      return "text-status-warning bg-amber-50 border-amber-200";
    case "low":
      return "text-status-negative bg-red-50 border-red-200";
    default:
      return "text-status-neutral bg-gray-50 border-gray-200";
  }
}

// Format distance score from similarity search
export function formatDistance(distance: number | undefined): string {
  if (distance === undefined || distance === null) return "N/A";
  return distance.toFixed(3);
}

// Format processing time
export function formatProcessingTime(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

// Format date for display
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Generate unique ID for components
export function generateId(): string {
  return `id-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Debounce function for search/filter inputs
export function debounce<T extends (...args: Parameters<T>) => ReturnType<T>>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

// Truncate text with ellipsis
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

// Calculate relative luminance for accessibility
export function getContrastColor(hexColor: string): "black" | "white" {
  const r = parseInt(hexColor.slice(1, 3), 16);
  const g = parseInt(hexColor.slice(3, 5), 16);
  const b = parseInt(hexColor.slice(5, 7), 16);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.5 ? "black" : "white";
}
