"use client";

import { useEffect, useCallback, useRef } from "react";

export interface KeyboardShortcut {
  key: string;
  description: string;
  category: string;
  handler: () => void;
  modifiers?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    meta?: boolean;
  };
  /** If true, this shortcut is an alias and should be hidden from the shortcuts modal */
  hidden?: boolean;
}

export interface UseKeyboardShortcutsOptions {
  enabled?: boolean;
  shortcuts: KeyboardShortcut[];
}

function isEditableElement(target: HTMLElement): boolean {
  const tagName = target.tagName.toLowerCase();
  if (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    target.isContentEditable
  ) {
    return true;
  }

  if (target.closest("[contenteditable='true']")) {
    return true;
  }

  const role = target.getAttribute("role");
  return role === "textbox" || role === "combobox" || role === "searchbox";
}

function isInteractiveElement(target: HTMLElement): boolean {
  if (target.closest("[data-shortcuts='ignore']")) {
    return true;
  }

  return !!target.closest(
    "button, a, summary, details, [role='button'], [role='link'], [role='menuitem'], [role='switch'], [role='tab'], [role='option']"
  );
}

/**
 * Hook for managing global keyboard shortcuts.
 * Automatically disables shortcuts when focus is in editable or interactive fields.
 */
export function useKeyboardShortcuts({
  enabled = true,
  shortcuts,
}: UseKeyboardShortcutsOptions) {
  const shortcutsRef = useRef(shortcuts);
  shortcutsRef.current = shortcuts;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;
      if (event.defaultPrevented || event.isComposing || event.repeat) return;

      const target = event.target as HTMLElement | null;
      const pressedKey = event.key.toLowerCase();
      const isEscape = pressedKey === "escape" || pressedKey === "esc";

      // Ignore shortcuts when typing/editing, and avoid hijacking key presses on interactive controls.
      // Keep Escape available so dialogs can still be dismissed.
      if (target && !isEscape && (isEditableElement(target) || isInteractiveElement(target))) {
        return;
      }

      // Find matching shortcut
      const matchingShortcut = shortcutsRef.current.find((shortcut) => {
        // Check key match (case-insensitive)
        const keyMatch =
          event.key.toLowerCase() === shortcut.key.toLowerCase() ||
          event.code.toLowerCase() === shortcut.key.toLowerCase();

        if (!keyMatch) return false;

        // Check modifiers
        const modifiers = shortcut.modifiers || {};
        const ctrlOrMetaPressed = event.ctrlKey || event.metaKey;

        const ctrlMatch = modifiers.ctrl ? ctrlOrMetaPressed : !event.ctrlKey;
        const metaMatch = modifiers.meta ? event.metaKey : !event.metaKey;
        const altMatch = modifiers.alt ? event.altKey : !event.altKey;
        const shiftMatch = modifiers.shift ? event.shiftKey : !event.shiftKey;

        return ctrlMatch && metaMatch && altMatch && shiftMatch;
      });

      if (matchingShortcut) {
        event.preventDefault();
        event.stopPropagation();
        matchingShortcut.handler();
      }
    },
    [enabled]
  );

  useEffect(() => {
    // Only add event listener in browser environment
    if (typeof window === "undefined") return;

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return {
    shortcuts: shortcutsRef.current,
  };
}

/**
 * Get a formatted display string for a shortcut key
 */
export function formatShortcutKey(
  key: string,
  modifiers?: KeyboardShortcut["modifiers"]
): string {
  const parts: string[] = [];

  if (modifiers?.ctrl || modifiers?.meta) {
    const isMac = typeof navigator !== "undefined" && navigator.platform.includes("Mac");
    parts.push(isMac ? "Cmd" : "Ctrl");
  }
  if (modifiers?.alt) {
    parts.push("Alt");
  }
  if (modifiers?.shift) {
    parts.push("Shift");
  }

  // Format the key nicely
  let displayKey = key;
  switch (key.toLowerCase()) {
    case "arrowup":
      displayKey = "Up";
      break;
    case "arrowdown":
      displayKey = "Down";
      break;
    case "arrowleft":
      displayKey = "Left";
      break;
    case "arrowright":
      displayKey = "Right";
      break;
    case "escape":
      displayKey = "Esc";
      break;
    case "enter":
      displayKey = "Enter";
      break;
    case " ":
      displayKey = "Space";
      break;
    default:
      displayKey = key.toUpperCase();
  }

  parts.push(displayKey);
  return parts.join(" + ");
}

/**
 * Group shortcuts by category for display
 */
export function groupShortcutsByCategory(
  shortcuts: KeyboardShortcut[]
): Map<string, KeyboardShortcut[]> {
  const groups = new Map<string, KeyboardShortcut[]>();

  for (const shortcut of shortcuts) {
    const category = shortcut.category;
    if (!groups.has(category)) {
      groups.set(category, []);
    }
    groups.get(category)!.push(shortcut);
  }

  return groups;
}
