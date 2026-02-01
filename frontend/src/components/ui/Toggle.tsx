"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
  size?: "sm" | "md";
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled = false,
  size = "md",
}: ToggleProps) {
  const sizes = {
    sm: {
      track: "w-8 h-4",
      thumb: "h-3 w-3",
      translate: "translate-x-4",
    },
    md: {
      track: "w-11 h-6",
      thumb: "h-5 w-5",
      translate: "translate-x-5",
    },
  };

  const s = sizes[size];

  return (
    <label
      className={cn(
        "inline-flex items-center gap-3 cursor-pointer",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={cn(
          "relative inline-flex shrink-0 rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:ring-offset-2",
          s.track,
          checked ? "bg-clinical-600" : "bg-gray-200"
        )}
      >
        <span
          className={cn(
            "pointer-events-none inline-block rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
            s.thumb,
            checked ? s.translate : "translate-x-0.5",
            "mt-0.5"
          )}
        />
      </button>
      {label && (
        <span className="text-sm font-medium text-gray-700">{label}</span>
      )}
    </label>
  );
}
