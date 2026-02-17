"use client";

import React, { useState } from "react";
import { AlertTriangle, X } from "lucide-react";

export function DisclaimerBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  return (
    <div className="relative z-[9999] bg-amber-50 border-b border-amber-200">
      <div className="max-w-screen-2xl mx-auto px-4 py-2.5 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <p className="text-sm text-amber-800 leading-relaxed flex-1">
          <span className="font-semibold">Research Demonstration Only.</span>{" "}
          All downstream models are trained on Path Foundation embeddings and are
          for demonstration purposes only. They do not reflect real-world
          clinical models &mdash; these models are trained on minimal data and
          have not been validated for clinical use.
        </p>
        <button
          onClick={() => setDismissed(true)}
          className="shrink-0 p-1 rounded hover:bg-amber-100 text-amber-600 hover:text-amber-800 transition-colors"
          aria-label="Dismiss disclaimer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
