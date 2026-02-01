"use client";

import React from "react";
import { Shield, Lock, FileWarning } from "lucide-react";

export function Footer() {
  return (
    <footer className="h-10 bg-surface-secondary border-t border-surface-border px-4 flex items-center justify-between shrink-0">
      {/* Left - Disclaimers */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <div className="flex items-center gap-1.5">
          <FileWarning className="h-3.5 w-3.5 text-amber-500" />
          <span>For Research Use Only - Not for Clinical Diagnosis</span>
        </div>
      </div>

      {/* Right - Security and Info */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <div className="flex items-center gap-1.5">
          <Lock className="h-3.5 w-3.5 text-clinical-600" />
          <span>Local Processing - No PHI Transmission</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Shield className="h-3.5 w-3.5 text-status-positive" />
          <span>Offline Mode</span>
        </div>
        <span className="text-gray-400">
          Enso Atlas - Powered by Path Foundation + MedGemma
        </span>
      </div>
    </footer>
  );
}
