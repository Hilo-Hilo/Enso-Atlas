"use client";

import React from "react";
import { Shield, Lock, FileWarning, ExternalLink, HelpCircle } from "lucide-react";

interface FooterProps {
  version?: string;
  buildDate?: string;
}

export function Footer({ version = "0.1.0", buildDate }: FooterProps) {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="h-12 bg-surface-secondary border-t border-surface-border px-6 flex items-center justify-between shrink-0">
      {/* Left: Regulatory Disclaimers */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-amber-600">
          <FileWarning className="h-4 w-4" />
          <span className="text-xs font-medium">
            For Research Use Only - Not for Clinical Diagnosis or Treatment Decisions
          </span>
        </div>
        <div className="h-4 w-px bg-gray-200" />
        <div className="flex items-center gap-1.5 text-gray-500">
          <Lock className="h-3.5 w-3.5 text-clinical-600" />
          <span className="text-xs">Local Processing Only</span>
        </div>
      </div>

      {/* Center: Powered By */}
      <div className="flex items-center gap-1.5 text-gray-400">
        <span className="text-xs">Powered by</span>
        <span className="text-xs font-medium text-gray-600">
          Path Foundation + MedGemma
        </span>
      </div>

      {/* Right: Version, Help, and Security */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <div className="flex items-center gap-1.5">
          <Shield className="h-3.5 w-3.5 text-status-positive" />
          <span>HIPAA Compliant Architecture</span>
        </div>
        <div className="h-4 w-px bg-gray-200" />
        <button className="flex items-center gap-1 hover:text-clinical-600 transition-colors">
          <HelpCircle className="h-3.5 w-3.5" />
          <span>Documentation</span>
          <ExternalLink className="h-3 w-3" />
        </button>
        <div className="h-4 w-px bg-gray-200" />
        <span className="font-mono text-gray-400">
          v{version}
          {buildDate && ` (${buildDate})`}
        </span>
      </div>
    </footer>
  );
}
