"use client";

import React from "react";
import { Shield, Lock, FileWarning, ExternalLink, HelpCircle } from "lucide-react";

interface FooterProps {
  version?: string;
  buildDate?: string;
}

export function Footer({ version = "0.1.0", buildDate }: FooterProps) {
  return (
    <footer className="h-8 sm:h-9 bg-gradient-to-r from-surface-secondary via-white to-surface-secondary dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 border-t border-surface-border dark:border-slate-700 px-3 sm:px-4 lg:px-5 flex items-center justify-between shrink-0">
      {/* Left: Regulatory Disclaimers */}
      <div className="flex items-center gap-2 sm:gap-4 lg:gap-6">
        <div className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 bg-amber-50 dark:bg-amber-900/30 rounded-full border border-amber-200 dark:border-amber-700">
          <FileWarning className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-amber-600 dark:text-amber-400" />
          <span className="text-2xs sm:text-xs font-semibold text-amber-700 dark:text-amber-300">
            Research Only
          </span>
        </div>
        <div className="hidden lg:flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
          <Lock className="h-3.5 w-3.5 text-clinical-600" />
          <span className="text-xs font-medium">Local Processing</span>
        </div>
      </div>

      {/* Center: Powered By - Hidden on small screens */}
      <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-gray-50 dark:bg-slate-800 rounded-full border border-gray-200 dark:border-slate-700">
        <span className="text-2xs sm:text-xs text-gray-400 dark:text-gray-500">Powered by</span>
        <span className="text-2xs sm:text-xs font-semibold bg-gradient-to-r from-clinical-600 to-violet-600 bg-clip-text text-transparent">
          Path Foundation + MedGemma
        </span>
      </div>

      {/* Right: Version, Help, and Security */}
      <div className="flex items-center gap-2 sm:gap-3 text-xs text-gray-500">
        <div className="hidden sm:flex items-center gap-1.5 px-2 sm:px-2.5 py-1 bg-green-50 dark:bg-green-900/30 rounded-full border border-green-200 dark:border-green-700">
          <Shield className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-status-positive" />
          <span className="text-green-700 dark:text-green-300 font-medium text-2xs sm:text-xs">HIPAA Ready</span>
        </div>
        <div className="h-4 w-px bg-gray-200 dark:bg-slate-700 hidden sm:block" />
        <a
          href="https://github.com/Hilo-Hilo/med-gemma-hackathon#readme"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:flex items-center gap-1.5 hover:text-clinical-600 dark:hover:text-clinical-400 transition-colors duration-150 group"
        >
          <HelpCircle className="h-3.5 w-3.5" />
          <span className="font-medium hidden md:inline">Docs</span>
          <ExternalLink className="h-3 w-3 opacity-50 group-hover:opacity-100 transition-opacity hidden md:inline" />
        </a>
        <span className="font-mono text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-slate-800 px-1.5 sm:px-2 py-0.5 rounded text-2xs sm:text-xs">
          v{version}
        </span>
      </div>
    </footer>
  );
}
