"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface LogoProps {
  size?: "sm" | "md" | "lg";
  variant?: "full" | "icon" | "mark";
  className?: string;
}

/**
 * Enso Atlas Logo - Professional pathology evidence engine branding
 * The logo combines a stylized microscope lens with concentric rings
 * representing the "enso" (circle of togetherness) concept and the
 * layered analysis of pathology slides.
 */
export function Logo({ size = "md", variant = "full", className }: LogoProps) {
  const sizeMap = {
    sm: { icon: 28, text: "text-sm" },
    md: { icon: 40, text: "text-lg" },
    lg: { icon: 56, text: "text-2xl" },
  };

  const { icon, text } = sizeMap[size];

  // For icon-only variant (alias for mark), just render the icon
  const showWordmark = variant === "full";

  return (
    <div className={cn("flex items-center gap-2 sm:gap-3", className)}>
      {/* Logo Mark - Stylized Enso with microscope lens */}
      <div
        className="relative flex items-center justify-center shrink-0"
        style={{ width: icon, height: icon }}
      >
        <svg
          viewBox="0 0 48 48"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-full"
        >
          {/* Outer ring - gradient stroke for depth */}
          <defs>
            <linearGradient id="logo-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#0ea5e9" />
              <stop offset="50%" stopColor="#0284c7" />
              <stop offset="100%" stopColor="#0369a1" />
            </linearGradient>
            <linearGradient id="logo-inner" x1="0%" y1="100%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#0ea5e9" />
            </linearGradient>
            {/* Glow effect filter */}
            <filter id="logo-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="1" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          
          {/* Background circle with subtle fill */}
          <circle
            cx="24"
            cy="24"
            r="22"
            fill="url(#logo-gradient)"
            opacity="0.1"
          />
          
          {/* Main enso ring - thick stroke with gap (zen circle) */}
          <path
            d="M 24 4 A 20 20 0 1 1 12 8"
            stroke="url(#logo-gradient)"
            strokeWidth="3.5"
            strokeLinecap="round"
            fill="none"
            filter="url(#logo-glow)"
          />
          
          {/* Inner analysis rings - concentric layers */}
          <circle
            cx="24"
            cy="24"
            r="14"
            stroke="url(#logo-inner)"
            strokeWidth="1.5"
            strokeDasharray="4 3"
            fill="none"
            opacity="0.7"
          />
          
          <circle
            cx="24"
            cy="24"
            r="8"
            stroke="url(#logo-inner)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.9"
          />
          
          {/* Center point - focus indicator */}
          <circle
            cx="24"
            cy="24"
            r="3"
            fill="url(#logo-gradient)"
          />
          
          {/* Crosshair elements - precision targeting */}
          <line x1="24" y1="17" x2="24" y2="20" stroke="#0ea5e9" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="24" y1="28" x2="24" y2="31" stroke="#0ea5e9" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="17" y1="24" x2="20" y2="24" stroke="#0ea5e9" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="28" y1="24" x2="31" y2="24" stroke="#0ea5e9" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>

      {/* Wordmark */}
      {showWordmark && (
        <div className="flex flex-col min-w-0">
          <span className={cn("font-semibold tracking-tight text-white leading-none", text)}>
            Enso<span className="text-clinical-400">Atlas</span>
          </span>
          <span className="text-[10px] text-gray-400 tracking-wider uppercase mt-0.5 hidden sm:block">
            Pathology Evidence Engine
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * Animated loading state for the logo
 */
export function LogoLoader({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sizeMap = { sm: 28, md: 40, lg: 56 };
  const s = sizeMap[size];

  return (
    <div className="relative flex items-center justify-center" style={{ width: s, height: s }}>
      <svg
        viewBox="0 0 48 48"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full animate-pulse"
      >
        <circle cx="24" cy="24" r="20" stroke="#0ea5e9" strokeWidth="3" strokeDasharray="10 5" fill="none" className="animate-spin" style={{ animationDuration: "3s" }} />
        <circle cx="24" cy="24" r="12" stroke="#38bdf8" strokeWidth="2" fill="none" opacity="0.5" className="animate-ping" style={{ animationDuration: "1.5s" }} />
        <circle cx="24" cy="24" r="4" fill="#0ea5e9" />
      </svg>
    </div>
  );
}
