"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info" | "clinical";
  size?: "sm" | "md" | "lg";
  dot?: boolean;
  pulse?: boolean;
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = "default", size = "md", dot = false, pulse = false, children, ...props }, ref) => {
    const variants = {
      default: "bg-gray-100 text-gray-700 border-gray-200 dark:bg-navy-700 dark:text-gray-300 dark:border-navy-600",
      success: "bg-gradient-to-r from-green-50 to-emerald-50 text-green-800 border-green-200 dark:from-green-900/30 dark:to-emerald-900/30 dark:text-green-300 dark:border-green-700 dark:border-green-800",
      warning: "bg-gradient-to-r from-amber-50 to-yellow-50 text-amber-800 border-amber-200 dark:from-amber-900/30 dark:to-yellow-900/30 dark:text-amber-300 dark:border-amber-700",
      danger: "bg-gradient-to-r from-red-50 to-rose-50 text-red-800 border-red-200 dark:from-red-900/30 dark:to-rose-900/30 dark:text-red-300 dark:border-red-700 dark:border-red-800",
      info: "bg-gradient-to-r from-blue-50 to-sky-50 text-blue-800 border-blue-200 dark:from-blue-900/30 dark:to-sky-900/30 dark:text-blue-300 dark:border-blue-700",
      clinical: "bg-gradient-to-r from-clinical-50 to-cyan-50 text-clinical-800 border-clinical-200 dark:from-clinical-900/30 dark:to-cyan-900/30 dark:text-clinical-300 dark:border-clinical-700",
    };

    const dotColors = {
      default: "bg-gray-400",
      success: "bg-green-500",
      warning: "bg-amber-500",
      danger: "bg-red-500",
      info: "bg-blue-500",
      clinical: "bg-clinical-500",
    };

    const sizes = {
      sm: "px-2 py-0.5 text-xs gap-1.5",
      md: "px-2.5 py-1 text-sm gap-1.5",
      lg: "px-3 py-1.5 text-sm gap-2",
    };

    return (
      <span
        ref={ref}
        className={cn(
          "inline-flex items-center font-medium rounded-full border transition-all duration-200 shadow-sm",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {dot && (
          <span className="relative flex h-2 w-2">
            {pulse && (
              <span className={cn(
                "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75",
                dotColors[variant]
              )} />
            )}
            <span className={cn(
              "relative inline-flex rounded-full h-2 w-2",
              dotColors[variant]
            )} />
          </span>
        )}
        {children}
      </span>
    );
  }
);

Badge.displayName = "Badge";
