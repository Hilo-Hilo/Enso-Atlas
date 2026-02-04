"use client";

import React, { createContext, useContext, useState, useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";
import { CheckCircle, XCircle, AlertTriangle, Info, X, Loader2 } from "lucide-react";

// Toast types
export type ToastType = "success" | "error" | "warning" | "info" | "loading";

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number; // ms, 0 = permanent
  progress?: number; // 0-100 for loading toasts
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, "id">) => string;
  removeToast: (id: string) => void;
  updateToast: (id: string, updates: Partial<Omit<Toast, "id">>) => void;
  // Convenience methods
  success: (title: string, message?: string) => string;
  error: (title: string, message?: string) => string;
  warning: (title: string, message?: string) => string;
  info: (title: string, message?: string) => string;
  loading: (title: string, message?: string) => string;
  // Legacy API compatibility
  showToast: (opts: { type?: "success" | "error" | "warning" | "info" | "loading"; message: string; title?: string }) => string;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}

const TOAST_ICONS: Record<ToastType, React.ReactNode> = {
  success: <CheckCircle className="h-5 w-5 text-green-500" />,
  error: <XCircle className="h-5 w-5 text-red-500" />,
  warning: <AlertTriangle className="h-5 w-5 text-amber-500" />,
  info: <Info className="h-5 w-5 text-blue-500" />,
  loading: <Loader2 className="h-5 w-5 text-clinical-500 animate-spin" />,
};

const TOAST_STYLES: Record<ToastType, string> = {
  success: "border-green-200 bg-green-50",
  error: "border-red-200 bg-red-50",
  warning: "border-amber-200 bg-amber-50",
  info: "border-blue-200 bg-blue-50",
  loading: "border-clinical-200 bg-clinical-50",
};

const DEFAULT_DURATIONS: Record<ToastType, number> = {
  success: 4000,
  error: 6000,
  warning: 5000,
  info: 4000,
  loading: 0, // permanent until removed/updated
};

function ToastItem({
  toast,
  onRemove,
}: {
  toast: Toast;
  onRemove: () => void;
}) {
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(onRemove, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.duration, onRemove]);

  return (
    <div
      className={cn(
        "flex items-start gap-3 p-4 rounded-lg border shadow-lg backdrop-blur-sm",
        "animate-in slide-in-from-right-full duration-300",
        TOAST_STYLES[toast.type]
      )}
      role="alert"
    >
      <div className="shrink-0 mt-0.5">
        {TOAST_ICONS[toast.type]}
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium text-sm text-gray-900">{toast.title}</p>
        {toast.message && (
          <p className="mt-1 text-xs text-gray-600">{toast.message}</p>
        )}
        {toast.type === "loading" && toast.progress !== undefined && (
          <div className="mt-2">
            <div className="flex justify-between text-2xs text-gray-500 mb-1">
              <span>Progress</span>
              <span>{Math.round(toast.progress)}%</span>
            </div>
            <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-clinical-500 rounded-full transition-all duration-300"
                style={{ width: `${toast.progress}%` }}
              />
            </div>
          </div>
        )}
      </div>
      {toast.type !== "loading" && (
        <button
          onClick={onRemove}
          className="shrink-0 p-1 rounded hover:bg-black/5 transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4 text-gray-400" />
        </button>
      )}
    </div>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((toast: Omit<Toast, "id">): string => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const duration = toast.duration ?? DEFAULT_DURATIONS[toast.type];
    setToasts((prev) => [...prev, { ...toast, id, duration }]);
    return id;
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const updateToast = useCallback((id: string, updates: Partial<Omit<Toast, "id">>) => {
    setToasts((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...updates } : t))
    );
  }, []);

  // Convenience methods
  const success = useCallback((title: string, message?: string) => 
    addToast({ type: "success", title, message }), [addToast]);
  
  const error = useCallback((title: string, message?: string) => 
    addToast({ type: "error", title, message }), [addToast]);
  
  const warning = useCallback((title: string, message?: string) => 
    addToast({ type: "warning", title, message }), [addToast]);
  
  const info = useCallback((title: string, message?: string) => 
    addToast({ type: "info", title, message }), [addToast]);
  
  const loading = useCallback((title: string, message?: string) => 
    addToast({ type: "loading", title, message }), [addToast]);

  const showToast = (opts: { type?: "success" | "error" | "warning" | "info" | "loading"; message: string; title?: string }) => {
    const type = opts.type || "info";
    return addToast({ type, title: opts.title || opts.message, message: opts.message });
  };

  const value: ToastContextValue = {
    toasts,
    addToast,
    removeToast,
    updateToast,
    success,
    error,
    warning,
    info,
    loading,
    showToast,
  };

  return (
    <ToastContext.Provider value={value}>
      {children}
      {/* Toast Container */}
      <div
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none"
        aria-live="polite"
      >
        {toasts.map((toast) => (
          <div key={toast.id} className="pointer-events-auto">
            <ToastItem toast={toast} onRemove={() => removeToast(toast.id)} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
