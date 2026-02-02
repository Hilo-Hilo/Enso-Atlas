"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertCircle, RefreshCw, ChevronDown, ChevronUp, Bug, Copy, Check } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetKeys?: unknown[];
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
  copied: boolean;
}

/**
 * Error Boundary component that catches JavaScript errors in child components
 * and displays a fallback UI with error details and recovery options.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null,
      showDetails: false,
      copied: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    
    // Call optional error handler
    this.props.onError?.(error, errorInfo);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    // Reset error state when resetKeys change
    if (this.state.hasError && this.props.resetKeys) {
      const prevResetKeys = prevProps.resetKeys || [];
      const currentResetKeys = this.props.resetKeys;
      
      const keysChanged = currentResetKeys.some(
        (key, index) => key !== prevResetKeys[index]
      );
      
      if (keysChanged) {
        this.handleReset();
      }
    }
  }

  handleReset = () => {
    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null,
      showDetails: false,
      copied: false,
    });
  };

  handleRefresh = () => {
    this.handleReset();
    window.location.reload();
  };

  handleCopyError = async () => {
    const { error, errorInfo } = this.state;
    const errorReport = [
      "=== Error Report ===",
      `Time: ${new Date().toISOString()}`,
      `URL: ${typeof window !== "undefined" ? window.location.href : "N/A"}`,
      "",
      "Error:",
      error?.message || "Unknown error",
      "",
      "Stack:",
      error?.stack || "No stack trace",
      "",
      "Component Stack:",
      errorInfo?.componentStack || "No component stack",
    ].join("\n");

    try {
      await navigator.clipboard.writeText(errorReport);
      this.setState({ copied: true });
      setTimeout(() => this.setState({ copied: false }), 2000);
    } catch (err) {
      console.error("Failed to copy error:", err);
    }
  };

  toggleDetails = () => {
    this.setState(prev => ({ showDetails: !prev.showDetails }));
  };

  render() {
    const { hasError, error, errorInfo, showDetails, copied } = this.state;
    const { children, fallback } = this.props;

    if (hasError) {
      if (fallback) {
        return fallback;
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
          <div className="max-w-lg w-full bg-white rounded-xl shadow-lg overflow-hidden">
            {/* Header */}
            <div className="bg-red-50 border-b border-red-100 px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                  <AlertCircle className="h-6 w-6 text-red-500" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    Something went wrong
                  </h2>
                  <p className="text-sm text-gray-600">
                    An unexpected error occurred in the application.
                  </p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="px-6 py-4">
              {/* Error message */}
              <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <p className="text-sm font-mono text-gray-700 break-all">
                  {error?.message || "Unknown error"}
                </p>
              </div>

              {/* Error details (collapsible) */}
              <div className="mb-4">
                <button
                  onClick={this.toggleDetails}
                  className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  {showDetails ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                  <Bug className="h-4 w-4" />
                  <span>{showDetails ? "Hide" : "Show"} technical details</span>
                </button>

                {showDetails && (
                  <div className="mt-3 space-y-3">
                    {error?.stack && (
                      <div className="p-3 bg-gray-900 rounded-lg overflow-x-auto">
                        <p className="text-xs font-mono text-gray-300 whitespace-pre-wrap">
                          {error.stack}
                        </p>
                      </div>
                    )}
                    {errorInfo?.componentStack && (
                      <div className="p-3 bg-gray-900 rounded-lg overflow-x-auto">
                        <p className="text-xs text-gray-400 mb-1">Component Stack:</p>
                        <p className="text-xs font-mono text-gray-300 whitespace-pre-wrap">
                          {errorInfo.componentStack}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Recovery suggestions */}
              <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
                <p className="text-sm text-blue-800 font-medium mb-1">
                  What you can try:
                </p>
                <ul className="text-sm text-blue-700 list-disc list-inside space-y-1">
                  <li>Refresh the page to start fresh</li>
                  <li>Clear your browser cache and cookies</li>
                  <li>Try again in a few minutes</li>
                  <li>Contact support if the issue persists</li>
                </ul>
              </div>
            </div>

            {/* Actions */}
            <div className="px-6 py-4 bg-gray-50 border-t border-gray-100 flex items-center justify-between gap-3">
              <button
                onClick={this.handleCopyError}
                className="inline-flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4 text-green-500" />
                    <span className="text-green-600">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    <span>Copy error</span>
                  </>
                )}
              </button>
              <div className="flex items-center gap-2">
                <button
                  onClick={this.handleReset}
                  className="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  Try Again
                </button>
                <button
                  onClick={this.handleRefresh}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-clinical-600 text-white text-sm rounded-lg hover:bg-clinical-700 transition-colors"
                >
                  <RefreshCw className="h-4 w-4" />
                  Refresh Page
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return children;
  }
}

/**
 * Panel-level error boundary with compact fallback UI
 */
interface PanelErrorBoundaryProps {
  children: ReactNode;
  panelName?: string;
  onRetry?: () => void;
}

interface PanelErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class PanelErrorBoundary extends Component<
  PanelErrorBoundaryProps,
  PanelErrorBoundaryState
> {
  constructor(props: PanelErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): PanelErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`PanelErrorBoundary (${this.props.panelName || "unnamed"}) caught:`, error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
    this.props.onRetry?.();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-800">
                {this.props.panelName 
                  ? `Failed to load ${this.props.panelName}`
                  : "Component error"
                }
              </p>
              <p className="text-xs text-red-600 mt-1">
                {this.state.error?.message || "An error occurred"}
              </p>
              <button
                onClick={this.handleRetry}
                className="mt-2 inline-flex items-center gap-1 text-xs text-red-700 hover:text-red-900 font-medium"
              >
                <RefreshCw className="h-3 w-3" />
                Try again
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
