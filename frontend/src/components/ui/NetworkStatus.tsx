"use client";

import React, { useEffect, useState } from "react";
import { Wifi, WifiOff, RefreshCw, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import { 
  checkConnection, 
  onConnectionStateChange, 
  getConnectionState 
} from "@/lib/api";

type ConnectionState = "connected" | "disconnected" | "connecting" | "error";

interface NetworkStatusProps {
  className?: string;
  showWhenConnected?: boolean;
  onConnectionChange?: (connected: boolean) => void;
}

/**
 * Network status indicator that shows backend connection state.
 * Automatically monitors connection and provides visual feedback.
 */
export function NetworkStatus({ 
  className, 
  showWhenConnected = false,
  onConnectionChange,
}: NetworkStatusProps) {
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    getConnectionState()
  );
  const [isRetrying, setIsRetrying] = useState(false);
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    // Subscribe to connection state changes
    const unsubscribe = onConnectionStateChange((state) => {
      setConnectionState(state);
      onConnectionChange?.(state === "connected");
      
      // Show banner on disconnect, hide on connect
      if (state === "disconnected" || state === "error") {
        setShowBanner(true);
      } else if (state === "connected") {
        // Delay hiding to show success state briefly
        setTimeout(() => setShowBanner(false), 2000);
      }
    });

    // Initial connection check
    checkConnection();

    // Set up periodic health checks
    const interval = setInterval(() => {
      checkConnection();
    }, 30000); // Check every 30 seconds

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [onConnectionChange]);

  const handleRetry = async () => {
    setIsRetrying(true);
    await checkConnection();
    setIsRetrying(false);
  };

  // Don't render anything when connected (unless showWhenConnected is true)
  if (connectionState === "connected" && !showWhenConnected && !showBanner) {
    return null;
  }

  // Compact indicator for header
  if (showWhenConnected) {
    return (
      <div 
        className={cn(
          "flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium transition-colors",
          connectionState === "connected" && "bg-green-100 text-green-700",
          connectionState === "disconnected" && "bg-red-100 text-red-700",
          connectionState === "connecting" && "bg-yellow-100 text-yellow-700",
          connectionState === "error" && "bg-orange-100 text-orange-700",
          className
        )}
        title={
          connectionState === "connected" ? "Backend connected" :
          connectionState === "disconnected" ? "Backend disconnected" :
          connectionState === "connecting" ? "Connecting..." :
          "Connection error"
        }
      >
        {connectionState === "connected" && (
          <>
            <Wifi className="h-3 w-3" />
            <span>Connected</span>
          </>
        )}
        {connectionState === "disconnected" && (
          <>
            <WifiOff className="h-3 w-3" />
            <span>Disconnected</span>
          </>
        )}
        {connectionState === "connecting" && (
          <>
            <RefreshCw className="h-3 w-3 animate-spin" />
            <span>Connecting</span>
          </>
        )}
        {connectionState === "error" && (
          <>
            <AlertTriangle className="h-3 w-3" />
            <span>Error</span>
          </>
        )}
      </div>
    );
  }

  // Full banner for disconnected state
  if (!showBanner) return null;

  return (
    <div
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-transform duration-300",
        showBanner ? "translate-y-0" : "-translate-y-full",
        className
      )}
    >
      <div
        className={cn(
          "flex items-center justify-center gap-3 px-4 py-2 text-sm",
          connectionState === "connected" && "bg-green-500 text-white",
          connectionState === "disconnected" && "bg-red-500 text-white",
          connectionState === "connecting" && "bg-yellow-500 text-white",
          connectionState === "error" && "bg-orange-500 text-white"
        )}
      >
        {connectionState === "connected" ? (
          <>
            <Wifi className="h-4 w-4" />
            <span>Connection restored</span>
          </>
        ) : connectionState === "disconnected" ? (
          <>
            <WifiOff className="h-4 w-4" />
            <span>Unable to connect to server. Some features may be unavailable.</span>
            <button
              onClick={handleRetry}
              disabled={isRetrying}
              className="inline-flex items-center gap-1 px-2 py-1 bg-white/20 rounded hover:bg-white/30 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={cn("h-3 w-3", isRetrying && "animate-spin")} />
              <span>{isRetrying ? "Retrying..." : "Retry"}</span>
            </button>
          </>
        ) : connectionState === "connecting" ? (
          <>
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Connecting to server...</span>
          </>
        ) : (
          <>
            <AlertTriangle className="h-4 w-4" />
            <span>Server error. Please try again later.</span>
            <button
              onClick={handleRetry}
              disabled={isRetrying}
              className="inline-flex items-center gap-1 px-2 py-1 bg-white/20 rounded hover:bg-white/30 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={cn("h-3 w-3", isRetrying && "animate-spin")} />
              <span>{isRetrying ? "Retrying..." : "Retry"}</span>
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Hook for programmatic access to connection state
 */
export function useNetworkStatus() {
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    getConnectionState()
  );

  useEffect(() => {
    const unsubscribe = onConnectionStateChange(setConnectionState);
    return unsubscribe;
  }, []);

  return {
    isConnected: connectionState === "connected",
    isDisconnected: connectionState === "disconnected",
    isConnecting: connectionState === "connecting",
    hasError: connectionState === "error",
    connectionState,
    retry: checkConnection,
  };
}

export default NetworkStatus;
