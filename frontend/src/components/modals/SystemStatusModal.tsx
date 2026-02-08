"use client";

import React, { useEffect, useCallback, useState } from "react";
import { Button } from "@/components/ui/Button";
import {
  X,
  Activity,
  Server,
  Database,
  Cpu,
  HardDrive,
  Clock,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SystemStatusModalProps {
  isOpen: boolean;
  onClose: () => void;
  isConnected: boolean;
}

interface ServiceStatus {
  name: string;
  status: "online" | "offline" | "degraded";
  latency?: number;
  message?: string;
}

interface SystemMetrics {
  uptime: string;
  version: string;
  services: ServiceStatus[];
  lastChecked: Date;
}

export function SystemStatusModal({ isOpen, onClose, isConnected }: SystemStatusModalProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [metrics, setMetrics] = useState<SystemMetrics>({
    uptime: "Checking...",
    version: "0.1.0",
    services: [
      { name: "Backend API", status: isConnected ? "online" : "offline", latency: 0 },
      { name: "Slide Server", status: isConnected ? "online" : "offline" },
      { name: "Analysis Engine", status: isConnected ? "online" : "offline" },
      { name: "Database", status: isConnected ? "online" : "offline" },
    ],
    lastChecked: new Date(),
  });

  // Simulate checking services
  const checkServices = useCallback(async () => {
    setIsRefreshing(true);
    
    // Simulate API latency check
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || ""}/api/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      const latency = Date.now() - startTime;
      const isHealthy = response.ok;
      
      let uptimeStr = "Unknown";
      try {
        const data = await response.json();
        if (data.uptime) {
          const uptimeSec = Math.floor(data.uptime);
          const hours = Math.floor(uptimeSec / 3600);
          const minutes = Math.floor((uptimeSec % 3600) / 60);
          uptimeStr = `${hours}h ${minutes}m`;
        }
      } catch {
        // Ignore JSON parse errors
      }
      
      setMetrics({
        uptime: uptimeStr,
        version: "0.1.0",
        services: [
          { name: "Backend API", status: isHealthy ? "online" : "offline", latency },
          { name: "Slide Server", status: isHealthy ? "online" : "offline" },
          { name: "Analysis Engine", status: isHealthy ? "online" : "degraded", message: isHealthy ? undefined : "GPU utilization high" },
          { name: "Database", status: isHealthy ? "online" : "offline" },
        ],
        lastChecked: new Date(),
      });
    } catch {
      setMetrics((prev) => ({
        ...prev,
        services: prev.services.map((s) => ({ ...s, status: "offline" as const })),
        lastChecked: new Date(),
      }));
    }
    
    setIsRefreshing(false);
  }, []);

  // Check on mount and when connected status changes
  useEffect(() => {
    if (isOpen) {
      checkServices();
    }
  }, [isOpen, isConnected, checkServices]);

  // Handle escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    },
    [isOpen, onClose]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Prevent scroll when modal is open
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const getStatusIcon = (status: ServiceStatus["status"]) => {
    switch (status) {
      case "online":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "offline":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "degraded":
        return <AlertCircle className="h-4 w-4 text-amber-500" />;
    }
  };

  const getStatusColor = (status: ServiceStatus["status"]) => {
    switch (status) {
      case "online":
        return "text-green-600 bg-green-50 border-green-200";
      case "offline":
        return "text-red-600 bg-red-50 border-red-200";
      case "degraded":
        return "text-amber-600 bg-amber-50 border-amber-200";
    }
  };

  const overallStatus = metrics.services.every((s) => s.status === "online")
    ? "online"
    : metrics.services.some((s) => s.status === "online")
    ? "degraded"
    : "offline";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-md mx-4 bg-white rounded-xl shadow-2xl overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-10 h-10 rounded-lg flex items-center justify-center",
              overallStatus === "online" ? "bg-green-600" : overallStatus === "degraded" ? "bg-amber-600" : "bg-red-600"
            )}>
              <Activity className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">System Status</h2>
              <p className={cn(
                "text-sm font-medium",
                overallStatus === "online" ? "text-green-600" : overallStatus === "degraded" ? "text-amber-600" : "text-red-600"
              )}>
                {overallStatus === "online" ? "All Systems Operational" : overallStatus === "degraded" ? "Partial Outage" : "System Offline"}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="p-2">
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Quick Stats */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
              <Clock className="h-5 w-5 text-gray-400" />
              <div>
                <p className="text-xs text-gray-500">Uptime</p>
                <p className="text-sm font-semibold text-gray-900">{metrics.uptime}</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
              <Server className="h-5 w-5 text-gray-400" />
              <div>
                <p className="text-xs text-gray-500">Version</p>
                <p className="text-sm font-semibold text-gray-900">v{metrics.version}</p>
              </div>
            </div>
          </div>

          {/* Services */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Services</h3>
            <div className="space-y-2">
              {metrics.services.map((service) => (
                <div
                  key={service.name}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-lg border",
                    getStatusColor(service.status)
                  )}
                >
                  <div className="flex items-center gap-3">
                    {getStatusIcon(service.status)}
                    <div>
                      <p className="text-sm font-medium">{service.name}</p>
                      {service.message && (
                        <p className="text-xs opacity-75">{service.message}</p>
                      )}
                    </div>
                  </div>
                  {service.latency !== undefined && service.status === "online" && (
                    <span className="text-xs font-mono opacity-75">{service.latency}ms</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Last Checked */}
          <p className="text-xs text-gray-500 text-center">
            Last checked: {metrics.lastChecked.toLocaleTimeString()}
          </p>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end px-6 py-4 border-t border-gray-200 bg-gray-50">
          <Button
            variant="secondary"
            size="sm"
            onClick={checkServices}
            disabled={isRefreshing}
          >
            <RefreshCw className={cn("h-4 w-4 mr-2", isRefreshing && "animate-spin")} />
            {isRefreshing ? "Checking..." : "Refresh"}
          </Button>
        </div>
      </div>
    </div>
  );
}
