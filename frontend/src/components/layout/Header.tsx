"use client";

import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Logo } from "@/components/ui/Logo";
import { DemoToggle } from "@/components/demo";
import { UserDropdown } from "./UserDropdown";
import { SettingsModal } from "@/components/modals/SettingsModal";
import { SystemStatusModal } from "@/components/modals/SystemStatusModal";
import {
  Microscope,
  Settings,
  HelpCircle,
  Activity,
  User,
  Building2,
  ChevronDown,
  Keyboard,
  Stethoscope,
  Layers,
  WifiOff,
  RefreshCw,
  AlertTriangle,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type UserViewMode = "oncologist" | "pathologist" | "batch";

interface HeaderProps {
  isConnected?: boolean;
  isProcessing?: boolean;
  version?: string;
  institutionName?: string;
  userName?: string;
  onOpenShortcuts?: () => void;
  viewMode?: UserViewMode;
  onViewModeChange?: (mode: UserViewMode) => void;
  demoMode?: boolean;
  onDemoModeToggle?: () => void;
  onReconnect?: () => void;
}

// Disconnection Banner Component
function DisconnectionBanner({ 
  onReconnect, 
  onDismiss,
  isReconnecting 
}: { 
  onReconnect?: () => void; 
  onDismiss: () => void;
  isReconnecting: boolean;
}) {
  return (
    <div className="bg-gradient-to-r from-red-600 via-red-500 to-rose-600 text-white px-4 py-2 flex items-center justify-center gap-4 animate-slide-down">
      {/* Pulsing icon */}
      <div className="flex items-center gap-2">
        <div className="relative">
          <WifiOff className="h-5 w-5 animate-pulse" />
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
          </span>
        </div>
        <span className="font-semibold text-sm">Backend Disconnected</span>
      </div>

      {/* Message */}
      <span className="text-sm text-red-100 hidden sm:inline">
        Unable to connect to the analysis server. Some features may be unavailable.
      </span>

      {/* Actions */}
      <div className="flex items-center gap-2">
        {onReconnect && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onReconnect}
            disabled={isReconnecting}
            className="bg-white/20 hover:bg-white/30 text-white border-white/30 text-xs px-3 py-1"
          >
            {isReconnecting ? (
              <>
                <RefreshCw className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                Reconnecting...
              </>
            ) : (
              <>
                <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                Reconnect
              </>
            )}
          </Button>
        )}
        <button
          onClick={onDismiss}
          className="p-1 hover:bg-white/20 rounded transition-colors"
          title="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export function Header({
  isConnected = false,
  isProcessing = false,
  version = "0.1.0",
  institutionName = "Research Laboratory",
  userName,
  onOpenShortcuts,
  viewMode = "oncologist",
  onViewModeChange,
  demoMode = false,
  onDemoModeToggle,
  onReconnect,
}: HeaderProps) {
  // Modal and dropdown state
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [statusOpen, setStatusOpen] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);

  // Reset banner dismissed state when connection changes
  useEffect(() => {
    if (isConnected) {
      setBannerDismissed(false);
      setIsReconnecting(false);
    }
  }, [isConnected]);

  const handleReconnect = async () => {
    if (onReconnect) {
      setIsReconnecting(true);
      try {
        await onReconnect();
      } finally {
        // Keep reconnecting state for a bit to show feedback
        setTimeout(() => setIsReconnecting(false), 2000);
      }
    }
  };

  const handleOpenDocs = () => {
    window.open("https://github.com/Enso-Labs/medgemma", "_blank");
  };

  const showDisconnectionBanner = !isConnected && !bannerDismissed;

  return (
    <>
      {/* Disconnection Banner - Shows at top when disconnected */}
      {showDisconnectionBanner && (
        <DisconnectionBanner
          onReconnect={handleReconnect}
          onDismiss={() => setBannerDismissed(true)}
          isReconnecting={isReconnecting}
        />
      )}

      <header className="h-16 bg-gradient-to-r from-navy-900 via-navy-900 to-navy-800 border-b border-navy-700/50 px-6 flex items-center justify-between shrink-0 shadow-lg">
        {/* Left: Logo and Branding */}
        <div className="flex items-center gap-4">
          {/* Professional Logo */}
          <div className="flex items-center gap-3">
            <Logo size="md" variant="full" />
            <span className="text-xs text-navy-100 font-mono bg-navy-800/80 px-2 py-0.5 rounded-full border border-navy-700/50 hidden sm:inline-flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-clinical-400 animate-pulse" />
              v{version}
            </span>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-gradient-to-b from-transparent via-navy-600 to-transparent mx-2 hidden md:block" />

          {/* Institution Context */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-navy-800/50 rounded-lg border border-navy-700/30">
            <Building2 className="h-4 w-4 text-clinical-400" />
            <span className="text-sm text-gray-300 font-medium">{institutionName}</span>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-gradient-to-b from-transparent via-navy-600 to-transparent mx-2 hidden lg:block" />

          {/* View Mode Toggle */}
          {onViewModeChange && (
            <div className="hidden lg:flex items-center bg-navy-800/80 rounded-xl p-1 border border-navy-700/30 shadow-inner">
              <button
                onClick={() => onViewModeChange("oncologist")}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                  viewMode === "oncologist"
                    ? "bg-gradient-to-r from-clinical-600 to-clinical-500 text-white shadow-md shadow-clinical-600/30"
                    : "text-gray-400 hover:text-white hover:bg-navy-700/50"
                )}
              >
                <Stethoscope className="h-4 w-4" />
                <span>Oncologist</span>
              </button>
              <button
                onClick={() => onViewModeChange("pathologist")}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                  viewMode === "pathologist"
                    ? "bg-gradient-to-r from-violet-600 to-violet-500 text-white shadow-md shadow-violet-600/30"
                    : "text-gray-400 hover:text-white hover:bg-navy-700/50"
                )}
              >
                <Microscope className="h-4 w-4" />
                <span>Pathologist</span>
              </button>
              <button
                onClick={() => onViewModeChange("batch")}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                  viewMode === "batch"
                    ? "bg-gradient-to-r from-amber-600 to-amber-500 text-white shadow-md shadow-amber-600/30"
                    : "text-gray-400 hover:text-white hover:bg-navy-700/50"
                )}
              >
                <Layers className="h-4 w-4" />
                <span>Batch</span>
              </button>
            </div>
          )}
        </div>

        {/* Center: Status Indicators */}
        <div className="hidden md:flex items-center gap-6">
          {/* Backend Connection Status - Enhanced for disconnected state */}
          <button
            onClick={() => setStatusOpen(true)}
            className={cn(
              "flex items-center gap-2.5 px-3 py-1.5 rounded-lg border transition-all",
              isConnected
                ? "bg-navy-800/50 border-navy-700/30 hover:bg-navy-700/50"
                : "bg-red-900/50 border-red-700/50 hover:bg-red-800/50 animate-pulse"
            )}
          >
            <div
              className={cn(
                "relative flex h-2.5 w-2.5",
                isConnected ? "text-status-positive" : "text-status-negative"
              )}
            >
              {isConnected && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-positive opacity-75" />
              )}
              <span className={cn(
                "relative inline-flex rounded-full h-2.5 w-2.5",
                isConnected ? "bg-status-positive" : "bg-status-negative"
              )} />
            </div>
            <div className="flex flex-col">
              <span className={cn(
                "text-xs font-medium flex items-center gap-1",
                isConnected ? "text-status-positive" : "text-red-400"
              )}>
                {isConnected ? "Connected" : (
                  <>
                    <AlertTriangle className="h-3 w-3" />
                    Disconnected
                  </>
                )}
              </span>
              <span className="text-2xs text-gray-500">Backend Service</span>
            </div>
          </button>

          {/* Processing Status */}
          {isProcessing && (
            <>
              <div className="h-6 w-px bg-navy-700" />
              <div className="flex items-center gap-2 px-3 py-1.5 bg-clinical-600/20 rounded-lg border border-clinical-500/30">
                <div className="flex items-center gap-1.5">
                  <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-clinical-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-clinical-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-clinical-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                  <span className="text-xs text-clinical-300 font-medium">Processing</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Right: Actions and User */}
        <div className="flex items-center gap-3">
          {/* Demo Mode Toggle */}
          {onDemoModeToggle && (
            <DemoToggle isActive={demoMode} onToggle={onDemoModeToggle} />
          )}

          {/* Research Mode Warning */}
          <Badge variant="warning" size="sm" className="font-semibold shadow-sm hidden sm:inline-flex">
            Research Use Only
          </Badge>

          {/* System Status */}
          <div className="flex items-center gap-0.5 bg-navy-800/50 rounded-lg p-1 border border-navy-700/30">
            <Button
              variant="ghost"
              size="sm"
              className="p-2 text-gray-400 hover:text-white hover:bg-navy-700 rounded-md transition-all duration-150"
              title="System Status"
              onClick={() => setStatusOpen(true)}
            >
              <Activity className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="p-2 text-gray-400 hover:text-white hover:bg-navy-700 rounded-md transition-all duration-150"
              title="Keyboard Shortcuts (?)"
              onClick={onOpenShortcuts}
            >
              <Keyboard className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="p-2 text-gray-400 hover:text-white hover:bg-navy-700 rounded-md transition-all duration-150 hidden sm:inline-flex"
              title="Documentation"
              onClick={handleOpenDocs}
            >
              <HelpCircle className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="p-2 text-gray-400 hover:text-white hover:bg-navy-700 rounded-md transition-all duration-150 hidden sm:inline-flex"
              title="Settings"
              onClick={() => setSettingsOpen(true)}
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-gradient-to-b from-transparent via-navy-600 to-transparent hidden sm:block" />

          {/* User Context */}
          <div className="relative">
            <button
              onClick={() => setUserDropdownOpen(!userDropdownOpen)}
              className="flex items-center gap-2 px-2 py-1.5 rounded-xl hover:bg-navy-800/80 transition-all duration-200 border border-transparent hover:border-navy-700/50 group"
            >
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-clinical-600 to-clinical-700 flex items-center justify-center shadow-md group-hover:shadow-clinical-600/30 transition-shadow">
                <User className="h-4 w-4 text-white" />
              </div>
              {userName && (
                <>
                  <span className="text-sm text-gray-200 font-medium hidden lg:inline">{userName}</span>
                  <ChevronDown className={cn(
                    "h-3.5 w-3.5 text-gray-400 group-hover:text-gray-300 transition-all hidden lg:inline",
                    userDropdownOpen && "rotate-180"
                  )} />
                </>
              )}
            </button>

            {/* User Dropdown */}
            <UserDropdown
              isOpen={userDropdownOpen}
              onClose={() => setUserDropdownOpen(false)}
              userName={userName}
              userRole="Researcher"
              onOpenSettings={() => setSettingsOpen(true)}
            />
          </div>
        </div>
      </header>

      {/* Modals */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
      <SystemStatusModal
        isOpen={statusOpen}
        onClose={() => setStatusOpen(false)}
        isConnected={isConnected}
      />
    </>
  );
}
