"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
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
  ChevronDown,
  Keyboard,
  Stethoscope,
  Layers,
  WifiOff,
  RefreshCw,
  AlertTriangle,
  X,
  Menu,
  FolderOpen,
  MoreHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useProject } from "@/contexts/ProjectContext";

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
  isReconnecting,
}: {
  onReconnect?: () => void;
  onDismiss: () => void;
  isReconnecting: boolean;
}) {
  return (
    <div className="bg-gradient-to-r from-red-600 via-red-500 to-rose-600 text-white px-3 sm:px-4 py-2 flex items-center justify-center gap-2 sm:gap-4 animate-slide-down flex-wrap">
      {/* Pulsing icon */}
      <div className="flex items-center gap-2">
        <div className="relative">
          <WifiOff className="h-4 w-4 sm:h-5 sm:w-5 animate-pulse" />
          <span className="absolute -top-1 -right-1 flex h-2 w-2 sm:h-3 sm:w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 sm:h-3 sm:w-3 bg-yellow-500"></span>
          </span>
        </div>
        <span className="font-semibold text-xs sm:text-sm">Disconnected</span>
      </div>

      {/* Message - hidden on very small screens */}
      <span className="text-xs sm:text-sm text-red-100 hidden md:inline">
        Unable to connect to the analysis server.
      </span>

      {/* Actions */}
      <div className="flex items-center gap-2">
        {onReconnect && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onReconnect}
            disabled={isReconnecting}
            className="bg-white/20 hover:bg-white/30 text-white border-white/30 text-xs px-2 sm:px-3 py-1"
          >
            {isReconnecting ? (
              <>
                <RefreshCw className="h-3 w-3 sm:h-3.5 sm:w-3.5 mr-1 animate-spin" />
                <span className="hidden sm:inline">Reconnecting...</span>
              </>
            ) : (
              <>
                <RefreshCw className="h-3 w-3 sm:h-3.5 sm:w-3.5 sm:mr-1.5" />
                <span className="hidden sm:inline">Reconnect</span>
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

// Mobile View Mode Selector (shown in mobile menu)
function MobileViewModeSelector({
  viewMode,
  onViewModeChange,
  onClose,
}: {
  viewMode: UserViewMode;
  onViewModeChange?: (mode: UserViewMode) => void;
  onClose: () => void;
}) {
  if (!onViewModeChange) return null;

  return (
    <div className="p-4 border-b border-navy-700/50">
      <p className="text-xs text-gray-400 mb-2">View Mode</p>
      <div className="flex flex-col gap-2">
        {[
          { mode: "oncologist" as const, icon: Stethoscope, label: "Oncologist", gradient: "from-clinical-600 to-clinical-500" },
          { mode: "pathologist" as const, icon: Microscope, label: "Pathologist", gradient: "from-violet-600 to-violet-500" },
          { mode: "batch" as const, icon: Layers, label: "Batch", gradient: "from-amber-600 to-amber-500" },
        ].map(({ mode, icon: Icon, label, gradient }) => (
          <button
            key={mode}
            onClick={() => {
              onViewModeChange(mode);
              onClose();
            }}
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all",
              viewMode === mode
                ? `bg-gradient-to-r ${gradient} text-white shadow-md`
                : "text-gray-300 hover:bg-navy-700/50"
            )}
          >
            <Icon className="h-5 w-5" />
            <span>{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// Project Switcher Dropdown Component
function ProjectSwitcher() {
  const { projects, currentProject, switchProject } = useProject();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [isOpen]);

  if (projects.length <= 1) {
    return null;
  }

  return (
    <div ref={dropdownRef} className="relative hidden md:block">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 bg-navy-800/50 hover:bg-navy-700/50 rounded-lg border border-navy-700/30 transition-colors"
      >
        <Layers className="h-4 w-4 text-clinical-400" />
        <span className="text-sm text-gray-300 font-medium truncate max-w-[200px] lg:max-w-[280px]">
          {currentProject.name}
        </span>
        <ChevronDown className={cn("h-3.5 w-3.5 text-gray-400 transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-80 bg-navy-900 border border-navy-700/50 rounded-lg shadow-xl z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-navy-700/50">
            <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">Switch Project</span>
          </div>
          <div className="max-h-64 overflow-y-auto py-1">
            {projects.map((project) => (
              <button
                key={project.id}
                onClick={() => { switchProject(project.id); setIsOpen(false); }}
                className={cn(
                  "w-full text-left px-3 py-2.5 hover:bg-navy-800/80 transition-colors",
                  project.id === currentProject.id && "bg-navy-800/60 border-l-2 border-clinical-400"
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-200">{project.name}</span>
                  {project.id === currentProject.id && (
                    <span className="w-2 h-2 rounded-full bg-clinical-400" />
                  )}
                </div>
                <span className="text-xs text-gray-500">{project.cancer_type} · {project.prediction_target}</span>
                {project.description && (
                  <p className="text-xs text-gray-600 mt-0.5 line-clamp-2">{project.description}</p>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Overflow menu for utility actions
function UtilityOverflowMenu({
  onOpenStatus,
  onOpenShortcuts,
  onOpenDocs,
  onOpenSettings,
}: {
  onOpenStatus: () => void;
  onOpenShortcuts?: () => void;
  onOpenDocs: () => void;
  onOpenSettings: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [isOpen]);

  const items = [
    { icon: Activity, label: "System Status", onClick: onOpenStatus },
    { icon: Keyboard, label: "Keyboard Shortcuts", onClick: onOpenShortcuts },
    { icon: HelpCircle, label: "Documentation", onClick: onOpenDocs },
    { icon: Settings, label: "Settings", onClick: onOpenSettings },
  ];

  return (
    <div ref={menuRef} className="relative hidden lg:block">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 text-gray-400 hover:text-white hover:bg-navy-700/50 rounded-lg border border-navy-700/30 transition-all duration-150"
        title="More options"
      >
        <MoreHorizontal className="h-4 w-4" />
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-1 w-52 bg-white rounded-lg shadow-xl z-50 overflow-hidden py-1">
          {items.map(({ icon: Icon, label, onClick }) => (
            <button
              key={label}
              onClick={() => {
                setIsOpen(false);
                onClick?.();
              }}
              className="flex items-center gap-3 w-full px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              <Icon className="h-4 w-4 text-gray-500" />
              <span>{label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function Header({
  isConnected = false,
  isProcessing = false,
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
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Reset banner dismissed state when connection changes
  useEffect(() => {
    if (isConnected) {
      setBannerDismissed(false);
      setIsReconnecting(false);
    }
  }, [isConnected]);

  // Close mobile menu on resize to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleReconnect = async () => {
    if (onReconnect) {
      setIsReconnecting(true);
      try {
        await onReconnect();
      } finally {
        setTimeout(() => setIsReconnecting(false), 2000);
      }
    }
  };

  const handleOpenDocs = () => {
    window.open("https://github.com/Hilo-Hilo/enso-atlas/blob/main/TECHNICAL_SPECIFICATION.md", "_blank");
  };

  const showDisconnectionBanner = !isConnected && !bannerDismissed;

  // Connection status tooltip text
  const connectionStatusText = isConnected
    ? "Connected - Backend Service"
    : "Disconnected - Backend Service";

  return (
    <>
      {/* Disconnection Banner */}
      {showDisconnectionBanner && (
        <DisconnectionBanner
          onReconnect={handleReconnect}
          onDismiss={() => setBannerDismissed(true)}
          isReconnecting={isReconnecting}
        />
      )}

      <header className="h-14 sm:h-16 bg-gradient-to-r from-navy-900 via-navy-900 to-navy-800 border-b border-navy-700/50 px-3 sm:px-4 lg:px-6 flex items-center justify-between shrink-0 shadow-lg">
        {/* Left: Logo and Navigation */}
        <div className="flex items-center gap-2 sm:gap-4">
          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 -ml-2 text-gray-400 hover:text-white hover:bg-navy-700/50 rounded-lg transition-colors"
          >
            <Menu className="h-5 w-5" />
          </button>

          {/* Logo */}
          <div className="flex items-center gap-2 sm:gap-3">
            <Logo size="md" variant="full" className="hidden sm:flex" />
            <Logo size="sm" variant="mark" className="sm:hidden" />
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-gradient-to-b from-transparent via-navy-600 to-transparent mx-1 sm:mx-2 hidden md:block" />

          {/* Project Switcher */}
          <ProjectSwitcher />

          {/* Slide Manager Link */}
          <Link
            href="/slides"
            className="hidden lg:flex items-center gap-2 px-2 xl:px-3 py-1.5 bg-navy-800/50 hover:bg-navy-700/50 rounded-lg border border-navy-700/30 transition-colors"
            title="Slides"
          >
            <Layers className="h-4 w-4 text-clinical-400" />
            <span className="text-sm text-gray-300 font-medium hidden xl:inline">Slides</span>
          </Link>

          {/* Project Management Link */}
          <Link
            href="/projects"
            className="hidden lg:flex items-center gap-2 px-2 xl:px-3 py-1.5 bg-navy-800/50 hover:bg-navy-700/50 rounded-lg border border-navy-700/30 transition-colors"
            title="Projects"
          >
            <FolderOpen className="h-4 w-4 text-clinical-400" />
            <span className="text-sm text-gray-300 font-medium hidden xl:inline">Projects</span>
          </Link>

          {/* Divider */}
          <div className="h-8 w-px bg-gradient-to-b from-transparent via-navy-600 to-transparent mx-2 hidden xl:block" />

          {/* View Mode Toggle - Desktop only */}
          {onViewModeChange && (
            <div className="hidden lg:flex items-center bg-navy-800/80 rounded-xl p-1 border border-navy-700/30 shadow-inner">
              <button
                onClick={() => onViewModeChange("oncologist")}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200",
                  viewMode === "oncologist"
                    ? "bg-gradient-to-r from-clinical-600 to-clinical-500 text-white shadow-md shadow-clinical-600/30"
                    : "text-gray-400 hover:text-white hover:bg-navy-700/50"
                )}
              >
                <Stethoscope className="h-4 w-4" />
                <span className="hidden 2xl:inline">Oncologist</span>
              </button>
              <button
                onClick={() => onViewModeChange("pathologist")}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200",
                  viewMode === "pathologist"
                    ? "bg-gradient-to-r from-violet-600 to-violet-500 text-white shadow-md shadow-violet-600/30"
                    : "text-gray-400 hover:text-white hover:bg-navy-700/50"
                )}
              >
                <Microscope className="h-4 w-4" />
                <span className="hidden 2xl:inline">Pathologist</span>
              </button>
              <button
                onClick={() => onViewModeChange("batch")}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200",
                  viewMode === "batch"
                    ? "bg-gradient-to-r from-amber-600 to-amber-500 text-white shadow-md shadow-amber-600/30"
                    : "text-gray-400 hover:text-white hover:bg-navy-700/50"
                )}
              >
                <Layers className="h-4 w-4" />
                <span className="hidden 2xl:inline">Batch</span>
              </button>
            </div>
          )}
        </div>

        {/* Center: Processing indicator only */}
        <div className="hidden lg:flex items-center gap-4 xl:gap-6">
          {isProcessing && (
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
          )}
        </div>

        {/* Right: Actions and User */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Connection status dot - always visible, just a dot with tooltip */}
          <button
            onClick={() => setStatusOpen(true)}
            className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-navy-700/50 transition-all"
            title={connectionStatusText}
          >
            <div className="relative flex h-2.5 w-2.5">
              {isConnected && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-positive opacity-75" />
              )}
              <span className={cn(
                "relative inline-flex rounded-full h-2.5 w-2.5",
                isConnected ? "bg-status-positive" : "bg-status-negative"
              )} />
            </div>
          </button>

          {/* Demo Mode Toggle */}
          {onDemoModeToggle && (
            <div className="hidden lg:block">
              <DemoToggle isActive={demoMode} onToggle={onDemoModeToggle} />
            </div>
          )}

          {/* Utility overflow menu (replaces 4 individual icon buttons) */}
          <UtilityOverflowMenu
            onOpenStatus={() => setStatusOpen(true)}
            onOpenShortcuts={onOpenShortcuts}
            onOpenDocs={handleOpenDocs}
            onOpenSettings={() => setSettingsOpen(true)}
          />

          {/* Divider - Desktop only */}
          <div className="h-8 w-px bg-gradient-to-b from-transparent via-navy-600 to-transparent hidden lg:block" />

          {/* User Context */}
          <div className="relative">
            <button
              onClick={() => setUserDropdownOpen(!userDropdownOpen)}
              className="flex items-center gap-2 px-1.5 sm:px-2 py-1.5 rounded-xl hover:bg-navy-800/80 transition-all duration-200 border border-transparent hover:border-navy-700/50 group"
            >
              <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-br from-clinical-600 to-clinical-700 flex items-center justify-center shadow-md group-hover:shadow-clinical-600/30 transition-shadow">
                <User className="h-4 w-4 text-white" />
              </div>
              {userName && (
                <>
                  <span className="text-sm text-gray-200 font-medium hidden xl:inline">{userName}</span>
                  <ChevronDown className={cn(
                    "h-3.5 w-3.5 text-gray-400 group-hover:text-gray-300 transition-all hidden xl:inline",
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

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Mobile Slide-out Menu */}
      <div
        className={cn(
          "fixed top-0 left-0 h-full w-72 bg-navy-900 z-50 transform transition-transform duration-300 ease-out lg:hidden",
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Menu Header */}
        <div className="flex items-center justify-between p-4 border-b border-navy-700/50">
          <Logo size="md" variant="full" />
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="p-2 text-gray-400 hover:text-white hover:bg-navy-700/50 rounded-lg"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* View Mode Selector */}
        <MobileViewModeSelector
          viewMode={viewMode}
          onViewModeChange={onViewModeChange}
          onClose={() => setMobileMenuOpen(false)}
        />

        {/* Navigation Links */}
        <div className="p-4 border-b border-navy-700/50 space-y-1">
          <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">Navigation</p>
          <Link
            href="/slides"
            onClick={() => setMobileMenuOpen(false)}
            className="flex items-center gap-3 w-full px-4 py-3 text-gray-300 hover:bg-navy-700/50 rounded-lg transition-colors"
          >
            <Layers className="h-5 w-5 text-clinical-400" />
            <span>Slides</span>
          </Link>
          <Link
            href="/projects"
            onClick={() => setMobileMenuOpen(false)}
            className="flex items-center gap-3 w-full px-4 py-3 text-gray-300 hover:bg-navy-700/50 rounded-lg transition-colors"
          >
            <FolderOpen className="h-5 w-5 text-clinical-400" />
            <span>Projects</span>
          </Link>
        </div>

        {/* Menu Actions */}
        <div className="p-4 space-y-2">
          <button
            onClick={() => {
              setStatusOpen(true);
              setMobileMenuOpen(false);
            }}
            className="flex items-center gap-3 w-full px-4 py-3 text-gray-300 hover:bg-navy-700/50 rounded-lg transition-colors"
          >
            <Activity className="h-5 w-5" />
            <span>System Status</span>
            <div
              className={cn(
                "ml-auto w-2.5 h-2.5 rounded-full",
                isConnected ? "bg-status-positive" : "bg-status-negative"
              )}
            />
          </button>
          <button
            onClick={handleOpenDocs}
            className="flex items-center gap-3 w-full px-4 py-3 text-gray-300 hover:bg-navy-700/50 rounded-lg transition-colors"
          >
            <HelpCircle className="h-5 w-5" />
            <span>Documentation</span>
          </button>
          <button
            onClick={() => {
              setSettingsOpen(true);
              setMobileMenuOpen(false);
            }}
            className="flex items-center gap-3 w-full px-4 py-3 text-gray-300 hover:bg-navy-700/50 rounded-lg transition-colors"
          >
            <Settings className="h-5 w-5" />
            <span>Settings</span>
          </button>
          {onDemoModeToggle && (
            <div className="flex items-center justify-between px-4 py-3">
              <span className="text-gray-300">Demo Mode</span>
              <DemoToggle isActive={demoMode} onToggle={onDemoModeToggle} />
            </div>
          )}
        </div>
      </div>

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
