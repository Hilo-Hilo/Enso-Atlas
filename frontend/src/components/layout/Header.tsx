"use client";

import React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
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
}: HeaderProps) {
  return (
    <header className="h-16 bg-navy-900 border-b border-navy-700 px-6 flex items-center justify-between shrink-0">
      {/* Left: Logo and Branding */}
      <div className="flex items-center gap-4">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 bg-clinical-600 rounded-lg shadow-lg">
            <Microscope className="h-5 w-5 text-white" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold text-white tracking-tight">
                Enso Atlas
              </h1>
              <span className="text-xs text-navy-100 font-mono bg-navy-800 px-1.5 py-0.5 rounded">
                v{version}
              </span>
            </div>
            <p className="text-xs text-gray-400 -mt-0.5">
              Pathology Evidence Engine
            </p>
          </div>
        </div>

        {/* Divider */}
        <div className="h-8 w-px bg-navy-700 mx-2" />

        {/* Institution Context */}
        <div className="flex items-center gap-2">
          <Building2 className="h-4 w-4 text-gray-400" />
          <span className="text-sm text-gray-300">{institutionName}</span>
        </div>

        {/* Divider */}
        <div className="h-8 w-px bg-navy-700 mx-2" />

        {/* View Mode Toggle */}
        {onViewModeChange && (
          <div className="flex items-center bg-navy-800 rounded-lg p-1">
            <button
              onClick={() => onViewModeChange("oncologist")}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all",
                viewMode === "oncologist"
                  ? "bg-clinical-600 text-white shadow-sm"
                  : "text-gray-400 hover:text-gray-200"
              )}
            >
              <Stethoscope className="h-4 w-4" />
              <span className="hidden lg:inline">Oncologist</span>
            </button>
            <button
              onClick={() => onViewModeChange("pathologist")}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all",
                viewMode === "pathologist"
                  ? "bg-violet-600 text-white shadow-sm"
                  : "text-gray-400 hover:text-gray-200"
              )}
            >
              <Microscope className="h-4 w-4" />
              <span className="hidden lg:inline">Pathologist</span>
            </button>
            <button
              onClick={() => onViewModeChange("batch")}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all",
                viewMode === "batch"
                  ? "bg-amber-600 text-white shadow-sm"
                  : "text-gray-400 hover:text-gray-200"
              )}
            >
              <Layers className="h-4 w-4" />
              <span className="hidden lg:inline">Batch</span>
            </button>
          </div>
        )}
      </div>

      {/* Center: Status Indicators */}
      <div className="flex items-center gap-6">
        {/* Backend Connection Status */}
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "status-indicator",
              isConnected ? "status-indicator-online" : "status-indicator-offline"
            )}
          />
          <div className="flex flex-col">
            <span className="text-xs text-gray-300">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
            <span className="text-2xs text-gray-500">Backend Service</span>
          </div>
        </div>

        {/* Processing Status */}
        {isProcessing && (
          <>
            <div className="h-6 w-px bg-navy-700" />
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-clinical-400 animate-pulse" />
                <span className="text-xs text-clinical-400">Processing</span>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Right: Actions and User */}
      <div className="flex items-center gap-4">
        {/* Research Mode Warning */}
        <Badge variant="warning" size="sm" className="font-medium">
          Research Use Only
        </Badge>

        {/* System Status */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="p-2 text-gray-400 hover:text-white hover:bg-navy-800"
            title="System Status"
          >
            <Activity className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="p-2 text-gray-400 hover:text-white hover:bg-navy-800"
            title="Keyboard Shortcuts (?)"
            onClick={onOpenShortcuts}
          >
            <Keyboard className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="p-2 text-gray-400 hover:text-white hover:bg-navy-800"
            title="Documentation"
          >
            <HelpCircle className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="p-2 text-gray-400 hover:text-white hover:bg-navy-800"
            title="Settings"
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>

        {/* Divider */}
        <div className="h-8 w-px bg-navy-700" />

        {/* User Context */}
        <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-navy-800 transition-colors">
          <div className="w-8 h-8 rounded-full bg-clinical-700 flex items-center justify-center">
            <User className="h-4 w-4 text-white" />
          </div>
          {userName && (
            <>
              <span className="text-sm text-gray-200">{userName}</span>
              <ChevronDown className="h-3.5 w-3.5 text-gray-400" />
            </>
          )}
        </button>
      </div>
    </header>
  );
}
