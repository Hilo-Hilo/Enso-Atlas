"use client";

import React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Microscope, Settings, HelpCircle, Activity } from "lucide-react";

interface HeaderProps {
  isConnected?: boolean;
  version?: string;
}

export function Header({ isConnected = false, version = "0.1.0" }: HeaderProps) {
  return (
    <header className="h-14 bg-white border-b border-surface-border px-4 flex items-center justify-between shrink-0">
      {/* Logo and Title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-9 h-9 bg-clinical-600 rounded-lg">
          <Microscope className="h-5 w-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-gray-900 leading-tight">
            Enso Atlas
          </h1>
          <p className="text-xs text-gray-500 leading-tight">
            Pathology Evidence Engine
          </p>
        </div>
        <Badge variant="info" size="sm" className="ml-2">
          v{version}
        </Badge>
      </div>

      {/* Status and Actions */}
      <div className="flex items-center gap-3">
        {/* Connection Status */}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? "bg-status-positive" : "bg-status-negative"
            }`}
          />
          <span className="text-xs text-gray-600">
            {isConnected ? "Backend Connected" : "Backend Offline"}
          </span>
        </div>

        {/* Research Mode Indicator */}
        <Badge variant="warning" size="sm">
          Research Use Only
        </Badge>

        {/* Action Buttons */}
        <div className="flex items-center gap-1 ml-2">
          <Button
            variant="ghost"
            size="sm"
            className="p-2"
            title="System Status"
          >
            <Activity className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" className="p-2" title="Help">
            <HelpCircle className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" className="p-2" title="Settings">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
}
