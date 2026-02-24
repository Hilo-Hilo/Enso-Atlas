"use client";

import React, { useEffect, useRef, useCallback } from "react";
import {
  User,
  Settings,
  LogOut,
  ChevronRight,
  Shield,
  Bell,
  HelpCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface UserDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  userName?: string;
  userRole?: string;
  onOpenSettings?: () => void;
}

export function UserDropdown({
  isOpen,
  onClose,
  userName = "Clinician",
  userRole = "Researcher",
  onOpenSettings,
}: UserDropdownProps) {
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose]);

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

  if (!isOpen) return null;

  const menuItems = [
    {
      icon: User,
      label: "Profile",
      description: "View your profile",
      onClick: () => {
        // Placeholder - could open profile modal
        onClose();
      },
    },
    {
      icon: Settings,
      label: "Settings",
      description: "Preferences",
      onClick: () => {
        onOpenSettings?.();
        onClose();
      },
    },
    {
      icon: Bell,
      label: "Notifications",
      description: "Manage alerts",
      onClick: () => {
        onClose();
      },
    },
    {
      icon: HelpCircle,
      label: "Help & Support",
      description: "Documentation",
      onClick: () => {
        window.open("https://github.com/Hilo-Hilo/Enso-Atlas/blob/main/docs.md", "_blank");
        onClose();
      },
    },
  ];

  return (
    <div
      ref={dropdownRef}
      className="absolute right-0 top-full mt-2 w-72 bg-white rounded-xl shadow-xl border border-gray-200 overflow-hidden z-50 animate-scale-in origin-top-right"
    >
      {/* User Info Header */}
      <div className="p-4 bg-gradient-to-r from-clinical-50 to-clinical-100 border-b border-clinical-200">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-clinical-600 to-clinical-700 flex items-center justify-center shadow-md">
            <User className="h-6 w-6 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-gray-900 truncate">{userName}</p>
            <div className="flex items-center gap-1.5">
              <Shield className="h-3 w-3 text-clinical-600" />
              <span className="text-xs text-clinical-700">{userRole}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Menu Items */}
      <div className="p-2">
        {menuItems.map((item, index) => (
          <button
            key={index}
            onClick={item.onClick}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-gray-50 transition-colors group"
          >
            <div className="w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center group-hover:bg-clinical-100 transition-colors">
              <item.icon className="h-4 w-4 text-gray-600 group-hover:text-clinical-600 transition-colors" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900">{item.label}</p>
              <p className="text-xs text-gray-500">{item.description}</p>
            </div>
            <ChevronRight className="h-4 w-4 text-gray-300 group-hover:text-gray-400 transition-colors" />
          </button>
        ))}
      </div>

      {/* Logout */}
      <div className="p-2 border-t border-gray-100">
        <button
          onClick={() => {
            // Placeholder - would handle logout
            onClose();
          }}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-red-50 transition-colors group"
        >
          <div className="w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center group-hover:bg-red-100 transition-colors">
            <LogOut className="h-4 w-4 text-gray-600 group-hover:text-red-600 transition-colors" />
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-900 group-hover:text-red-600 transition-colors">Sign Out</p>
            <p className="text-xs text-gray-500">End your session</p>
          </div>
        </button>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
        <p className="text-xs text-gray-400 text-center">
          Atlas Pathology v0.1.0 - Research Use Only
        </p>
      </div>
    </div>
  );
}
