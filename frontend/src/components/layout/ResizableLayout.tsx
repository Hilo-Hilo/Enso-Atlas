"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import {
  PanelGroup,
  Panel,
  PanelResizeHandle,
  ImperativePanelHandle,
} from "react-resizable-panels";
import { ChevronLeft, ChevronRight, GripVertical, Layers, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

interface ResizableLayoutProps {
  leftPanel: React.ReactNode;
  centerPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  className?: string;
  defaultLeftSize?: number;
  defaultRightSize?: number;
  minLeftSize?: number;
  minRightSize?: number;
}

export function ResizableLayout({
  leftPanel,
  centerPanel,
  rightPanel,
  className,
  defaultLeftSize = 20,
  defaultRightSize = 25,
  minLeftSize = 15,
  minRightSize = 20,
}: ResizableLayoutProps) {
  const leftPanelRef = useRef<ImperativePanelHandle>(null);
  const rightPanelRef = useRef<ImperativePanelHandle>(null);

  const [isLeftCollapsed, setIsLeftCollapsed] = useState(false);
  const [isRightCollapsed, setIsRightCollapsed] = useState(false);

  const handleLeftCollapse = useCallback(() => setIsLeftCollapsed(true), []);
  const handleLeftExpand = useCallback(() => setIsLeftCollapsed(false), []);
  const handleRightCollapse = useCallback(() => setIsRightCollapsed(true), []);
  const handleRightExpand = useCallback(() => setIsRightCollapsed(false), []);

  const toggleLeftPanel = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    const panel = leftPanelRef.current;
    if (panel) {
      if (isLeftCollapsed) {
        panel.expand();
      } else {
        panel.collapse();
      }
    }
  };

  const toggleRightPanel = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    const panel = rightPanelRef.current;
    if (panel) {
      if (isRightCollapsed) {
        panel.expand();
      } else {
        panel.collapse();
      }
    }
  };

  return (
    <PanelGroup
      direction="horizontal"
      autoSaveId="med-gemma-layout-v1"
      className={cn("h-full w-full bg-surface-secondary", className)}
    >
      {/* Left Panel */}
      <Panel
        ref={leftPanelRef}
        collapsible={true}
        collapsedSize={3}
        minSize={minLeftSize}
        maxSize={30}
        defaultSize={defaultLeftSize}
        order={1}
        onCollapse={handleLeftCollapse}
        onExpand={handleLeftExpand}
        className={cn(
          "bg-white border-r border-surface-border transition-all duration-300 ease-in-out relative flex flex-col",
          isLeftCollapsed ? "items-center bg-gray-50 overflow-hidden" : "overflow-hidden"
        )}
      >
        <div className={cn("h-full w-full flex flex-col transition-opacity duration-300", isLeftCollapsed ? "opacity-0 invisible absolute w-0" : "opacity-100 visible")}>
            {leftPanel}
        </div>
        
        {/* Vertical Strip Content (Icons) when collapsed */}
        <div className={cn(
            "absolute inset-0 flex flex-col items-center pt-4 w-full h-full transition-opacity duration-300",
            isLeftCollapsed ? "opacity-100 visible" : "opacity-0 invisible pointer-events-none"
        )}>
            <button 
                onClick={toggleLeftPanel}
                className="p-2 hover:bg-white hover:shadow-sm rounded-lg transition-all text-clinical-600 mb-4"
                title="Expand Case Selection"
            >
                <Layers className="h-5 w-5" />
            </button>
            <div className="w-8 h-px bg-gray-200 mb-4" />
            <div 
                className="writing-vertical-lr text-xs font-semibold text-gray-500 tracking-wider uppercase select-none cursor-pointer hover:text-clinical-600 transition-colors"
                onClick={toggleLeftPanel}
                style={{ writingMode: 'vertical-rl', textOrientation: 'mixed', transform: 'rotate(180deg)' }}
            >
                Case Selection
            </div>
        </div>
      </Panel>

      {/* Resize Handle Left */}
      <PanelResizeHandle className="w-1 bg-transparent hover:bg-clinical-500/20 transition-colors group/handle relative z-10 focus:outline-none">
        <div className="absolute inset-y-0 -left-1 -right-1 cursor-col-resize" />
        {/* The collapse button when expanded */}
        {!isLeftCollapsed && (
             <button
                onClick={toggleLeftPanel}
                className="absolute top-1/2 -translate-y-1/2 -left-3 z-50 w-6 h-12 flex items-center justify-center bg-white border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors focus:outline-none rounded-r-lg border-l-0 text-gray-400 hover:text-gray-600"
                title="Collapse sidebar"
            >
                <ChevronLeft className="h-4 w-4" />
            </button>
        )}
      </PanelResizeHandle>

      {/* Center Panel (Viewer) */}
      <Panel order={2} minSize={30} className="relative bg-white z-0">
        {centerPanel}
      </Panel>

      {/* Resize Handle Right */}
      <PanelResizeHandle className="w-1 bg-transparent hover:bg-clinical-500/20 transition-colors group/handle relative z-10 focus:outline-none">
        <div className="absolute inset-y-0 -left-1 -right-1 cursor-col-resize" />
        {/* The collapse button when expanded */}
        {!isRightCollapsed && (
             <button
                onClick={toggleRightPanel}
                className="absolute top-1/2 -translate-y-1/2 -right-3 z-50 w-6 h-12 flex items-center justify-center bg-white border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors focus:outline-none rounded-l-lg border-r-0 text-gray-400 hover:text-gray-600"
                title="Collapse sidebar"
            >
                <ChevronRight className="h-4 w-4" />
            </button>
        )}
      </PanelResizeHandle>

      {/* Right Panel */}
      <Panel
        ref={rightPanelRef}
        collapsible={true}
        collapsedSize={3}
        minSize={minRightSize}
        maxSize={40}
        defaultSize={defaultRightSize}
        order={3}
        onCollapse={handleRightCollapse}
        onExpand={handleRightExpand}
        className={cn(
          "bg-white border-l border-surface-border transition-all duration-300 ease-in-out relative flex flex-col",
          isRightCollapsed ? "items-center bg-gray-50 overflow-hidden" : "overflow-hidden"
        )}
      >
        <div className={cn("h-full w-full flex flex-col transition-opacity duration-300", isRightCollapsed ? "opacity-0 invisible absolute w-0" : "opacity-100 visible")}>
            {rightPanel}
        </div>

        {/* Vertical Strip Content (Icons) when collapsed */}
        <div className={cn(
            "absolute inset-0 flex flex-col items-center pt-4 w-full h-full transition-opacity duration-300",
            isRightCollapsed ? "opacity-100 visible" : "opacity-0 invisible pointer-events-none"
        )}>
            <button 
                onClick={toggleRightPanel}
                className="p-2 hover:bg-white hover:shadow-sm rounded-lg transition-all text-clinical-600 mb-4"
                title="Expand Analysis Results"
            >
                <BarChart3 className="h-5 w-5" />
            </button>
            <div className="w-8 h-px bg-gray-200 mb-4" />
            <div 
                className="writing-vertical-lr text-xs font-semibold text-gray-500 tracking-wider uppercase select-none cursor-pointer hover:text-clinical-600 transition-colors"
                onClick={toggleRightPanel}
                style={{ writingMode: 'vertical-rl', textOrientation: 'mixed', transform: 'rotate(180deg)' }}
            >
                Analysis Results
            </div>
        </div>
      </Panel>
    </PanelGroup>
  );
}
