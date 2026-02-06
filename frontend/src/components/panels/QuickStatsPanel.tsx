"use client";

import React, { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { cn } from "@/lib/utils";
import {
  BarChart3,
  Activity,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  XCircle,
  Clock,
  Microscope,
  Target,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { useProject } from "@/contexts/ProjectContext";

export interface AnalysisStats {
  totalAnalyzed: number;
  todayAnalyzed: number;
  avgConfidence: number;
  responderCount: number;
  nonResponderCount: number;
  avgProcessingTime: number;
  lastAnalyzedAt?: string;
  recentAnalyses: Array<{
    slideId: string;
    prediction: string;
    confidence: number;
    timestamp: string;
  }>;
}

interface QuickStatsPanelProps {
  onRefresh?: () => void;
}

// LocalStorage key for stats
const STATS_STORAGE_KEY = "atlas-quick-stats";

// Helper to get today's date string
function getTodayKey(): string {
  return new Date().toISOString().split("T")[0];
}

// Helper to load stats from localStorage
function loadStats(): AnalysisStats {
  if (typeof window === "undefined") {
    return getDefaultStats();
  }
  try {
    const stored = localStorage.getItem(STATS_STORAGE_KEY);
    if (!stored) return getDefaultStats();
    const stats = JSON.parse(stored);
    // Reset todayAnalyzed if it's a new day
    if (stats.lastDate !== getTodayKey()) {
      stats.todayAnalyzed = 0;
      stats.lastDate = getTodayKey();
    }
    return stats;
  } catch (err) {
    console.error("Failed to load stats from localStorage:", err);
    return getDefaultStats();
  }
}

// Helper to save stats to localStorage
function saveStats(stats: AnalysisStats): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STATS_STORAGE_KEY, JSON.stringify({
      ...stats,
      lastDate: getTodayKey(),
    }));
  } catch (err) {
    console.error("Failed to save stats to localStorage:", err);
  }
}

// Default stats
function getDefaultStats(): AnalysisStats {
  return {
    totalAnalyzed: 0,
    todayAnalyzed: 0,
    avgConfidence: 0,
    responderCount: 0,
    nonResponderCount: 0,
    avgProcessingTime: 0,
    recentAnalyses: [],
  };
}

// Persistent stats store
let statsStore: AnalysisStats = getDefaultStats();

// Initialize from localStorage on load
if (typeof window !== "undefined") {
  statsStore = loadStats();
}

// Function to record a new analysis (called from main page)
export function recordAnalysis(
  slideId: string,
  prediction: string,
  confidence: number,
  processingTime: number
) {
  // Reload from localStorage to get latest
  statsStore = loadStats();

  statsStore.totalAnalyzed += 1;
  statsStore.todayAnalyzed += 1;

  if (prediction.toLowerCase().includes("responder") && !prediction.toLowerCase().includes("non")) {
    statsStore.responderCount += 1;
  } else {
    statsStore.nonResponderCount += 1;
  }

  // Update average confidence
  const totalConfidence =
    statsStore.avgConfidence * (statsStore.totalAnalyzed - 1) + confidence;
  statsStore.avgConfidence = totalConfidence / statsStore.totalAnalyzed;

  // Update average processing time
  const totalTime =
    statsStore.avgProcessingTime * (statsStore.totalAnalyzed - 1) +
    processingTime;
  statsStore.avgProcessingTime = totalTime / statsStore.totalAnalyzed;

  statsStore.lastAnalyzedAt = new Date().toISOString();

  // Add to recent analyses (keep last 10)
  statsStore.recentAnalyses.unshift({
    slideId,
    prediction,
    confidence,
    timestamp: new Date().toISOString(),
  });
  if (statsStore.recentAnalyses.length > 10) {
    statsStore.recentAnalyses.pop();
  }

  // Persist to localStorage
  saveStats(statsStore);
}

// Function to get current stats
export function getStats(): AnalysisStats {
  statsStore = loadStats();
  return { ...statsStore };
}

// Function to reset stats (for testing or user request)
export function resetStats(): void {
  statsStore = getDefaultStats();
  saveStats(statsStore);
}

export function QuickStatsPanel({ onRefresh }: QuickStatsPanelProps) {
  // Initialize with default stats to avoid hydration mismatch
  // (localStorage is not available on server)
  const [stats, setStats] = useState<AnalysisStats>(getDefaultStats);
  const [isLoading, setIsLoading] = useState(false);
  const [isHydrated, setIsHydrated] = useState(false);

  // Load stats from localStorage after hydration
  useEffect(() => {
    setStats(getStats());
    setIsHydrated(true);
  }, []);

  // Refresh stats periodically (only after hydration)
  useEffect(() => {
    if (!isHydrated) return;
    
    const interval = setInterval(() => {
      setStats(getStats());
    }, 2000);

    return () => clearInterval(interval);
  }, [isHydrated]);

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => {
      setStats(getStats());
      setIsLoading(false);
      onRefresh?.();
    }, 300);
  };

  const formatTime = (dateStr?: string) => {
    if (!dateStr) return "Never";
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  // Project-aware labels
  const { currentProject } = useProject();
  const positiveLabel = currentProject.positive_class || currentProject.classes?.[1] || "Responder";
  const negativeLabel = currentProject.classes?.find(c => c !== currentProject.positive_class) || currentProject.classes?.[0] || "Non-Responder";

  const responderRate =
    stats.totalAnalyzed > 0
      ? (stats.responderCount / stats.totalAnalyzed) * 100
      : 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-clinical-600" />
            Quick Stats
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            disabled={isLoading}
            className="p-1.5"
            title="Refresh stats"
          >
            <RefreshCw
              className={cn("h-4 w-4", isLoading && "animate-spin")}
            />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="pt-2 space-y-4">
        {/* Main Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          {/* Total Analyzed */}
          <StatCard
            icon={<Microscope className="h-4 w-4" />}
            label="Total Analyzed"
            value={stats.totalAnalyzed}
            sublabel={`${stats.todayAnalyzed} today`}
            color="blue"
          />

          {/* Average Confidence */}
          <StatCard
            icon={<Target className="h-4 w-4" />}
            label="Avg Confidence"
            value={`${Math.round(stats.avgConfidence * 100)}%`}
            sublabel={stats.avgConfidence >= 0.7 ? "High" : stats.avgConfidence >= 0.4 ? "Moderate" : "Low"}
            color={stats.avgConfidence >= 0.7 ? "green" : stats.avgConfidence >= 0.4 ? "amber" : "red"}
          />

          {/* Positive Class */}
          <StatCard
            icon={<CheckCircle className="h-4 w-4" />}
            label={`${positiveLabel}s`}
            value={stats.responderCount}
            sublabel={`${responderRate.toFixed(0)}% rate`}
            color="green"
          />

          {/* Negative Class */}
          <StatCard
            icon={<XCircle className="h-4 w-4" />}
            label={`${negativeLabel}s`}
            value={stats.nonResponderCount}
            sublabel={`${(100 - responderRate).toFixed(0)}% rate`}
            color="red"
          />
        </div>

        {/* Response Distribution Bar */}
        {stats.totalAnalyzed > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-600 font-medium">
                Response Distribution
              </span>
              <span className="text-gray-500">
                {stats.responderCount}:{stats.nonResponderCount}
              </span>
            </div>
            <div className="h-3 bg-gray-100 rounded-full overflow-hidden flex">
              <div
                className="h-full bg-green-500 transition-all duration-500"
                style={{ width: `${responderRate}%` }}
              />
              <div
                className="h-full bg-red-400 transition-all duration-500"
                style={{ width: `${100 - responderRate}%` }}
              />
            </div>
            <div className="flex justify-between text-2xs text-gray-500">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                {positiveLabel}
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-red-400" />
                {negativeLabel}
              </span>
            </div>
          </div>
        )}

        {/* Processing Time */}
        {stats.totalAnalyzed > 0 && (
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-600">Avg Processing</span>
            </div>
            <span className="text-sm font-mono font-medium text-gray-900">
              {stats.avgProcessingTime > 1000
                ? `${(stats.avgProcessingTime / 1000).toFixed(1)}s`
                : `${Math.round(stats.avgProcessingTime)}ms`}
            </span>
          </div>
        )}

        {/* Last Activity */}
        <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-100">
          <span>Last analysis</span>
          <span className="font-medium">
            {formatTime(stats.lastAnalyzedAt)}
          </span>
        </div>

        {/* Recent Analyses */}
        {stats.recentAnalyses.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
              Recent Analyses
            </h4>
            <div className="space-y-1.5 max-h-32 overflow-y-auto">
              {stats.recentAnalyses.slice(0, 5).map((analysis, idx) => (
                <div
                  key={`${analysis.slideId}-${idx}`}
                  className="flex items-center justify-between p-2 bg-white border border-gray-100 rounded-lg text-xs"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className={cn(
                        "w-2 h-2 rounded-full shrink-0",
                        analysis.prediction.toLowerCase().includes("non")
                          ? "bg-red-400"
                          : "bg-green-500"
                      )}
                    />
                    <span className="font-mono truncate text-gray-700">
                      {analysis.slideId.slice(0, 12)}...
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Badge
                      variant={
                        analysis.confidence >= 0.7
                          ? "success"
                          : analysis.confidence >= 0.4
                          ? "warning"
                          : "default"
                      }
                      size="sm"
                    >
                      {Math.round(analysis.confidence * 100)}%
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {stats.totalAnalyzed === 0 && (
          <div className="text-center py-6 text-gray-500">
            <Activity className="h-8 w-8 mx-auto mb-2 text-gray-300" />
            <p className="text-sm font-medium text-gray-600">No analyses yet</p>
            <p className="text-xs mt-1">
              Run an analysis to see statistics
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Individual Stat Card Component
interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sublabel?: string;
  color: "blue" | "green" | "amber" | "red" | "gray";
}

function StatCard({ icon, label, value, sublabel, color }: StatCardProps) {
  const colorClasses = {
    blue: "bg-blue-50 text-blue-600 border-blue-100",
    green: "bg-green-50 text-green-600 border-green-100",
    amber: "bg-amber-50 text-amber-600 border-amber-100",
    red: "bg-red-50 text-red-600 border-red-100",
    gray: "bg-gray-50 text-gray-600 border-gray-100",
  };

  const iconColorClasses = {
    blue: "text-blue-500",
    green: "text-green-500",
    amber: "text-amber-500",
    red: "text-red-500",
    gray: "text-gray-500",
  };

  return (
    <div
      className={cn(
        "p-3 rounded-lg border transition-all hover:shadow-sm",
        colorClasses[color]
      )}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className={iconColorClasses[color]}>{icon}</span>
        <span className="text-xs font-medium text-gray-600">{label}</span>
      </div>
      <div className="text-xl font-bold">{value}</div>
      {sublabel && (
        <div className="text-2xs text-gray-500 mt-0.5">{sublabel}</div>
      )}
    </div>
  );
}
