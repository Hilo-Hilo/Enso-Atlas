"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { BatchAnalysisPanel } from "@/components/panels";
import { Button } from "@/components/ui/Button";
import { healthCheck } from "@/lib/api";
import { useProject } from "@/contexts/ProjectContext";
import { ArrowLeft } from "lucide-react";

export default function BatchPage() {
  const router = useRouter();
  const { currentProject, switchProject } = useProject();
  const [isConnected, setIsConnected] = useState(false);
  const [initialSlideIds, setInitialSlideIds] = useState<string[]>([]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const params = new URLSearchParams(window.location.search);
    const requestedProject = params.get("project");
    if (requestedProject && requestedProject !== currentProject.id) {
      switchProject(requestedProject);
    }

    const requestedSlides = (params.get("slides") || "")
      .split(",")
      .map((id) => id.trim())
      .filter(Boolean);
    setInitialSlideIds(requestedSlides);
  }, [currentProject.id, switchProject]);

  useEffect(() => {
    let cancelled = false;

    const checkConnection = async () => {
      try {
        await healthCheck();
        if (!cancelled) setIsConnected(true);
      } catch {
        if (!cancelled) setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-surface-secondary">
      <Header isConnected={isConnected} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-6">
        <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
          <div>
            <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Batch Analysis</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Run asynchronous multi-slide analysis for the current project scope.
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              router.push(
                currentProject.id && currentProject.id !== "default"
                  ? `/slides?project=${currentProject.id}`
                  : "/slides"
              )
            }
            className="gap-1.5"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Slide Manager
          </Button>
        </div>

        <BatchAnalysisPanel
          className="h-[calc(100vh-220px)]"
          initialSelectedIds={initialSlideIds}
          onSlideSelect={(slideId) => {
            const params = new URLSearchParams();
            if (currentProject.id && currentProject.id !== "default") {
              params.set("project", currentProject.id);
            }
            params.set("slide", slideId);
            router.push(`/?${params.toString()}`);
          }}
        />
      </main>

      <div className="hidden sm:block">
        <Footer />
      </div>
    </div>
  );
}
