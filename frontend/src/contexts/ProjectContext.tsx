"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import type { Project } from "@/types";

// Default fallback project when API is not available
const DEFAULT_PROJECT: Project = {
  id: "default",
  name: "Cancer Analysis",
  cancer_type: "Cancer",
  prediction_target: "treatment_response",
  classes: ["unfavorable", "favorable"],
  positive_class: "favorable",
  description: "AI-powered pathology analysis for treatment response prediction",
  dataset_source: "TCGA",
  disclaimer: "",
  models: {
    embedder: "PathFoundation (DINOv2-L)",
    mil_architecture: "TransMIL",
    report_generator: "MedGemma",
    semantic_search: "MedSigLIP",
  },
};

interface ProjectContextValue {
  projects: Project[];
  currentProject: Project;
  switchProject: (id: string) => void;
  isLoading: boolean;
  error: string | null;
}

const ProjectContext = createContext<ProjectContextValue | undefined>(undefined);

const STORAGE_KEY = "enso-atlas-selected-project";

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProject, setCurrentProject] = useState<Project>(DEFAULT_PROJECT);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch projects from backend
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
        const res = await fetch(`${apiUrl}/api/projects`, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const projectList: Project[] = Array.isArray(data) ? data : data.projects ?? [];
        if (projectList.length > 0) {
          setProjects(projectList);

          // URL ?project= param takes priority, then localStorage, then first
          const urlParams =
            typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
          const urlProjectId = urlParams?.get("project");
          const savedId =
            typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
          const fromUrl = urlProjectId
            ? projectList.find((p) => p.id === urlProjectId)
            : null;
          const fromStorage = savedId
            ? projectList.find((p) => p.id === savedId)
            : null;
          const resolved = fromUrl ?? fromStorage ?? projectList[0];
          setCurrentProject(resolved);

          if (typeof window !== "undefined") {
            localStorage.setItem(STORAGE_KEY, resolved.id);
          }
        } else {
          // No projects returned — use default
          setProjects([DEFAULT_PROJECT]);
        }
      } catch (err) {
        console.warn("Projects API not available, using defaults:", err);
        setError(err instanceof Error ? err.message : "Failed to load projects");
        setProjects([DEFAULT_PROJECT]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchProjects();
  }, []);

  const switchProject = useCallback(
    (id: string) => {
      const project = projects.find((p) => p.id === id);
      if (!project) return;

      setCurrentProject(project);

      if (typeof window !== "undefined") {
        localStorage.setItem(STORAGE_KEY, id);

        // Keep URL project query in sync so deep-links and refreshes preserve context.
        const url = new URL(window.location.href);
        if (id === "default") {
          url.searchParams.delete("project");
        } else {
          url.searchParams.set("project", id);
        }
        window.history.replaceState(window.history.state, "", url.toString());
      }
    },
    [projects]
  );

  return (
    <ProjectContext.Provider value={{ projects, currentProject, switchProject, isLoading, error }}>
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject(): ProjectContextValue {
  const ctx = useContext(ProjectContext);
  if (!ctx) {
    // If used outside provider, return defaults gracefully
    return {
      projects: [DEFAULT_PROJECT],
      currentProject: DEFAULT_PROJECT,
      switchProject: () => {},
      isLoading: false,
      error: null,
    };
  }
  return ctx;
}
