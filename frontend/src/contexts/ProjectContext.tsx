"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import type { Project } from "@/types";

// Default fallback project when API is not available
const DEFAULT_PROJECT: Project = {
  id: "default",
  name: "Ovarian Cancer - Platinum Sensitivity",
  cancer_type: "Ovarian Cancer",
  prediction_target: "platinum_sensitivity",
  classes: ["resistant", "sensitive"],
  positive_class: "sensitive",
  description: "Predicts platinum-based chemotherapy sensitivity in high-grade serous ovarian carcinoma",
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
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003";
        const res = await fetch(`${apiUrl}/api/projects`, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const projectList: Project[] = Array.isArray(data) ? data : data.projects ?? [];
        if (projectList.length > 0) {
          setProjects(projectList);

          // Restore previously selected project from localStorage
          const savedId = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
          const saved = savedId ? projectList.find((p) => p.id === savedId) : null;
          setCurrentProject(saved ?? projectList[0]);
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
      if (project) {
        setCurrentProject(project);
        if (typeof window !== "undefined") {
          localStorage.setItem(STORAGE_KEY, id);
        }
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
