"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { useToast } from "@/components/ui";
import { useProject } from "@/contexts/ProjectContext";
import { cn } from "@/lib/utils";
import {
  Plus,
  Pencil,
  Trash2,
  FolderOpen,
  Upload,
  Layers,
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ChevronRight,
  ArrowLeft,
  Microscope,
  Database,
  Cpu,
  FileText,
  RefreshCw,
  Settings,
  X,
} from "lucide-react";

// -------------------------------------------------------------------
// Types
// -------------------------------------------------------------------

interface ProjectStatus {
  project_id: string;
  name: string;
  ready: {
    slides_dir: boolean;
    embeddings_dir: boolean;
    mil_checkpoint: boolean;
    labels_file: boolean;
  };
  counts: {
    slides: number;
    embeddings: number;
  };
  threshold: number;
}

interface ProjectDetail {
  id: string;
  name: string;
  cancer_type: string;
  prediction_target: string;
  classes: string[];
  positive_class: string;
  description: string;
  is_default: boolean;
  dataset?: {
    slides_dir: string;
    embeddings_dir: string;
    labels_file: string;
    label_column: string;
  };
  models?: {
    embedder: string;
    mil_architecture: string;
    mil_checkpoint: string;
    report_generator: string;
    semantic_search: string;
  };
  threshold?: number;
}

interface CreateProjectForm {
  id: string;
  name: string;
  cancer_type: string;
  prediction_target: string;
  classes: string;
  positive_class: string;
  description: string;
}

const EMPTY_FORM: CreateProjectForm = {
  id: "",
  name: "",
  cancer_type: "",
  prediction_target: "",
  classes: "class_a, class_b",
  positive_class: "",
  description: "",
};

// -------------------------------------------------------------------
// API helpers
// -------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function fetchProjects(): Promise<ProjectDetail[]> {
  const res = await fetch(`${API_BASE}/api/projects`);
  if (!res.ok) throw new Error(`Failed to fetch projects: ${res.status}`);
  const data = await res.json();
  return data.projects ?? [];
}

async function fetchProjectStatus(projectId: string): Promise<ProjectStatus> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}/status`);
  if (!res.ok) throw new Error(`Failed to fetch status: ${res.status}`);
  return res.json();
}

async function createProject(body: Record<string, unknown>): Promise<ProjectDetail> {
  const res = await fetch(`${API_BASE}/api/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to create project: ${res.status}`);
  }
  return res.json();
}

async function updateProject(
  projectId: string,
  body: Record<string, unknown>
): Promise<ProjectDetail> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to update project: ${res.status}`);
  }
  return res.json();
}

async function deleteProject(projectId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to delete project: ${res.status}`);
  }
}

async function uploadSlide(projectId: string, file: File): Promise<Record<string, unknown>> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(
    `${API_BASE}/api/projects/${encodeURIComponent(projectId)}/upload`,
    { method: "POST", body: formData }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

// -------------------------------------------------------------------
// Sub-components
// -------------------------------------------------------------------

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5 text-xs">
      {ok ? (
        <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />
      ) : (
        <XCircle className="h-3.5 w-3.5 text-gray-300" />
      )}
      <span className={ok ? "text-gray-700" : "text-gray-400"}>{label}</span>
    </div>
  );
}

// -------------------------------------------------------------------
// Create / Edit Modal
// -------------------------------------------------------------------

function ProjectFormModal({
  mode,
  initial,
  onClose,
  onSave,
}: {
  mode: "create" | "edit";
  initial: CreateProjectForm;
  onClose: () => void;
  onSave: (form: CreateProjectForm) => Promise<void>;
}) {
  const [form, setForm] = useState<CreateProjectForm>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSaving(true);
    try {
      await onSave(form);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  };

  const set = (field: keyof CreateProjectForm, value: string) =>
    setForm((f) => ({ ...f, [field]: value }));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-gray-50/50">
          <h2 className="text-lg font-semibold text-gray-900">
            {mode === "create" ? "New Project" : "Edit Project"}
          </h2>
          <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded-lg transition-colors">
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-50 text-red-700 text-sm rounded-lg border border-red-100">
              <AlertTriangle className="h-4 w-4 flex-shrink-0" />
              {error}
            </div>
          )}

          {/* Project ID - only on create */}
          {mode === "create" && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Project ID
              </label>
              <input
                type="text"
                value={form.id}
                onChange={(e) => set("id", e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "-"))}
                placeholder="e.g. lung-immunotherapy"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                required
              />
              <p className="text-xs text-gray-400 mt-1">
                Lowercase, hyphens only. Used in URLs and API.
              </p>
            </div>
          )}

          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Project Name
            </label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => set("name", e.target.value)}
              placeholder="e.g. Lung Cancer - Immunotherapy Response"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
              required
            />
          </div>

          {/* Cancer Type + Prediction Target */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Cancer Type
              </label>
              <input
                type="text"
                value={form.cancer_type}
                onChange={(e) => set("cancer_type", e.target.value)}
                placeholder="e.g. lung, breast, ovarian"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Prediction Target
              </label>
              <input
                type="text"
                value={form.prediction_target}
                onChange={(e) => set("prediction_target", e.target.value)}
                placeholder="e.g. immunotherapy_response"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                required
              />
            </div>
          </div>

          {/* Classes */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Prediction Classes
            </label>
            <input
              type="text"
              value={form.classes}
              onChange={(e) => set("classes", e.target.value)}
              placeholder="e.g. responder, non-responder"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
              required
            />
            <p className="text-xs text-gray-400 mt-1">Comma-separated class labels</p>
          </div>

          {/* Positive Class */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Positive Class
            </label>
            <input
              type="text"
              value={form.positive_class}
              onChange={(e) => set("positive_class", e.target.value)}
              placeholder="e.g. responder"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
              required
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              value={form.description}
              onChange={(e) => set("description", e.target.value)}
              placeholder="Brief description of the prediction task and clinical context..."
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent resize-none"
            />
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={saving}>
              {saving ? "Saving..." : mode === "create" ? "Create Project" : "Save Changes"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------
// Upload Modal
// -------------------------------------------------------------------

function UploadModal({
  projectId,
  projectName,
  onClose,
  onUploaded,
}: {
  projectId: string;
  projectName: string;
  onClose: () => void;
  onUploaded: () => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<Record<string, "pending" | "uploading" | "done" | "error">>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const toast = useToast();

  const ALLOWED_EXTENSIONS = [".svs", ".tiff", ".tif", ".ndpi"];

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || []);
    const valid = selected.filter((f) =>
      ALLOWED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (valid.length < selected.length) {
      toast.warning(
        "Invalid files skipped",
        `Only ${ALLOWED_EXTENSIONS.join(", ")} files are accepted.`
      );
    }
    setFiles((prev) => [...prev, ...valid]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files);
    const valid = dropped.filter((f) =>
      ALLOWED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    setFiles((prev) => [...prev, ...valid]);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    setUploading(true);
    const newProgress: Record<string, "pending" | "uploading" | "done" | "error"> = {};
    const newErrors: Record<string, string> = {};
    files.forEach((f) => (newProgress[f.name] = "pending"));
    setProgress(newProgress);

    for (const file of files) {
      setProgress((p) => ({ ...p, [file.name]: "uploading" }));
      try {
        await uploadSlide(projectId, file);
        setProgress((p) => ({ ...p, [file.name]: "done" }));
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        newErrors[file.name] = msg;
        setProgress((p) => ({ ...p, [file.name]: "error" }));
        setErrors((e) => ({ ...e, [file.name]: msg }));
      }
    }

    setUploading(false);
    const doneCount = files.filter((f) => !newErrors[f.name]).length;
    if (doneCount > 0) {
      toast.success("Upload complete", `${doneCount} slide(s) uploaded to ${projectName}`);
      onUploaded();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-gray-50/50">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Upload Slides</h2>
            <p className="text-xs text-gray-500 mt-0.5">{projectName}</p>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded-lg transition-colors">
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {/* Drop zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="border-2 border-dashed border-gray-200 rounded-xl p-8 text-center hover:border-clinical-400 hover:bg-clinical-50/30 transition-colors cursor-pointer"
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <Upload className="h-8 w-8 text-gray-400 mx-auto mb-3" />
            <p className="text-sm text-gray-600 font-medium">
              Drop WSI files here or click to browse
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Supported: {ALLOWED_EXTENSIONS.join(", ")}
            </p>
            <input
              id="file-input"
              type="file"
              multiple
              accept={ALLOWED_EXTENSIONS.join(",")}
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {files.map((file, i) => (
                <div
                  key={`${file.name}-${i}`}
                  className="flex items-center justify-between px-3 py-2 bg-gray-50 rounded-lg text-sm"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <Microscope className="h-4 w-4 text-gray-400 flex-shrink-0" />
                    <span className="truncate text-gray-700">{file.name}</span>
                    <span className="text-xs text-gray-400 flex-shrink-0">
                      {(file.size / (1024 * 1024)).toFixed(0)} MB
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    {progress[file.name] === "uploading" && (
                      <RefreshCw className="h-3.5 w-3.5 text-clinical-500 animate-spin" />
                    )}
                    {progress[file.name] === "done" && (
                      <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />
                    )}
                    {progress[file.name] === "error" && (
                      <span className="text-xs text-red-500" title={errors[file.name]}>
                        Failed
                      </span>
                    )}
                    {!uploading && (
                      <button
                        onClick={() => removeFile(i)}
                        className="p-0.5 hover:bg-gray-200 rounded"
                      >
                        <X className="h-3.5 w-3.5 text-gray-400" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="ghost" onClick={onClose} disabled={uploading}>
              {uploading ? "Uploading..." : "Cancel"}
            </Button>
            <Button
              onClick={handleUpload}
              disabled={files.length === 0 || uploading}
            >
              {uploading ? "Uploading..." : `Upload ${files.length} File${files.length !== 1 ? "s" : ""}`}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------
// Confirm Delete Modal
// -------------------------------------------------------------------

function DeleteConfirmModal({
  projectName,
  onConfirm,
  onCancel,
}: {
  projectName: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm mx-4 p-6 space-y-4">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-red-50 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-red-500" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Delete Project</h3>
            <p className="text-sm text-gray-500 mt-1">
              Remove <strong>{projectName}</strong> from the system? Slide data files will not be deleted.
            </p>
          </div>
        </div>
        <div className="flex justify-end gap-3">
          <Button variant="ghost" onClick={onCancel}>Cancel</Button>
          <Button
            onClick={onConfirm}
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            Delete
          </Button>
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------
// Project Card
// -------------------------------------------------------------------

function ProjectCard({
  project,
  status,
  onEdit,
  onDelete,
  onUpload,
}: {
  project: ProjectDetail;
  status: ProjectStatus | null;
  onEdit: () => void;
  onDelete: () => void;
  onUpload: () => void;
}) {
  const router = useRouter();

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
      {/* Card header */}
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-gray-900 truncate">
                {project.name}
              </h3>
              {project.is_default && (
                <Badge variant="info" className="text-2xs">Default</Badge>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-0.5 font-mono">{project.id}</p>
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={onEdit}
              className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
              title="Edit project"
            >
              <Pencil className="h-4 w-4 text-gray-400" />
            </button>
            <button
              onClick={onDelete}
              className="p-1.5 hover:bg-red-50 rounded-lg transition-colors"
              title="Delete project"
            >
              <Trash2 className="h-4 w-4 text-gray-400 hover:text-red-500" />
            </button>
          </div>
        </div>
      </div>

      {/* Card body */}
      <div className="px-5 py-4 space-y-3">
        {project.description && (
          <p className="text-sm text-gray-600 line-clamp-2">{project.description}</p>
        )}

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="default" className="text-2xs">
            {project.cancer_type}
          </Badge>
          <Badge variant="info" className="text-2xs">
            Target: {project.prediction_target}
          </Badge>
          {project.classes.map((cls) => (
            <Badge
              key={cls}
              variant={cls === project.positive_class ? "success" : "default"}
              className="text-2xs"
            >
              {cls}
            </Badge>
          ))}
        </div>

        {/* Status */}
        {status && (
          <div className="pt-2 border-t border-gray-50 space-y-2">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
              <StatusBadge ok={status.ready.slides_dir} label="Slides directory" />
              <StatusBadge ok={status.ready.embeddings_dir} label="Embeddings" />
              <StatusBadge ok={status.ready.mil_checkpoint} label="MIL model" />
              <StatusBadge ok={status.ready.labels_file} label="Labels" />
            </div>
            <div className="flex items-center gap-4 text-xs text-gray-500 pt-1">
              <span className="flex items-center gap-1">
                <Microscope className="h-3 w-3" />
                {status.counts.slides} slides
              </span>
              <span className="flex items-center gap-1">
                <Database className="h-3 w-3" />
                {status.counts.embeddings} embeddings
              </span>
              <span className="flex items-center gap-1">
                <Activity className="h-3 w-3" />
                Threshold: {status.threshold.toFixed(3)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Card footer */}
      <div className="px-5 py-3 bg-gray-50/50 border-t border-gray-100 flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onUpload}
          className="text-xs"
        >
          <Upload className="h-3.5 w-3.5 mr-1.5" />
          Upload Slides
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.push(`/slides?project=${project.id}`)}
          className="text-xs"
        >
          <Layers className="h-3.5 w-3.5 mr-1.5" />
          View Slides
        </Button>
        <div className="flex-1" />
        <Button
          variant="secondary"
          size="sm"
          onClick={() => router.push(`/?project=${project.id}`)}
          className="text-xs"
        >
          Open
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------
// Main Page
// -------------------------------------------------------------------

export default function ProjectsPage() {
  const router = useRouter();
  const toast = useToast();
  const { switchProject } = useProject();

  const [projects, setProjects] = useState<ProjectDetail[]>([]);
  const [statuses, setStatuses] = useState<Record<string, ProjectStatus>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Check backend connectivity
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch("/api/health", { signal: AbortSignal.timeout(5000) });
        if (!cancelled) setIsConnected(res.ok);
      } catch (err) {
        if (!cancelled) setIsConnected(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  // Modal state
  const [showCreate, setShowCreate] = useState(false);
  const [editProject, setEditProject] = useState<ProjectDetail | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ProjectDetail | null>(null);
  const [uploadTarget, setUploadTarget] = useState<ProjectDetail | null>(null);

  // Load projects
  const loadProjects = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await fetchProjects();
      setProjects(list);

      // Fetch status for each project
      const statusMap: Record<string, ProjectStatus> = {};
      await Promise.allSettled(
        list.map(async (p) => {
          try {
            statusMap[p.id] = await fetchProjectStatus(p.id);
          } catch (err) {
            // Status fetch failed - skip
            console.warn("Status fetch failed for project:", err);
          }
        })
      );
      setStatuses(statusMap);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load projects");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  // Handlers
  const handleCreate = async (form: CreateProjectForm) => {
    const classes = form.classes.split(",").map((c) => c.trim()).filter(Boolean);
    await createProject({
      id: form.id,
      name: form.name,
      cancer_type: form.cancer_type,
      prediction_target: form.prediction_target,
      classes,
      positive_class: form.positive_class,
      description: form.description,
    });
    toast.success("Project created", `${form.name} is ready for slides`);
    setShowCreate(false);
    loadProjects();
  };

  const handleEdit = async (form: CreateProjectForm) => {
    if (!editProject) return;
    const classes = form.classes.split(",").map((c) => c.trim()).filter(Boolean);
    await updateProject(editProject.id, {
      name: form.name,
      cancer_type: form.cancer_type,
      prediction_target: form.prediction_target,
      classes,
      positive_class: form.positive_class,
      description: form.description,
    });
    toast.success("Project updated", `${form.name} saved`);
    setEditProject(null);
    loadProjects();
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteProject(deleteTarget.id);
      toast.success("Project deleted", `${deleteTarget.name} removed`);
      setDeleteTarget(null);
      loadProjects();
    } catch (err) {
      toast.error("Delete failed", err instanceof Error ? err.message : "Unknown error");
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-gray-50 to-gray-100">
      <Header isConnected={isConnected} />

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 py-6">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-gray-500 mb-6">
          <Link href="/" className="hover:text-gray-700 transition-colors">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <Link href="/" className="hover:text-gray-700">Home</Link>
          <ChevronRight className="h-3 w-3" />
          <span className="text-gray-900 font-medium">Projects</span>
        </div>

        {/* Page header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Project Management</h1>
            <p className="text-sm text-gray-500 mt-1">
              Configure cancer types, prediction targets, and manage slides per project.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={loadProjects} disabled={loading}>
              <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
              Refresh
            </Button>
            <Button onClick={() => setShowCreate(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Project
            </Button>
          </div>
        </div>

        {/* Error state */}
        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 mb-6">
            <AlertTriangle className="h-5 w-5 flex-shrink-0" />
            <div>
              <p className="font-medium">Failed to load projects</p>
              <p className="text-sm mt-0.5">{error}</p>
            </div>
            <Button variant="ghost" size="sm" onClick={loadProjects} className="ml-auto">
              Retry
            </Button>
          </div>
        )}

        {/* Loading state */}
        {loading && projects.length === 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[1, 2].map((i) => (
              <div key={i} className="bg-white rounded-xl border border-gray-200 p-6 animate-pulse">
                <div className="h-5 bg-gray-200 rounded w-2/3 mb-3" />
                <div className="h-3 bg-gray-100 rounded w-1/3 mb-4" />
                <div className="h-12 bg-gray-50 rounded mb-3" />
                <div className="flex gap-2">
                  <div className="h-6 bg-gray-100 rounded w-20" />
                  <div className="h-6 bg-gray-100 rounded w-24" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {!loading && projects.length === 0 && !error && (
          <div className="text-center py-16 bg-white rounded-2xl border border-gray-200">
            <FolderOpen className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900">No projects yet</h3>
            <p className="text-sm text-gray-500 mt-1 max-w-sm mx-auto">
              Create a project to organize slides by cancer type and prediction target.
            </p>
            <Button onClick={() => setShowCreate(true)} className="mt-4">
              <Plus className="h-4 w-4 mr-2" />
              Create First Project
            </Button>
          </div>
        )}

        {/* Project grid */}
        {projects.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {projects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                status={statuses[project.id] ?? null}
                onEdit={() => setEditProject(project)}
                onDelete={() => setDeleteTarget(project)}
                onUpload={() => setUploadTarget(project)}
              />
            ))}
          </div>
        )}

        {/* Quick info */}
        {projects.length > 0 && (
          <div className="mt-8 space-y-3">
            <div className="p-5 bg-blue-50/50 rounded-xl border border-blue-100 text-sm text-blue-800">
              <div className="flex items-start gap-3">
                <Settings className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="font-semibold text-blue-900">How projects work</p>
                  <ul className="mt-2 space-y-1.5 text-blue-700 text-sm">
                    <li>Each project defines a cancer type, prediction classes, and associated models.</li>
                    <li>Upload slides to a project, then generate embeddings from the Batch tab.</li>
                    <li>Train MIL models per-project for accurate treatment response prediction.</li>
                    <li>Switch between projects using the dropdown in the navigation bar.</li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="p-4 bg-gray-50 rounded-xl border border-gray-200 text-sm text-gray-600">
              <p>
                <span className="font-medium text-gray-700">Adding new cancer modules:</span>{" "}
                Full project configuration — including model paths, embedding directories, dataset labels, and MIL checkpoints — is managed via the backend{" "}
                <code className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700 font-mono text-xs">config/projects.yaml</code>{" "}
                file. Projects created through this UI provide basic metadata only; data paths and trained models must be configured server-side.
              </p>
            </div>
          </div>
        )}
      </main>

      <Footer />

      {/* Modals */}
      {showCreate && (
        <ProjectFormModal
          mode="create"
          initial={EMPTY_FORM}
          onClose={() => setShowCreate(false)}
          onSave={handleCreate}
        />
      )}

      {editProject && (
        <ProjectFormModal
          mode="edit"
          initial={{
            id: editProject.id,
            name: editProject.name,
            cancer_type: editProject.cancer_type,
            prediction_target: editProject.prediction_target,
            classes: editProject.classes.join(", "),
            positive_class: editProject.positive_class,
            description: editProject.description,
          }}
          onClose={() => setEditProject(null)}
          onSave={handleEdit}
        />
      )}

      {deleteTarget && (
        <DeleteConfirmModal
          projectName={deleteTarget.name}
          onConfirm={handleDelete}
          onCancel={() => setDeleteTarget(null)}
        />
      )}

      {uploadTarget && (
        <UploadModal
          projectId={uploadTarget.id}
          projectName={uploadTarget.name}
          onClose={() => setUploadTarget(null)}
          onUploaded={() => {
            setUploadTarget(null);
            loadProjects();
          }}
        />
      )}
    </div>
  );
}
