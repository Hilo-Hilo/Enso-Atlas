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
import { getClientApiBaseUrl } from "@/lib/clientApiBase";
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
import { healthCheck } from "@/lib/api";

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

interface GdcDownloadItem {
  status: "downloaded" | "exists" | "error";
  requested_file_id: string;
  resolved_file_id?: string;
  slide_id?: string;
  file_name?: string;
  source?: string;
  stale_fallback?: boolean;
  path?: string;
  error?: string;
}

interface GdcDownloadResponse {
  project_id: string;
  mode: "specific" | "random";
  requested: number;
  downloaded: number;
  existing: number;
  failed: number;
  results: GdcDownloadItem[];
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

const API_BASE = getClientApiBaseUrl();

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

async function downloadSlidesFromGdc(
  projectId: string,
  body: {
    mode: "specific" | "random";
    file_id?: string;
    barcode?: string;
    count?: number;
    tcga_projects?: string[];
    force?: boolean;
    use_master_csv?: boolean;
  }
): Promise<GdcDownloadResponse> {
  const res = await fetch(
    `${API_BASE}/api/projects/${encodeURIComponent(projectId)}/download-gdc`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail ?? err);
    throw new Error(detail || `GDC download failed: ${res.status}`);
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
  const [source, setSource] = useState<"local" | "gdc">("local");

  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<Record<string, "pending" | "uploading" | "done" | "error">>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const [gdcMode, setGdcMode] = useState<"specific" | "random">("specific");
  const [gdcFileId, setGdcFileId] = useState("");
  const [gdcBarcode, setGdcBarcode] = useState("");
  const [gdcCount, setGdcCount] = useState(1);
  const [gdcProjectFilter, setGdcProjectFilter] = useState("");
  const [gdcDownloading, setGdcDownloading] = useState(false);
  const [gdcResult, setGdcResult] = useState<GdcDownloadResponse | null>(null);

  const toast = useToast();
  const isBusy = uploading || gdcDownloading;

  const ALLOWED_EXTENSIONS = [".svs", ".tiff", ".tif", ".ndpi"];
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024; // 10 GiB

  const validateSelectedFiles = (incoming: File[]): File[] => {
    const validExtension = incoming.filter((f) =>
      ALLOWED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))
    );

    const oversized = validExtension.filter((f) => f.size > MAX_UPLOAD_BYTES);
    if (oversized.length > 0) {
      toast.warning(
        "Large files skipped",
        `${oversized.length} file(s) exceed 10 GiB upload limit.`
      );
    }

    const valid = validExtension.filter((f) => f.size <= MAX_UPLOAD_BYTES);

    if (validExtension.length < incoming.length) {
      toast.warning(
        "Invalid files skipped",
        `Only ${ALLOWED_EXTENSIONS.join(", ")} files are accepted.`
      );
    }

    return valid;
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = validateSelectedFiles(Array.from(e.target.files || []));
    setFiles((prev) => [...prev, ...selected]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = validateSelectedFiles(Array.from(e.dataTransfer.files));
    setFiles((prev) => [...prev, ...dropped]);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    setUploading(true);
    setGdcResult(null);
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

  const handleGdcDownload = async () => {
    setGdcDownloading(true);
    setGdcResult(null);
    try {
      const projectFilters = gdcProjectFilter
        .split(",")
        .map((s) => s.trim().toUpperCase())
        .filter(Boolean);

      const response = await downloadSlidesFromGdc(projectId, {
        mode: gdcMode,
        file_id: gdcMode === "specific" ? (gdcFileId.trim() || undefined) : undefined,
        barcode: gdcMode === "specific" ? (gdcBarcode.trim() || undefined) : undefined,
        count: gdcMode === "random" ? gdcCount : 1,
        tcga_projects: projectFilters.length > 0 ? projectFilters : undefined,
        force: false,
        use_master_csv: true,
      });

      setGdcResult(response);

      if (response.downloaded > 0) {
        toast.success(
          "GDC download complete",
          `${response.downloaded} slide(s) downloaded to ${projectName}`
        );
        onUploaded();
        return;
      }

      if (response.existing > 0 && response.failed === 0) {
        toast.success(
          "Slides already available",
          `${response.existing} slide(s) already existed in ${projectName}`
        );
        onUploaded();
        return;
      }

      const firstError = response.results.find((r) => r.status === "error")?.error;
      toast.error("GDC download failed", firstError || "No slides were downloaded");
    } catch (err) {
      toast.error("GDC download failed", err instanceof Error ? err.message : "Unknown error");
    } finally {
      setGdcDownloading(false);
    }
  };

  const canSubmitGdc =
    gdcMode === "specific"
      ? Boolean(gdcFileId.trim() || gdcBarcode.trim())
      : gdcCount >= 1 && gdcCount <= 20;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-gray-50/50">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Upload Slides</h2>
            <p className="text-xs text-gray-500 mt-0.5">{projectName}</p>
          </div>
          <button
            onClick={onClose}
            disabled={isBusy}
            className="p-1 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
          >
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          <div className="inline-flex w-full p-1 rounded-lg bg-gray-100">
            <button
              type="button"
              onClick={() => setSource("local")}
              disabled={isBusy}
              className={cn(
                "flex-1 px-3 py-2 text-xs font-medium rounded-md transition-colors",
                source === "local" ? "bg-white text-gray-900 shadow-sm" : "text-gray-600 hover:text-gray-900"
              )}
            >
              Upload Local Files
            </button>
            <button
              type="button"
              onClick={() => setSource("gdc")}
              disabled={isBusy}
              className={cn(
                "flex-1 px-3 py-2 text-xs font-medium rounded-md transition-colors",
                source === "gdc" ? "bg-white text-gray-900 shadow-sm" : "text-gray-600 hover:text-gray-900"
              )}
            >
              Download From GDC
            </button>
          </div>

          {source === "local" ? (
            <>
              {/* Drop zone */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                className="border-2 border-dashed border-gray-200 rounded-xl p-8 text-center hover:border-clinical-400 hover:bg-clinical-50/30 transition-colors cursor-pointer"
                onClick={() => document.getElementById("file-input-local")?.click()}
              >
                <Upload className="h-8 w-8 text-gray-400 mx-auto mb-3" />
                <p className="text-sm text-gray-600 font-medium">
                  Drop WSI files here or click to browse
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  Supported: {ALLOWED_EXTENSIONS.join(", ")} · Max 10 GiB each
                </p>
                <input
                  id="file-input-local"
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
                            type="button"
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
            </>
          ) : (
            <>
              <div className="space-y-3 rounded-xl border border-gray-200 p-4 bg-gray-50/50">
                <div className="inline-flex w-full p-1 rounded-lg bg-white border border-gray-200">
                  <button
                    type="button"
                    onClick={() => setGdcMode("specific")}
                    disabled={gdcDownloading}
                    className={cn(
                      "flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
                      gdcMode === "specific" ? "bg-gray-900 text-white" : "text-gray-600"
                    )}
                  >
                    Specific Slide
                  </button>
                  <button
                    type="button"
                    onClick={() => setGdcMode("random")}
                    disabled={gdcDownloading}
                    className={cn(
                      "flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
                      gdcMode === "random" ? "bg-gray-900 text-white" : "text-gray-600"
                    )}
                  >
                    Random Slides
                  </button>
                </div>

                {gdcMode === "specific" ? (
                  <div className="space-y-2">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        GDC File ID (UUID)
                      </label>
                      <input
                        type="text"
                        value={gdcFileId}
                        onChange={(e) => setGdcFileId(e.target.value.trim())}
                        placeholder="e.g. 53d9aebf-..."
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Optional TCGA Barcode
                      </label>
                      <input
                        type="text"
                        value={gdcBarcode}
                        onChange={(e) => setGdcBarcode(e.target.value.trim())}
                        placeholder="e.g. TCGA-2G-AAGX-01Z-00-DX1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Number of random slides (1-20)
                      </label>
                      <input
                        type="number"
                        min={1}
                        max={20}
                        value={gdcCount}
                        onChange={(e) => setGdcCount(Number(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Optional TCGA project filters (comma-separated)
                      </label>
                      <input
                        type="text"
                        value={gdcProjectFilter}
                        onChange={(e) => setGdcProjectFilter(e.target.value)}
                        placeholder="e.g. TCGA-BRCA,TCGA-LUAD"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                      />
                    </div>
                  </div>
                )}

                <p className="text-xs text-gray-500">
                  Uses public GDC sources with stale-UUID barcode fallback, matching your embedding script strategy.
                </p>
              </div>

              {gdcResult && (
                <div className="rounded-xl border border-gray-200 bg-white p-3 space-y-2">
                  <div className="text-xs font-medium text-gray-700">
                    Downloaded {gdcResult.downloaded} · Existing {gdcResult.existing} · Failed {gdcResult.failed}
                  </div>
                  {gdcResult.results.slice(0, 6).map((item, idx) => (
                    <div
                      key={`${item.requested_file_id}-${idx}`}
                      className="text-xs text-gray-600 bg-gray-50 rounded-md px-2 py-1"
                    >
                      <span className="font-mono">{item.requested_file_id}</span>{" "}
                      <span className="uppercase">{item.status}</span>
                      {item.source ? ` via ${item.source}` : ""}
                      {item.error ? ` · ${item.error}` : ""}
                    </div>
                  ))}
                  {gdcResult.results.length > 6 && (
                    <div className="text-2xs text-gray-400">
                      Showing 6 of {gdcResult.results.length} results
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="ghost" onClick={onClose} disabled={isBusy}>
              {isBusy ? "Working..." : "Cancel"}
            </Button>

            {source === "local" ? (
              <Button
                onClick={handleUpload}
                disabled={files.length === 0 || uploading}
              >
                {uploading ? "Uploading..." : `Upload ${files.length} File${files.length !== 1 ? "s" : ""}`}
              </Button>
            ) : (
              <Button
                onClick={handleGdcDownload}
                disabled={!canSubmitGdc || gdcDownloading}
              >
                {gdcDownloading ? (
                  <span className="inline-flex items-center gap-2">
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    Downloading...
                  </span>
                ) : (
                  "Download From GDC"
                )}
              </Button>
            )}
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

  // Check backend connectivity and recover immediately after idle/background.
  useEffect(() => {
    let cancelled = false;
    let failureStreak = 0;

    const check = async () => {
      try {
        await healthCheck();
        failureStreak = 0;
        if (!cancelled) setIsConnected(true);
      } catch (err) {
        failureStreak += 1;
        if (!cancelled && failureStreak >= 2) {
          setIsConnected(false);
        }
      }
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void check();
      }
    };

    const handleWindowFocus = () => {
      void check();
    };

    check();
    const interval = setInterval(check, 15000);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("focus", handleWindowFocus);
    window.addEventListener("online", handleWindowFocus);

    return () => {
      cancelled = true;
      clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("focus", handleWindowFocus);
      window.removeEventListener("online", handleWindowFocus);
    };
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
        {/* Page header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push("/")}
              className="gap-1.5"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
            <div className="h-6 w-px bg-gray-200" />
            <div className="flex items-center gap-3">
              <div className="p-2 bg-clinical-50 rounded-lg">
                <FolderOpen className="h-5 w-5 text-clinical-600" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Project Management</h1>
                <p className="text-sm text-gray-500">
                  Configure cancer types, prediction targets, and manage slides per project.
                </p>
              </div>
            </div>
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

        {/* Config hint */}
        {projects.length > 0 && (
          <p className="mt-6 text-xs text-gray-400 text-center">
            Project configuration is managed via <code className="px-1 py-0.5 bg-gray-100 rounded font-mono">config/projects.yaml</code> on the backend.
          </p>
        )}
      </main>

      {/* Footer - hidden on mobile for layout consistency */}
      <div className="hidden sm:block">
        <Footer />
      </div>

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
