"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  ArrowRight,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  Database,
  Download,
  ExternalLink,
  FileText,
  FolderOpen,
  Grid3X3,
  HelpCircle,
  Home,
  ImageIcon,
  Layers,
  Loader2,
  Menu,
  Microscope,
  Moon,
  MoreVertical,
  PanelLeft,
  Play,
  RefreshCcw,
  Search,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Sun,
  Table2,
  Upload,
  Users,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";
import { useProject } from "@/contexts/ProjectContext";
import {
  analyzeSlideMultiModel,
  cancelBatchAnalysis,
  fetchSimilarCases,
  generateReportWithProgress,
  getDziUrl,
  getHeatmapUrl,
  getProjectAvailableModels,
  getSlideCachedResults,
  getSlides,
  getThumbnailUrl,
  healthCheck,
  pollBatchAnalysis,
  semanticSearch,
  startBatchAnalysisAsync,
  type AsyncBatchTaskStatus,
  type AvailableModelDetail,
} from "@/lib/api";
import type {
  ModelPrediction,
  MultiModelResponse,
  Project,
  SemanticSearchResponse,
  SimilarCase,
  SlideInfo,
  StructuredReport,
} from "@/types";

const WSIViewer = dynamic(
  () => import("@/components/viewer/WSIViewer").then((mod) => mod.WSIViewer),
  {
    ssr: false,
    loading: () => (
      <div className="atlas-viewer-placeholder">
        <Loader2 className="atlas-spin" size={22} />
        Loading whole-slide viewer
      </div>
    ),
  }
);

type RouteKey = "workspace" | "slides" | "batch" | "projects" | "settings";
type ThemeMode = "light" | "dark" | "system";
type ViewMode = "viewer" | "summary";

interface AtlasAppProps {
  initialRoute?: RouteKey;
}

interface Preferences {
  density: "compact" | "comfortable" | "spacious";
  defaultMag: "5x" | "10x" | "20x" | "40x";
  heatmapDefault: boolean;
  heatmapOpacity: number;
  smoothHeatmap: boolean;
  autoRun: boolean;
  multiModel: boolean;
  topK: number;
  reportStyle: "concise" | "standard" | "detailed";
  reportMorphology: boolean;
  reportSimilar: boolean;
  reportAsync: boolean;
  reduceMotion: boolean;
  tooltips: boolean;
}

const DEFAULT_PREFS: Preferences = {
  density: "comfortable",
  defaultMag: "20x",
  heatmapDefault: true,
  heatmapOpacity: 60,
  smoothHeatmap: true,
  autoRun: false,
  multiModel: true,
  topK: 5,
  reportStyle: "standard",
  reportMorphology: true,
  reportSimilar: true,
  reportAsync: true,
  reduceMotion: false,
  tooltips: true,
};

const ROUTES: Array<{ id: RouteKey; label: string; href: string; icon: LucideIcon }> = [
  { id: "workspace", label: "Workspace", href: "/", icon: Microscope },
  { id: "slides", label: "Slide manager", href: "/slides", icon: Grid3X3 },
  { id: "batch", label: "Batch analysis", href: "/batch", icon: Workflow },
  { id: "projects", label: "Projects", href: "/projects", icon: FolderOpen },
  { id: "settings", label: "Settings", href: "/settings", icon: Settings },
];

const FALLBACK_MODEL: AvailableModelDetail = {
  id: "default",
  displayName: "Project model",
  description: "Primary classifier configured for this project.",
  auc: 0,
  category: "project",
  positiveLabel: "Positive",
  negativeLabel: "Negative",
};

function routeHref(route: RouteKey): string {
  return ROUTES.find((item) => item.id === route)?.href ?? "/";
}

function compactId(slide: SlideInfo | null | undefined): string {
  if (!slide) return "";
  return slide.displayName || slide.id;
}

function projectShortName(project: Project): string {
  const cancer = project.cancer_type || project.name;
  const target = project.prediction_target?.replace(/_/g, " ");
  return target ? `${cancer} / ${target}` : cancer;
}

function projectClasses(project: Project): [string, string] {
  const positive = project.positive_class || project.classes?.[1] || "Positive";
  const negative = project.classes?.find((item) => item !== positive) || project.classes?.[0] || "Negative";
  return [positive, negative];
}

function formatPct(value: number | undefined | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

function classTone(label: string | undefined, project: Project): "success" | "error" | "info" {
  if (!label) return "info";
  const [positive, negative] = projectClasses(project);
  if (label.toLowerCase() === positive.toLowerCase()) return "success";
  if (label.toLowerCase() === negative.toLowerCase()) return "error";
  return "info";
}

function usePersistentTheme(): [ThemeMode, (theme: ThemeMode) => void] {
  const [theme, setThemeState] = useState<ThemeMode>("light");

  useEffect(() => {
    const saved = (localStorage.getItem("atlas-theme") as ThemeMode | null) || "light";
    setThemeState(saved);
  }, []);

  const setTheme = useCallback((next: ThemeMode) => {
    setThemeState(next);
    localStorage.setItem("atlas-theme", next);
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", next === "dark" || (next === "system" && prefersDark));
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.classList.toggle("dark", theme === "dark" || (theme === "system" && mq.matches));
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [theme]);

  return [theme, setTheme];
}

function usePersistentPrefs(): [Preferences, (prefs: Preferences) => void] {
  const [prefs, setPrefsState] = useState<Preferences>(DEFAULT_PREFS);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("atlas-prefs") || "null") as Partial<Preferences> | null;
      if (saved) setPrefsState({ ...DEFAULT_PREFS, ...saved });
    } catch {
      setPrefsState(DEFAULT_PREFS);
    }
  }, []);

  const setPrefs = useCallback((next: Preferences) => {
    setPrefsState(next);
    localStorage.setItem("atlas-prefs", JSON.stringify(next));
  }, []);

  return [prefs, setPrefs];
}

function IconButton({
  icon: Icon,
  label,
  onClick,
  active,
  disabled,
}: {
  icon: LucideIcon;
  label: string;
  onClick?: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button className={`atlas-icon-btn ${active ? "active" : ""}`} type="button" title={label} aria-label={label} onClick={onClick} disabled={disabled}>
      <Icon size={19} />
    </button>
  );
}

function Button({
  children,
  icon: Icon,
  trailing: Trailing,
  variant = "filled",
  size = "md",
  onClick,
  disabled,
  type = "button",
}: {
  children: React.ReactNode;
  icon?: LucideIcon;
  trailing?: LucideIcon;
  variant?: "filled" | "tonal" | "outlined" | "text";
  size?: "sm" | "md" | "lg";
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <button className={`atlas-btn ${variant} ${size}`} type={type} onClick={onClick} disabled={disabled}>
      {Icon ? <Icon size={size === "sm" ? 15 : 17} /> : null}
      <span>{children}</span>
      {Trailing ? <Trailing size={size === "sm" ? 15 : 17} /> : null}
    </button>
  );
}

function Chip({
  children,
  tone = "neutral",
  icon: Icon,
  selected,
  onClick,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "primary" | "success" | "error" | "warning" | "info";
  icon?: LucideIcon;
  selected?: boolean;
  onClick?: () => void;
}) {
  const Tag = onClick ? "button" : "span";
  return (
    <Tag className={`atlas-chip tone-${tone} ${selected ? "selected" : ""}`} onClick={onClick as never}>
      {Icon ? <Icon size={13} /> : null}
      {children}
    </Tag>
  );
}

function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  options: Array<{ value: T; label: string; icon?: LucideIcon }>;
}) {
  return (
    <div className="atlas-segmented">
      {options.map((option) => {
        const Icon = option.icon;
        return (
          <button key={option.value} className={value === option.value ? "active" : ""} type="button" onClick={() => onChange(option.value)}>
            {Icon ? <Icon size={15} /> : null}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function Switch({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  return <button className={`atlas-switch ${checked ? "on" : ""}`} type="button" aria-pressed={checked} onClick={() => onChange(!checked)} />;
}

function EnsoMark() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="atlas-enso-grad" x1="0" y1="0" x2="32" y2="32">
          <stop offset="0" stopColor="#0b57d0" />
          <stop offset="0.55" stopColor="#14b8a6" />
          <stop offset="1" stopColor="#0b57d0" />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="13" stroke="url(#atlas-enso-grad)" strokeWidth="3.6" strokeLinecap="round" strokeDasharray="68 12" fill="none" transform="rotate(-30 16 16)" />
      <circle cx="16" cy="16" r="3.7" fill="url(#atlas-enso-grad)" />
    </svg>
  );
}

function ProjectPicker({ projects, currentProject, switchProject }: ReturnType<typeof useProject>) {
  const [open, setOpen] = useState(false);
  if (!projects.length) return null;
  return (
    <div className="atlas-menu-anchor">
      <button className="atlas-project-picker" type="button" onClick={() => setOpen((value) => !value)}>
        <span className="atlas-project-dot" />
        <span>
          <span>Project</span>
          <strong>{projectShortName(currentProject)}</strong>
        </span>
        <ChevronDown size={16} />
      </button>
      {open ? (
        <div className="atlas-menu">
          <div className="atlas-menu-label">Switch project</div>
          {projects.map((project) => (
            <button
              key={project.id}
              className="atlas-menu-item"
              type="button"
              onClick={() => {
                switchProject(project.id);
                setOpen(false);
              }}
            >
              {project.id === currentProject.id ? <CheckCircle2 size={17} /> : <Circle size={17} />}
              <span>
                <strong>{project.name}</strong>
                <small>{project.slide_count ?? 0} slides · {project.prediction_target}</small>
              </span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function AtlasShell({
  children,
  route,
  setRoute,
  theme,
  setTheme,
  railExpanded,
  setRailExpanded,
}: {
  children: React.ReactNode;
  route: RouteKey;
  setRoute: (route: RouteKey) => void;
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;
  railExpanded: boolean;
  setRailExpanded: (expanded: boolean) => void;
}) {
  const router = useRouter();
  const projectCtx = useProject();
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    let cancelled = false;
    healthCheck()
      .then(() => !cancelled && setStatus("online"))
      .catch(() => !cancelled && setStatus("offline"));
    return () => {
      cancelled = true;
    };
  }, []);

  const navigate = (next: RouteKey) => {
    setRoute(next);
    router.push(routeHref(next));
  };

  return (
    <div className={`atlas-root ${railExpanded ? "rail-expanded" : ""}`}>
      <header className="atlas-topbar">
        <div className="atlas-topbar-leading">
          <IconButton icon={PanelLeft} label="Toggle navigation" onClick={() => setRailExpanded(!railExpanded)} />
          <button className="atlas-brand" type="button" onClick={() => navigate("workspace")}>
            <EnsoMark />
            <span>
              <strong>Enso<b>Atlas</b></strong>
              <small>Pathology Evidence Engine</small>
            </span>
          </button>
        </div>
        <ProjectPicker {...projectCtx} />
        <div className="atlas-global-search">
          <Search size={18} />
          <input placeholder="Search slides, cases, models..." aria-label="Search slides, cases, models" />
          <kbd>/</kbd>
        </div>
        <div className="atlas-topbar-actions">
          <span className={`atlas-status-pill ${status}`}>
            <span />
            {status === "online" ? "All systems operational" : status === "offline" ? "Backend offline" : "Checking systems"}
          </span>
          <IconButton icon={theme === "dark" ? Sun : Moon} label="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} />
          <IconButton icon={HelpCircle} label="Help" onClick={() => window.open("https://github.com/Hilo-Hilo/Enso-Atlas/blob/main/docs.md", "_blank")} />
          <IconButton icon={MoreVertical} label="More" />
        </div>
      </header>

      <nav className="atlas-nav-rail" aria-label="Primary">
        {ROUTES.slice(0, 4).map((item) => {
          const Icon = item.icon;
          return (
            <button key={item.id} className={`atlas-nav-item ${route === item.id ? "active" : ""}`} type="button" onClick={() => navigate(item.id)}>
              <span><Icon size={20} /></span>
              <strong>{item.label}</strong>
            </button>
          );
        })}
        <div className="atlas-nav-spacer" />
        <button className={`atlas-nav-item ${route === "settings" ? "active" : ""}`} type="button" onClick={() => navigate("settings")}>
          <span><Settings size={20} /></span>
          <strong>Settings</strong>
        </button>
      </nav>
      <main className="atlas-main">{children}</main>
    </div>
  );
}

export function AtlasApp({ initialRoute = "workspace" }: AtlasAppProps) {
  const [route, setRoute] = useState<RouteKey>(initialRoute);
  const [theme, setTheme] = usePersistentTheme();
  const [prefs, setPrefs] = usePersistentPrefs();
  const [railExpanded, setRailExpanded] = useState(true);
  const { currentProject } = useProject();

  useEffect(() => setRoute(initialRoute), [initialRoute]);

  return (
    <AtlasShell route={route} setRoute={setRoute} theme={theme} setTheme={setTheme} railExpanded={railExpanded} setRailExpanded={setRailExpanded}>
      {route === "workspace" ? <WorkspacePage project={currentProject} prefs={prefs} setRoute={setRoute} /> : null}
      {route === "slides" ? <SlidesPage project={currentProject} setRoute={setRoute} /> : null}
      {route === "batch" ? <BatchPage project={currentProject} /> : null}
      {route === "projects" ? <ProjectsPage /> : null}
      {route === "settings" ? <SettingsPage theme={theme} setTheme={setTheme} prefs={prefs} setPrefs={setPrefs} /> : null}
    </AtlasShell>
  );
}

function useProjectSlides(projectId: string) {
  const [slides, setSlides] = useState<SlideInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    getSlides({ projectId })
      .then((data) => setSlides(data.slides))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load slides"))
      .finally(() => setLoading(false));
  }, [projectId]);

  useEffect(() => refresh(), [refresh]);

  return { slides, loading, error, refresh };
}

function useProjectModels(projectId: string) {
  const [models, setModels] = useState<AvailableModelDetail[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getProjectAvailableModels(projectId)
      .then((items) => !cancelled && setModels(items.length ? items : [FALLBACK_MODEL]))
      .catch(() => !cancelled && setModels([FALLBACK_MODEL]))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  return { models, loading };
}

function EmptyState({ title, body, action }: { title: string; body: string; action?: React.ReactNode }) {
  return (
    <div className="atlas-empty">
      <AlertCircle size={36} />
      <h3>{title}</h3>
      <p>{body}</p>
      {action}
    </div>
  );
}

function SlideThumb({ slide, projectId }: { slide: SlideInfo; projectId?: string }) {
  const url = getThumbnailUrl(slide.id, projectId, 256);
  return (
    <div className="atlas-slide-thumb">
      {/* eslint-disable-next-line @next/next/no-img-element -- backend thumbnails are proxied dynamic slide assets */}
      <img src={url} alt="" onError={(event) => { event.currentTarget.style.display = "none"; }} />
      <div className="atlas-slide-thumb-fallback">
        <ImageIcon size={22} />
      </div>
    </div>
  );
}

function WorkspacePage({ project, prefs, setRoute }: { project: Project; prefs: Preferences; setRoute: (route: RouteKey) => void }) {
  const { slides, loading, error, refresh } = useProjectSlides(project.id);
  const { models } = useProjectModels(project.id);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("viewer");
  const [modelId, setModelId] = useState<string>("");
  const [showHeatmap, setShowHeatmap] = useState(prefs.heatmapDefault);
  const [heatmapOpacity, setHeatmapOpacity] = useState(prefs.heatmapOpacity);
  const [analysis, setAnalysis] = useState<MultiModelResponse | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [analysisMessage, setAnalysisMessage] = useState("");
  const [report, setReport] = useState<StructuredReport | null>(null);
  const [reportProgress, setReportProgress] = useState<{ running: boolean; progress: number; message: string }>({ running: false, progress: 0, message: "" });
  const [similar, setSimilar] = useState<SimilarCase[]>([]);
  const [semantic, setSemantic] = useState<SemanticSearchResponse | null>(null);
  const [semanticQuery, setSemanticQuery] = useState("");

  useEffect(() => {
    setSelectedId(slides[0]?.id ?? null);
  }, [project.id, slides]);

  useEffect(() => {
    setModelId(models[0]?.id ?? "");
  }, [models]);

  const selected = slides.find((slide) => slide.id === selectedId) ?? slides[0] ?? null;
  const selectedPrediction = useMemo(() => {
    if (!analysis || !modelId) return null;
    return analysis.predictions[modelId] ?? Object.values(analysis.predictions)[0] ?? null;
  }, [analysis, modelId]);

  useEffect(() => {
    if (!selected) return;
    setAnalysis(null);
    setReport(null);
    setSimilar([]);
    setSemantic(null);
    setAnalysisStatus("idle");
    getSlideCachedResults(selected.id, project.id)
      .then((cached) => {
        if (!cached.results.length) return;
        const predictions: Record<string, ModelPrediction> = {};
        cached.results.forEach((result) => {
          predictions[result.model_id] = {
            modelId: result.model_id,
            modelName: result.model_id.replace(/[_-]/g, " "),
            category: "cached",
            score: result.score,
            decisionThreshold: result.threshold ?? undefined,
            label: result.label,
            positiveLabel: projectClasses(project)[0],
            negativeLabel: projectClasses(project)[1],
            confidence: result.confidence,
            auc: 0,
            nTrainingSlides: 0,
            description: "Cached backend result",
          };
        });
        setAnalysis({
          slideId: selected.id,
          predictions,
          byCategory: { cancerSpecific: Object.values(predictions), generalPathology: [] },
          nPatches: selected.numPatches ?? 0,
          processingTimeMs: 0,
        });
        setAnalysisStatus("done");
      })
      .catch(() => {});
    fetchSimilarCases(selected.id, prefs.topK, project.id).then(setSimilar).catch(() => setSimilar([]));
  }, [selected, project, prefs.topK]);

  const filteredSlides = slides.filter((slide) => compactId(slide).toLowerCase().includes(query.toLowerCase()));

  const runAnalysis = async () => {
    if (!selected) return;
    setAnalysisStatus("running");
    setAnalysisMessage("Running project-scoped model inference...");
    try {
      const modelIds = prefs.multiModel ? undefined : modelId ? [modelId] : undefined;
      const result = await analyzeSlideMultiModel(selected.id, modelIds, true, 0, false, project.id);
      setAnalysis(result);
      setAnalysisStatus("done");
      setAnalysisMessage("");
      if (!modelId) {
        setModelId(Object.keys(result.predictions)[0] ?? models[0]?.id ?? "");
      }
    } catch (err) {
      setAnalysisStatus("error");
      setAnalysisMessage(err instanceof Error ? err.message : "Analysis failed");
    }
  };

  const generateReport = async () => {
    if (!selected) return;
    setReportProgress({ running: true, progress: 5, message: "Starting MedGemma report..." });
    try {
      const result = await generateReportWithProgress(
        { slideId: selected.id, evidencePatchIds: [], includeDetails: true, projectId: project.id },
        (progress, message) => setReportProgress({ running: true, progress, message })
      );
      setReport(result);
      setReportProgress({ running: false, progress: 100, message: "" });
    } catch (err) {
      setReportProgress({ running: false, progress: 0, message: err instanceof Error ? err.message : "Report failed" });
    }
  };

  const runSemantic = async () => {
    if (!selected || !semanticQuery.trim()) return;
    const result = await semanticSearch(selected.id, semanticQuery.trim(), prefs.topK, project.id);
    setSemantic(result);
  };

  if (loading) {
    return <div className="atlas-page"><EmptyState title="Loading project slides" body="Fetching project-scoped WSI inventory from the backend." /></div>;
  }

  if (error) {
    return (
      <div className="atlas-page">
        <EmptyState title="Slides could not load" body={error} action={<Button variant="tonal" icon={RefreshCcw} onClick={refresh}>Retry</Button>} />
      </div>
    );
  }

  if (!selected) {
    return (
      <div className="atlas-page">
        <EmptyState title="No slides in this project" body="Assign slides to the selected project or switch to a project with available whole-slide images." action={<Button icon={FolderOpen} onClick={() => setRoute("projects")}>Open projects</Button>} />
      </div>
    );
  }

  const currentModel = models.find((model) => model.id === modelId) ?? models[0] ?? FALLBACK_MODEL;
  const [positiveLabel, negativeLabel] = projectClasses(project);

  return (
    <div className="atlas-workspace">
      <aside className="atlas-panel atlas-cases-panel">
        <div className="atlas-panel-header">
          <strong><Microscope size={17} /> Cases</strong>
          <IconButton icon={SlidersHorizontal} label="Filter cases" />
        </div>
        <div className="atlas-panel-tools">
          <div className="atlas-input">
            <Search size={16} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search cases" />
          </div>
          <div className="atlas-muted-row">
            <span>{filteredSlides.length} cases</span>
            <span>{slides.filter((slide) => slide.hasLevel0Embeddings).length} Level-0 ready</span>
          </div>
        </div>
        <div className="atlas-case-list">
          {filteredSlides.slice(0, 80).map((slide) => (
            <button key={slide.id} className={`atlas-case-item ${slide.id === selected.id ? "active" : ""}`} type="button" onClick={() => setSelectedId(slide.id)}>
              <span className={`atlas-case-dot ${slide.hasLevel0Embeddings ? "ready" : slide.hasEmbeddings ? "embedded" : ""}`} />
              <span>{compactId(slide)}</span>
              {slide.id === selected.id ? <CheckCircle2 size={15} /> : null}
            </button>
          ))}
        </div>
      </aside>

      <section className={`atlas-panel atlas-viewer-panel ${viewMode === "summary" ? "summary-mode" : ""}`}>
        <div className="atlas-analysis-bar">
          <Segmented<ViewMode>
            value={viewMode}
            onChange={setViewMode}
            options={[
              { value: "viewer", label: "Viewer", icon: Microscope },
              { value: "summary", label: "Summary", icon: FileText },
            ]}
          />
          <div className="atlas-analysis-actions">
            <select className="atlas-select" value={modelId} onChange={(event) => setModelId(event.target.value)}>
              {models.map((model) => (
                <option key={model.id} value={model.id}>{model.displayName}</option>
              ))}
            </select>
            <IconButton icon={Layers} label="Toggle heatmap" active={showHeatmap} onClick={() => setShowHeatmap(!showHeatmap)} />
          </div>
        </div>

        {viewMode === "viewer" ? (
          <div className="atlas-viewer-wrap">
            <WSIViewer
              slideId={selected.id}
              dziUrl={getDziUrl(selected.id, project.id)}
              hasWsi={selected.hasWsi}
              mpp={selected.mpp}
              heatmap={showHeatmap ? { imageUrl: getHeatmapUrl(selected.id, modelId || undefined, 2, 0.7, project.id, prefs.smoothHeatmap), opacity: heatmapOpacity / 100 } : undefined}
              heatmapModel={modelId}
              availableModels={models.map((model) => ({ id: model.id, name: model.displayName }))}
              heatmapSmooth={prefs.smoothHeatmap}
            />
          </div>
        ) : (
          <SummaryView
            slide={selected}
            project={project}
            prediction={selectedPrediction}
            analysisStatus={analysisStatus}
            runAnalysis={runAnalysis}
            report={report}
            generateReport={generateReport}
            reportProgress={reportProgress}
            similar={similar}
          />
        )}

        <div className="atlas-slide-meta">
          <span><ImageIcon size={14} />{compactId(selected)}</span>
          <span><Grid3X3 size={14} />{(selected.numPatches ?? 0).toLocaleString()} patches</span>
          <span><Search size={14} />{selected.magnification}x</span>
          {selected.hasLevel0Embeddings ? <Chip tone="success" icon={Check}>Level-0</Chip> : <Chip tone={selected.hasEmbeddings ? "info" : "warning"}>{selected.hasEmbeddings ? "Embeddings" : "No embeddings"}</Chip>}
        </div>
      </section>

      <aside className="atlas-panel atlas-results-panel">
        <div className="atlas-tabs">
          <a href="#analysis">Analysis</a>
          <a href="#similar">Similar</a>
          <a href="#report">Report</a>
          <a href="#search">Search</a>
        </div>
        <div className="atlas-results-scroll">
          <section id="analysis" className="atlas-stack">
            <div className="atlas-section-title">Prediction</div>
            {analysisStatus === "running" ? (
              <ProgressCard icon={Loader2} spinning title="Running analysis" body={analysisMessage} progress={55} />
            ) : selectedPrediction ? (
              <PredictionCard prediction={selectedPrediction} project={project} model={currentModel} />
            ) : (
              <div className="atlas-callout">
                <Play size={24} />
                <strong>Run prediction on this slide</strong>
                <p>Project-scoped model inference with cached results when available.</p>
                <Button icon={Play} onClick={runAnalysis}>Run analysis</Button>
              </div>
            )}
            {analysisStatus === "error" ? <div className="atlas-error">{analysisMessage}</div> : null}
            {analysis ? (
              <div className="atlas-mini-list">
                <div className="atlas-section-title">All model outputs</div>
                {Object.values(analysis.predictions).map((prediction) => (
                  <div key={prediction.modelId} className="atlas-model-row">
                    <span>
                      <strong>{prediction.modelName}</strong>
                      <small>{prediction.label}</small>
                    </span>
                    <span className="atlas-bar"><i style={{ width: `${Math.round(prediction.score * 100)}%` }} /></span>
                    <b>{formatPct(prediction.score)}</b>
                  </div>
                ))}
              </div>
            ) : null}
          </section>

          <section id="similar" className="atlas-stack">
            <div className="atlas-section-title">Similar cases</div>
            <div className="atlas-callout compact">
              <Users size={20} />
              <span>{similar.length ? `${similar.length} nearest cases from FAISS retrieval` : "No similar-case results yet."}</span>
            </div>
            {similar.map((item, index) => (
              <div key={`${item.slideId}-${index}`} className="atlas-similar-row">
                <SlideThumb slide={{ ...selected, id: item.slideId }} projectId={project.id} />
                <span>
                  <strong>{item.slideId}</strong>
                  <small>Rank #{index + 1}</small>
                </span>
                <Chip tone={classTone(item.label, project)}>{item.label || "unlabeled"}</Chip>
                <b>{item.similarity.toFixed(2)}</b>
              </div>
            ))}
          </section>

          <section id="report" className="atlas-stack">
            <div className="atlas-section-title">Clinical brief</div>
            {reportProgress.running ? (
              <ProgressCard icon={Sparkles} title="MedGemma is drafting" body={reportProgress.message} progress={reportProgress.progress} />
            ) : report ? (
              <ReportPreview report={report} />
            ) : (
              <div className="atlas-callout">
                <Sparkles size={24} />
                <strong>Generate clinical brief</strong>
                <p>MedGemma will ground a structured draft in prediction, evidence, and similar cases.</p>
                <Button icon={Sparkles} onClick={generateReport}>Generate brief</Button>
              </div>
            )}
            {!reportProgress.running && reportProgress.message ? <div className="atlas-error">{reportProgress.message}</div> : null}
          </section>

          <section id="search" className="atlas-stack">
            <div className="atlas-section-title">Semantic patch search</div>
            <div className="atlas-input atlas-search-row">
              <Search size={16} />
              <input value={semanticQuery} onChange={(event) => setSemanticQuery(event.target.value)} placeholder="dense tumor nests, necrosis..." />
              <Button size="sm" onClick={runSemantic}>Search</Button>
            </div>
            {["dense tumor nests", "stromal lymphocytes", "necrotic core"].map((item) => (
              <Chip key={item} onClick={() => setSemanticQuery(item)}>{item}</Chip>
            ))}
            {semantic ? (
              <div className="atlas-semantic-grid">
                {semantic.results.map((result) => (
                  <div key={result.patch_index}>
                    <SlideThumb slide={selected} projectId={project.id} />
                    <Chip tone="info">{result.similarity.toFixed(2)}</Chip>
                  </div>
                ))}
              </div>
            ) : null}
          </section>
        </div>
      </aside>
    </div>
  );
}

function PredictionCard({ prediction, project, model }: { prediction: ModelPrediction; project: Project; model: AvailableModelDetail }) {
  const tone = classTone(prediction.label, project);
  return (
    <div className={`atlas-prediction-card tone-${tone}`}>
      <span>{model.displayName}</span>
      <div>
        <strong>{prediction.label}</strong>
        <b>{formatPct(prediction.score)}</b>
      </div>
      <small>Threshold {prediction.decisionThreshold?.toFixed(3) ?? "project"} · confidence {formatPct(prediction.confidence)}</small>
      <i><span style={{ width: `${Math.round(prediction.score * 100)}%` }} /></i>
    </div>
  );
}

function ProgressCard({ icon: Icon, title, body, progress, spinning }: { icon: LucideIcon; title: string; body: string; progress: number; spinning?: boolean }) {
  return (
    <div className="atlas-progress-card">
      <div>
        <Icon className={spinning ? "atlas-spin" : ""} size={22} />
        <strong>{title}</strong>
      </div>
      <p>{body}</p>
      <span className="atlas-linear"><i style={{ width: `${progress}%` }} /></span>
    </div>
  );
}

function ReportPreview({ report }: { report: StructuredReport }) {
  return (
    <div className="atlas-report-preview">
      <div>
        <Sparkles size={20} />
        <strong>MedGemma Clinical Brief</strong>
      </div>
      <h4>Impression</h4>
      <p>{report.summary || report.modelOutput.calibrationNote}</p>
      <h4>Decision support</h4>
      <p>{report.decisionSupport?.primary_recommendation || report.safetyStatement}</p>
      <div>
        <Button variant="tonal" size="sm" icon={Download}>Export</Button>
      </div>
    </div>
  );
}

function SummaryView({
  slide,
  project,
  prediction,
  analysisStatus,
  runAnalysis,
  report,
  generateReport,
  reportProgress,
  similar,
}: {
  slide: SlideInfo;
  project: Project;
  prediction: ModelPrediction | null;
  analysisStatus: "idle" | "running" | "done" | "error";
  runAnalysis: () => void;
  report: StructuredReport | null;
  generateReport: () => void;
  reportProgress: { running: boolean; progress: number; message: string };
  similar: SimilarCase[];
}) {
  return (
    <div className="atlas-summary">
      <div className="atlas-summary-head">
        <FileText size={24} />
        <h1>Case Summary</h1>
        <Chip tone="info">{compactId(slide)}</Chip>
      </div>
      {!prediction ? (
        <div className="atlas-callout summary">
          <Play size={34} />
          <strong>{analysisStatus === "running" ? "Analysis running" : "Run analysis to generate a summary"}</strong>
          <p>{project.description}</p>
          <Button icon={Play} onClick={runAnalysis} disabled={analysisStatus === "running"}>{analysisStatus === "running" ? "Running..." : "Run analysis"}</Button>
        </div>
      ) : (
        <>
          <PredictionCard prediction={prediction} project={project} model={{ ...FALLBACK_MODEL, id: prediction.modelId, displayName: prediction.modelName }} />
          <div className="atlas-summary-grid">
            <div className="atlas-card">
              <span>Similar cases</span>
              <strong>{similar.length}</strong>
              <small>nearest neighbors retrieved</small>
            </div>
            <div className="atlas-card">
              <span>Slide</span>
              <strong>{(slide.numPatches ?? 0).toLocaleString()}</strong>
              <small>{slide.magnification}x · {slide.mpp} mpp</small>
            </div>
          </div>
          <div className="atlas-card">
            <div className="atlas-card-head">
              <span>Clinical brief</span>
              {!report && !reportProgress.running ? <Button variant="tonal" size="sm" icon={Sparkles} onClick={generateReport}>Generate</Button> : null}
            </div>
            {reportProgress.running ? <ProgressCard icon={Sparkles} title="Drafting report" body={reportProgress.message} progress={reportProgress.progress} /> : null}
            {report ? <p>{report.summary}</p> : !reportProgress.running ? <p className="atlas-muted">Generate a MedGemma-authored decision brief grounded in this slide analysis.</p> : null}
          </div>
        </>
      )}
    </div>
  );
}

function SlidesPage({ project, setRoute }: { project: Project; setRoute: (route: RouteKey) => void }) {
  const { slides, loading, refresh } = useProjectSlides(project.id);
  const [view, setView] = useState<"grid" | "table">("grid");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<"all" | "embedded" | "level0" | "analyzed">("all");

  const filtered = slides.filter((slide) => {
    if (query && !compactId(slide).toLowerCase().includes(query.toLowerCase())) return false;
    if (filter === "embedded" && !slide.hasEmbeddings) return false;
    if (filter === "level0" && !slide.hasLevel0Embeddings) return false;
    return true;
  });

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="atlas-page">
      <PageHeader eyebrow={projectShortName(project)} title="Slide manager" subtitle="All whole-slide images assigned to this project. Manage embeddings, filter by readiness, and queue cases for batch analysis.">
        <Button variant="outlined" icon={RefreshCcw} onClick={refresh}>Refresh</Button>
        <Button icon={Workflow} disabled={!selected.size} onClick={() => setRoute("batch")}>{selected.size ? `Queue ${selected.size}` : "Queue analysis"}</Button>
      </PageHeader>
      <StatsStrip
        stats={[
          ["Total slides", slides.length, Grid3X3],
          ["With embeddings", slides.filter((s) => s.hasEmbeddings).length, Database],
          ["Level-0 ready", slides.filter((s) => s.hasLevel0Embeddings).length, CheckCircle2],
          ["Labels", slides.filter((s) => s.label).length, BarChart3],
        ]}
      />
      <div className="atlas-toolbar-card">
        <div className="atlas-input">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by slide ID" />
          {query ? <button type="button" onClick={() => setQuery("")}><X size={16} /></button> : null}
        </div>
        <Segmented value={view} onChange={setView} options={[{ value: "grid", label: "Grid", icon: Grid3X3 }, { value: "table", label: "Table", icon: Table2 }]} />
        <div className="atlas-chip-row">
          {(["all", "embedded", "level0", "analyzed"] as const).map((item) => (
            <Chip key={item} selected={filter === item} onClick={() => setFilter(item)}>{item}</Chip>
          ))}
        </div>
      </div>
      {loading ? <EmptyState title="Loading slides" body="Fetching project slides." /> : view === "grid" ? (
        <div className="atlas-slide-grid">
          {filtered.map((slide) => (
            <button key={slide.id} className={`atlas-slide-card ${selected.has(slide.id) ? "selected" : ""}`} type="button" onClick={() => toggle(slide.id)}>
              <SlideThumb slide={slide} projectId={project.id} />
              <span>
                <strong>{compactId(slide)}</strong>
                <small>{(slide.numPatches ?? 0).toLocaleString()} patches · {slide.magnification}x</small>
              </span>
              {slide.hasLevel0Embeddings ? <Chip tone="success">L0</Chip> : slide.hasEmbeddings ? <Chip tone="info">embedded</Chip> : <Chip tone="warning">no emb.</Chip>}
            </button>
          ))}
        </div>
      ) : (
        <SlideTable slides={filtered} selected={selected} toggle={toggle} project={project} />
      )}
    </div>
  );
}

function SlideTable({ slides, selected, toggle, project }: { slides: SlideInfo[]; selected: Set<string>; toggle: (id: string) => void; project: Project }) {
  return (
    <div className="atlas-table-card">
      <table>
        <thead><tr><th></th><th>Slide ID</th><th>Patches</th><th>Magnification</th><th>Embeddings</th><th>Label</th></tr></thead>
        <tbody>
          {slides.map((slide) => (
            <tr key={slide.id} onClick={() => toggle(slide.id)}>
              <td><input checked={selected.has(slide.id)} onChange={() => toggle(slide.id)} onClick={(event) => event.stopPropagation()} type="checkbox" /></td>
              <td><span className="atlas-table-slide"><SlideThumb slide={slide} projectId={project.id} />{compactId(slide)}</span></td>
              <td>{(slide.numPatches ?? 0).toLocaleString()}</td>
              <td>{slide.magnification}x</td>
              <td>{slide.hasLevel0Embeddings ? <Chip tone="success">Level 0</Chip> : slide.hasEmbeddings ? <Chip tone="info">Default</Chip> : <Chip tone="warning">None</Chip>}</td>
              <td>{slide.label ? <Chip tone={classTone(slide.label, project)}>{slide.label}</Chip> : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BatchPage({ project }: { project: Project }) {
  const { slides } = useProjectSlides(project.id);
  const { models } = useProjectModels(project.id);
  const [selectedSlides, setSelectedSlides] = useState<Set<string>>(new Set());
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [task, setTask] = useState<AsyncBatchTaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setSelectedModels(new Set(models[0]?.id ? [models[0].id] : [])), [models]);

  const toggleSlide = (id: string) => {
    setSelectedSlides((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleModel = (id: string) => {
    setSelectedModels((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const start = async () => {
    setError(null);
    try {
      const startResp = await startBatchAnalysisAsync(Array.from(selectedSlides), 4, {
        modelIds: Array.from(selectedModels),
        level: 0,
        projectId: project.id,
      });
      const finalStatus = await pollBatchAnalysis(startResp.task_id, setTask);
      setTask(finalStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch analysis failed");
    }
  };

  const cancel = async () => {
    if (!task) return;
    await cancelBatchAnalysis(task.task_id);
  };

  return (
    <div className="atlas-page">
      <PageHeader eyebrow={projectShortName(project)} title="Batch analysis" subtitle="Queue multiple slides through one or more project classifiers. Tasks run asynchronously and remain local to this deployment." />
      <div className="atlas-batch-grid">
        <div className="atlas-stack">
          <div className="atlas-card flush">
            <div className="atlas-card-head padded">
              <span>Step 1 · Select slides</span>
              <div>
                <Button variant="text" size="sm" onClick={() => setSelectedSlides(new Set(slides.map((s) => s.id)))}>Select all</Button>
                <Button variant="text" size="sm" onClick={() => setSelectedSlides(new Set())}>Clear</Button>
              </div>
            </div>
            <div className="atlas-scroll-table">
              {slides.map((slide) => (
                <button key={slide.id} className="atlas-batch-row" type="button" onClick={() => toggleSlide(slide.id)}>
                  <input checked={selectedSlides.has(slide.id)} readOnly type="checkbox" />
                  <span>{compactId(slide)}</span>
                  <small>{(slide.numPatches ?? 0).toLocaleString()} patches</small>
                  {slide.hasLevel0Embeddings ? <Chip tone="success">L0</Chip> : null}
                </button>
              ))}
            </div>
          </div>
          <div className="atlas-card">
            <div className="atlas-section-title">Step 2 · Choose models</div>
            <div className="atlas-model-grid">
              {models.map((model) => (
                <button key={model.id} className={selectedModels.has(model.id) ? "active" : ""} type="button" onClick={() => toggleModel(model.id)}>
                  <span><strong>{model.displayName}</strong><small>{model.description}</small></span>
                  {selectedModels.has(model.id) ? <CheckCircle2 size={20} /> : <Circle size={20} />}
                </button>
              ))}
            </div>
          </div>
        </div>
        <aside className="atlas-stack">
          <div className="atlas-card">
            <div className="atlas-section-title">Run summary</div>
            <MetricRow label="Slides" value={selectedSlides.size} />
            <MetricRow label="Models" value={selectedModels.size} />
            <MetricRow label="Total jobs" value={selectedSlides.size * selectedModels.size} />
            <Button size="lg" icon={Play} disabled={!selectedSlides.size || !selectedModels.size || task?.status === "running"} onClick={start}>
              {task?.status === "running" ? "Running..." : "Start batch analysis"}
            </Button>
            {error ? <div className="atlas-error">{error}</div> : null}
          </div>
          {task ? (
            <div className="atlas-card">
              <div className="atlas-card-head">
                <span>{task.status}</span>
                <small>{task.task_id.slice(-8)}</small>
              </div>
              <p className="atlas-muted">{task.message}</p>
              <span className="atlas-linear"><i style={{ width: `${task.progress}%` }} /></span>
              <MetricRow label="Completed" value={`${task.completed_slides} / ${task.total_slides}`} />
              {task.status === "running" ? <Button variant="text" icon={X} onClick={cancel}>Cancel</Button> : <Button variant="tonal" icon={Download}>Export CSV</Button>}
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}

function ProjectsPage() {
  const { projects, currentProject, switchProject, isLoading } = useProject();
  return (
    <div className="atlas-page">
      <PageHeader eyebrow="Project registry" title="Projects" subtitle="Configure cancer types, prediction targets, and project-scoped slide/model resources sourced from backend project metadata.">
        <Button variant="outlined" icon={RefreshCcw} onClick={() => window.location.reload()}>Refresh</Button>
      </PageHeader>
      {isLoading ? <EmptyState title="Loading projects" body="Fetching project registry." /> : (
        <div className="atlas-project-grid">
          {projects.map((project) => {
            const active = project.id === currentProject.id;
            const [positive, negative] = projectClasses(project);
            return (
              <div key={project.id} className={`atlas-project-card ${active ? "active" : ""}`}>
                <div className="atlas-project-banner" />
                <div className="atlas-project-body">
                  <div className="atlas-project-icon"><FolderOpen size={28} /></div>
                  <div className="atlas-card-head">
                    <span>
                      <strong>{project.name}</strong>
                      <small>{project.id}</small>
                    </span>
                    {active ? <Chip tone="primary">Active</Chip> : null}
                  </div>
                  <p>{project.description}</p>
                  <div className="atlas-chip-row">
                    <Chip icon={Microscope}>{project.cancer_type}</Chip>
                    <Chip tone="primary">Target: {project.prediction_target}</Chip>
                    <Chip tone="success">{positive}</Chip>
                    <Chip tone="error">{negative}</Chip>
                  </div>
                  <div className="atlas-project-stats">
                    <ProjectStat icon={Grid3X3} label="Slides" value={project.slide_count ?? 0} />
                    <ProjectStat icon={Users} label="Patients" value={project.patient_count ?? 0} />
                    <ProjectStat icon={Database} label="Embedder" value={project.models?.embedder || "—"} />
                    <ProjectStat icon={Sparkles} label="Reports" value={project.models?.report_generator || "—"} />
                  </div>
                  <div className="atlas-card-actions">
                    <Button variant={active ? "tonal" : "filled"} trailing={ArrowRight} onClick={() => switchProject(project.id)}>{active ? "Active" : "Open"}</Button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SettingsPage({ theme, setTheme, prefs, setPrefs }: { theme: ThemeMode; setTheme: (theme: ThemeMode) => void; prefs: Preferences; setPrefs: (prefs: Preferences) => void }) {
  const [section, setSection] = useState("appearance");
  const sections = [
    ["appearance", "Appearance", Sun],
    ["viewer", "Viewer", Microscope],
    ["analysis", "Analysis", SlidersHorizontal],
    ["reporting", "Reporting", Sparkles],
    ["system", "System", Database],
    ["about", "About", HelpCircle],
  ] as const;

  return (
    <div className="atlas-page atlas-settings-page">
      <PageHeader eyebrow="Preferences" title="Settings" subtitle="Local presentation, viewer, analysis, and report defaults. No login or clinician-profile settings are shown because the backend does not support them." />
      <div className="atlas-settings-grid">
        <nav className="atlas-settings-nav">
          {sections.map(([id, label, Icon]) => (
            <button key={id} className={section === id ? "active" : ""} type="button" onClick={() => setSection(id)}>
              <Icon size={18} />{label}
            </button>
          ))}
        </nav>
        <div className="atlas-stack">
          {section === "appearance" ? (
            <SettingsSection title="Appearance" description="Theme, density and motion preferences.">
              <div className="atlas-theme-grid">
                {(["light", "dark", "system"] as const).map((item) => (
                  <button key={item} className={theme === item ? "active" : ""} type="button" onClick={() => setTheme(item)}>
                    <span className={`atlas-theme-preview ${item}`} />
                    <strong>{item}</strong>
                    {theme === item ? <CheckCircle2 size={18} /> : null}
                  </button>
                ))}
              </div>
              <SettingsRow label="Density" description="Adjust spacing across dashboards.">
                <Segmented value={prefs.density} onChange={(density) => setPrefs({ ...prefs, density })} options={[{ value: "compact", label: "Compact" }, { value: "comfortable", label: "Comfortable" }, { value: "spacious", label: "Spacious" }]} />
              </SettingsRow>
              <SettingsRow label="Reduce motion" description="Disable non-essential transitions."><Switch checked={prefs.reduceMotion} onChange={(reduceMotion) => setPrefs({ ...prefs, reduceMotion })} /></SettingsRow>
            </SettingsSection>
          ) : null}
          {section === "viewer" ? (
            <SettingsSection title="Slide viewer" description="Defaults for whole-slide viewing and heatmaps.">
              <SettingsRow label="Default magnification"><Segmented value={prefs.defaultMag} onChange={(defaultMag) => setPrefs({ ...prefs, defaultMag })} options={[{ value: "5x", label: "5x" }, { value: "10x", label: "10x" }, { value: "20x", label: "20x" }, { value: "40x", label: "40x" }]} /></SettingsRow>
              <SettingsRow label="Heatmap on by default"><Switch checked={prefs.heatmapDefault} onChange={(heatmapDefault) => setPrefs({ ...prefs, heatmapDefault })} /></SettingsRow>
              <SettingsRow label="Default heatmap opacity"><input className="atlas-range" min={0} max={100} type="range" value={prefs.heatmapOpacity} onChange={(event) => setPrefs({ ...prefs, heatmapOpacity: Number(event.target.value) })} /></SettingsRow>
              <SettingsRow label="Interpolated heatmap"><Switch checked={prefs.smoothHeatmap} onChange={(smoothHeatmap) => setPrefs({ ...prefs, smoothHeatmap })} /></SettingsRow>
            </SettingsSection>
          ) : null}
          {section === "analysis" ? (
            <SettingsSection title="Analysis defaults" description="Project-scoped inference and retrieval defaults.">
              <SettingsRow label="Auto-run on slide open"><Switch checked={prefs.autoRun} onChange={(autoRun) => setPrefs({ ...prefs, autoRun })} /></SettingsRow>
              <SettingsRow label="Multi-model inference"><Switch checked={prefs.multiModel} onChange={(multiModel) => setPrefs({ ...prefs, multiModel })} /></SettingsRow>
              <SettingsRow label="Similar cases top K"><input className="atlas-range" min={1} max={20} type="range" value={prefs.topK} onChange={(event) => setPrefs({ ...prefs, topK: Number(event.target.value) })} /></SettingsRow>
            </SettingsSection>
          ) : null}
          {section === "reporting" ? (
            <SettingsSection title="Clinical reporting" description="MedGemma report generation defaults.">
              <SettingsRow label="Report style"><Segmented value={prefs.reportStyle} onChange={(reportStyle) => setPrefs({ ...prefs, reportStyle })} options={[{ value: "concise", label: "Concise" }, { value: "standard", label: "Standard" }, { value: "detailed", label: "Detailed" }]} /></SettingsRow>
              <SettingsRow label="Include morphologic features"><Switch checked={prefs.reportMorphology} onChange={(reportMorphology) => setPrefs({ ...prefs, reportMorphology })} /></SettingsRow>
              <SettingsRow label="Include similar-case evidence"><Switch checked={prefs.reportSimilar} onChange={(reportSimilar) => setPrefs({ ...prefs, reportSimilar })} /></SettingsRow>
              <SettingsRow label="Generate asynchronously"><Switch checked={prefs.reportAsync} onChange={(reportAsync) => setPrefs({ ...prefs, reportAsync })} /></SettingsRow>
            </SettingsSection>
          ) : null}
          {section === "system" ? (
            <SettingsSection title="System" description="Local deployment endpoints and resources.">
              <StatsStrip stats={[["Backend API", "same-origin /api", Database], ["Swagger", "/api/docs", ExternalLink], ["Model runtime", "local", Sparkles]]} />
            </SettingsSection>
          ) : null}
          {section === "about" ? (
            <SettingsSection title="About Enso Atlas">
              <div className="atlas-about">
                <EnsoMark />
                <div><strong>Enso Atlas</strong><p>Fully local clinical decision support for pathology, built around project-scoped WSI analysis, evidence retrieval, and MedGemma reporting.</p></div>
              </div>
            </SettingsSection>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function PageHeader({ eyebrow, title, subtitle, children }: { eyebrow: string; title: string; subtitle: string; children?: React.ReactNode }) {
  return (
    <div className="atlas-page-header">
      <div>
        <div className="atlas-eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      {children ? <div className="atlas-page-actions">{children}</div> : null}
    </div>
  );
}

function StatsStrip({ stats }: { stats: Array<[string, string | number, LucideIcon]> }) {
  return (
    <div className="atlas-stats-strip">
      {stats.map(([label, value, Icon]) => (
        <div key={label} className="atlas-stat-card">
          <span><Icon size={19} /></span>
          <div><strong>{value}</strong><small>{label}</small></div>
        </div>
      ))}
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="atlas-metric-row"><span>{label}</span><strong>{value}</strong></div>;
}

function ProjectStat({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: React.ReactNode }) {
  return <div className="atlas-project-stat"><Icon size={17} /><span><small>{label}</small><strong>{value}</strong></span></div>;
}

function SettingsSection({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return <section className="atlas-card atlas-settings-section"><h2>{title}</h2>{description ? <p>{description}</p> : null}{children}</section>;
}

function SettingsRow({ label, description, children }: { label: string; description?: string; children: React.ReactNode }) {
  return <div className="atlas-settings-row"><span><strong>{label}</strong>{description ? <small>{description}</small> : null}</span>{children}</div>;
}
