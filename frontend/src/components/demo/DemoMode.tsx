"use client";

import React, { useState, useCallback, useEffect } from "react";
import Joyride, {
  CallBackProps,
  STATUS,
  Step,
  EVENTS,
  ACTIONS,
  TooltipRenderProps,
} from "react-joyride";
import { cn } from "@/lib/utils";
import {
  Play,
  X,
  ChevronLeft,
  ChevronRight,
  Microscope,
  Sparkles,
  Target,
  BarChart3,
  FileText,
  Layers,
  Zap,
  Brain,
} from "lucide-react";

interface DemoModeProps {
  isActive: boolean;
  onClose: () => void;
  onStepChange?: (step: number) => void;
}

// Custom tooltip component for a more impressive look
function CustomTooltip({
  continuous,
  index,
  step,
  backProps,
  closeProps,
  primaryProps,
  tooltipProps,
  isLastStep,
  size,
}: TooltipRenderProps) {
  const progress = ((index + 1) / size) * 100;

  return (
    <div
      {...tooltipProps}
      className="bg-white dark:bg-navy-800 rounded-2xl shadow-2xl border border-gray-100 dark:border-navy-600 max-w-md overflow-hidden animate-scale-in"
    >
      {/* Progress bar */}
      <div className="h-1 bg-gray-100 dark:bg-navy-700">
        <div
          className="h-full bg-gradient-to-r from-clinical-500 to-clinical-600 transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Header with step icon */}
      <div className="px-6 pt-5 pb-3">
        <div className="flex items-start gap-4">
          <div className="shrink-0">
            <div
              className={cn(
                "w-12 h-12 rounded-xl flex items-center justify-center",
                "bg-gradient-to-br from-clinical-500 to-clinical-600 shadow-lg"
              )}
            >
              {(step.data as { icon?: React.ReactNode })?.icon || (
                <Sparkles className="w-6 h-6 text-white" />
              )}
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-semibold text-clinical-600 dark:text-clinical-300 bg-clinical-50 dark:bg-clinical-900/40 px-2 py-0.5 rounded-full">
                Step {index + 1} of {size}
              </span>
            </div>
            {step.title && (
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 leading-tight">
                {step.title}
              </h3>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 pb-4">
        <div className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
          {step.content}
        </div>
      </div>

      {/* Feature highlights if present */}
      {(step.data as { features?: string[] })?.features && (
        <div className="px-6 pb-4">
          <div className="bg-gray-50 dark:bg-navy-700/60 rounded-lg p-3 space-y-2">
            {(step.data as { features: string[] }).features.map((feature, i) => (
              <div key={i} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-200">
                <div className="w-5 h-5 rounded-full bg-clinical-100 dark:bg-clinical-900/40 flex items-center justify-center shrink-0">
                  <Zap className="w-3 h-3 text-clinical-600 dark:text-clinical-300" />
                </div>
                {feature}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="px-6 pb-5 flex items-center justify-between gap-3">
        <button
          {...closeProps}
          className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
        >
          Skip tour
        </button>
        <div className="flex items-center gap-2">
          {index > 0 && (
            <button
              {...backProps}
              className="flex items-center gap-1 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-navy-700 border border-transparent dark:border-navy-500 hover:bg-gray-200 dark:hover:bg-navy-600 rounded-lg transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              Back
            </button>
          )}
          <button
            {...primaryProps}
            className={cn(
              "flex items-center gap-1 px-5 py-2 text-sm font-medium rounded-lg transition-all",
              "bg-gradient-to-r from-clinical-500 to-clinical-600 text-white",
              "hover:from-clinical-600 hover:to-clinical-700 shadow-md hover:shadow-lg"
            )}
          >
            {isLastStep ? (
              <>
                Get Started
                <Sparkles className="w-4 h-4" />
              </>
            ) : (
              <>
                Next
                <ChevronRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// Tour steps with rich content
const tourSteps: Step[] = [
  {
    target: '[data-demo="slide-selector"]',
    title: "Select a Pathology Slide",
    content: (
      <div className="space-y-2">
        <p>
          Start by selecting a whole-slide image (WSI) from our curated dataset. 
          Each slide contains a biopsy sample from a cancer patient.
        </p>
        <p className="text-clinical-600 dark:text-clinical-300 font-medium">
          The system supports gigapixel-scale images at multiple magnification levels.
        </p>
      </div>
    ),
    placement: "right" as const,
    disableBeacon: true,
    data: {
      icon: <Microscope className="w-6 h-6 text-white" />,
      features: [
        "Multi-gigapixel WSI support",
        "Multiple magnification levels",
        "Patient cohort organization",
      ],
    },
  },
  {
    target: '[data-demo="slide-selector"] [data-demo="analyze-button"]',
    title: "Run AI Analysis",
    content: (
      <div className="space-y-2">
        <p>
          Click <strong>Analyze Slide</strong> to run our MedGemma-powered pathology AI. 
          The system will process thousands of tissue patches to predict treatment response.
        </p>
        <p className="text-amber-600 dark:text-amber-300 text-sm">
          ⚡ Analysis typically completes in 10–30 seconds.
        </p>
      </div>
    ),
    placement: "right" as const,
    disableBeacon: true,
    data: {
      icon: <Brain className="w-6 h-6 text-white" />,
      features: [
        "MedGemma vision model",
        "8,000+ patches analyzed",
        "Real-time progress tracking",
      ],
    },
  },
  {
    target: '[data-demo="wsi-viewer"]',
    title: "Interactive WSI Viewer",
    content: (
      <div className="space-y-2">
        <p>
          Explore the whole-slide image with smooth pan and zoom controls. 
          The AI-generated heatmap overlay shows regions of high diagnostic significance.
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Toggle heatmap visibility, zoom to evidence patches, or enter fullscreen mode.
        </p>
      </div>
    ),
    placement: "bottom" as const,
    disableBeacon: true,
    data: {
      icon: <Target className="w-6 h-6 text-white" />,
      features: [
        "Smooth zoomable interface",
        "AI attention heatmap overlay",
        "Evidence patch highlighting",
      ],
    },
  },
  {
    target: '[data-demo="right-tab-prediction"]',
    title: "Treatment Response Prediction",
    content: (
      <div className="space-y-2">
        <p>
          View the AI&apos;s prediction with confidence scores. The model classifies patients
          into treatment response categories with associated confidence levels.
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Includes quality metrics and uncertainty quantification.
        </p>
      </div>
    ),
    placement: "left" as const,
    disableBeacon: true,
    data: {
      icon: <BarChart3 className="w-6 h-6 text-white" />,
      features: [
        "Binary classification",
        "Confidence calibration",
        "Slide quality assessment",
      ],
    },
  },
  {
    target: '[data-demo="right-tab-semantic-search"]',
    title: "Semantic Search",
    content: (
      <div className="space-y-2">
        <p>
          Use MedSigLIP-powered semantic search to locate morphologically related tissue patterns.
          This helps you quickly discover regions and slides that match a pathology concept.
        </p>
        <p className="text-clinical-600 dark:text-clinical-300 font-medium">
          Search results can be inspected and compared directly in the workflow.
        </p>
      </div>
    ),
    placement: "left" as const,
    disableBeacon: true,
    data: {
      icon: <Layers className="w-6 h-6 text-white" />,
      features: [
        "MedSigLIP semantic retrieval",
        "Patch-level similarity",
        "Fast concept-driven exploration",
      ],
    },
  },
  {
    target: '[data-demo="right-tab-similar-cases"]',
    title: "Similar Historical Cases",
    content: (
      <div className="space-y-2">
        <p>
          The AI retrieves morphologically similar cases from the database, 
          showing their outcomes to provide clinical context and validation.
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Click any case to view that patient&apos;s slide and analysis.
        </p>
      </div>
    ),
    placement: "left" as const,
    disableBeacon: true,
    data: {
      icon: <Sparkles className="w-6 h-6 text-white" />,
      features: [
        "Embedding-based retrieval",
        "Outcome correlation",
        "One-click navigation",
      ],
    },
  },
  {
    target: '[data-demo="right-tab-medgemma"]',
    title: "Clinical Report Generation",
    content: (
      <div className="space-y-2">
        <p>
          Generate a comprehensive clinical report summarizing the AI analysis, 
          evidence patches, and recommendations. Export to PDF for clinical workflows.
        </p>
        <p className="text-green-600 dark:text-green-300 font-medium">
          Reports are structured for easy integration with EMR systems.
        </p>
      </div>
    ),
    placement: "left" as const,
    disableBeacon: true,
    data: {
      icon: <FileText className="w-6 h-6 text-white" />,
      features: [
        "Structured clinical format",
        "PDF/JSON export",
        "Evidence documentation",
      ],
    },
  },
];

const STARTUP_READY_CHECK_INTERVAL_MS = 120;
const STARTUP_MAX_READY_CHECKS = 18;
const TARGET_RETRY_INTERVAL_MS = 250;
const MAX_TARGET_RETRY_ATTEMPTS = 8;
/** After exhausting primary retries, slow down before trying fallback cycle again. */
const FALLBACK_RETRY_INTERVAL_MS = 500;

/**
 * Fallback selectors by step index. When the primary target for a step is not
 * found, we try these in order before giving up. This prevents skipping steps
 * due to transient DOM flicker during tab remount/scroll.
 */
export const STEP_FALLBACK_TARGETS: Partial<Record<number, string[]>> = {
  3: ['[data-demo="right-tablist"]'],
  4: ['[data-demo="right-tablist"]'],
  5: ['[data-demo="right-tablist"]'],
  6: ['[data-demo="right-tablist"]'],
};

export function getStepSelector(stepIndex: number, steps: Step[] = tourSteps): string {
  const step = steps[stepIndex];
  return typeof step?.target === "string" ? step.target : "";
}

export function isTargetVisible(selector: string): boolean {
  if (!selector) return false;

  const element = document.querySelector(selector);
  if (!element) return false;

  const rect = element.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return false;

  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden";
}

/**
 * Resolve a visible selector for the given step — tries the primary target
 * first, then any configured fallbacks.
 */
export function resolveStepTarget(
  stepIndex: number,
  steps: Step[] = tourSteps
): string | null {
  const primary = getStepSelector(stepIndex, steps);
  if (primary && isTargetVisible(primary)) return primary;

  const fallbacks = STEP_FALLBACK_TARGETS[stepIndex];
  if (fallbacks) {
    for (const fb of fallbacks) {
      if (isTargetVisible(fb)) return fb;
    }
  }

  return null;
}

function clampStep(step: number): number {
  return Math.max(0, Math.min(step, tourSteps.length - 1));
}

export { tourSteps };

export function DemoMode({ isActive, onClose, onStepChange }: DemoModeProps) {
  const [stepIndex, setStepIndex] = useState(0);
  const [run, setRun] = useState(false);

  // Mutable copy of steps so fallback targets can be patched at runtime
  const stepsRef = React.useRef<Step[]>(tourSteps.map((s) => ({ ...s })));

  const startupTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const stepIndexRef = React.useRef(0);
  const targetRetryRef = React.useRef<{
    token: number;
    step: number | null;
    attempts: number;
    timer: ReturnType<typeof setTimeout> | null;
  }>({
    token: 0,
    step: null,
    attempts: 0,
    timer: null,
  });

  const clearStartupTimer = useCallback(() => {
    if (startupTimerRef.current) {
      clearTimeout(startupTimerRef.current);
      startupTimerRef.current = null;
    }
  }, []);

  const resetTargetRetry = useCallback(() => {
    if (targetRetryRef.current.timer) {
      clearTimeout(targetRetryRef.current.timer);
    }

    targetRetryRef.current = {
      token: targetRetryRef.current.token + 1,
      step: null,
      attempts: 0,
      timer: null,
    };
  }, []);

  const setTourStep = useCallback((nextStep: number) => {
    const bounded = clampStep(nextStep);
    stepIndexRef.current = bounded;
    setStepIndex(bounded);
  }, []);

  const retriggerCurrentStep = useCallback(
    (expectedStep: number) => {
      setRun(false);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (!isActive) return;
          if (stepIndexRef.current !== expectedStep) return;
          setRun(true);
        });
      });
    },
    [isActive]
  );

  const scheduleMissingTargetRetry = useCallback(
    (missingStep: number) => {
      const activeRetry = targetRetryRef.current;
      if (activeRetry.step === missingStep && activeRetry.timer) {
        return;
      }

      resetTargetRetry();
      targetRetryRef.current.step = missingStep;
      const token = targetRetryRef.current.token;

      const tryResolveTarget = () => {
        const retryState = targetRetryRef.current;
        if (retryState.token !== token || !isActive) return;

        if (stepIndexRef.current !== missingStep) {
          resetTargetRetry();
          return;
        }

        // Re-signal the host so it can ensure the right panel/tab is visible
        onStepChange?.(missingStep);

        // Check primary target first
        const selector = getStepSelector(missingStep);
        if (selector && isTargetVisible(selector)) {
          resetTargetRetry();
          retriggerCurrentStep(missingStep);
          return;
        }

        // Check fallback targets
        const fallbacks = STEP_FALLBACK_TARGETS[missingStep];
        if (fallbacks) {
          for (const fb of fallbacks) {
            if (isTargetVisible(fb)) {
              // Re-point the live step target to the fallback so Joyride anchors there
              stepsRef.current[missingStep] = {
                ...stepsRef.current[missingStep],
                target: fb,
              };
              resetTargetRetry();
              retriggerCurrentStep(missingStep);
              return;
            }
          }
        }

        if (retryState.attempts >= MAX_TARGET_RETRY_ATTEMPTS) {
          // ── NEVER auto-advance to the next step ──
          // Reset the attempt counter and keep retrying at a slower cadence.
          // This prevents the tour from skipping steps due to transient
          // target misses (tab remount, scroll, React re-render batching).
          retryState.attempts = 0;
          retryState.timer = setTimeout(tryResolveTarget, FALLBACK_RETRY_INTERVAL_MS);
          return;
        }

        retryState.attempts += 1;
        retryState.timer = setTimeout(tryResolveTarget, TARGET_RETRY_INTERVAL_MS);
      };

      targetRetryRef.current.timer = setTimeout(
        tryResolveTarget,
        TARGET_RETRY_INTERVAL_MS
      );
    },
    [isActive, onStepChange, resetTargetRetry, retriggerCurrentStep]
  );

  useEffect(() => {
    stepIndexRef.current = stepIndex;
  }, [stepIndex]);

  useEffect(() => {
    if (!isActive) {
      clearStartupTimer();
      resetTargetRetry();
      setRun(false);
      return;
    }

    clearStartupTimer();
    resetTargetRetry();

    // Reset the mutable steps to pristine copies so previous fallback patches
    // don't persist across demo restarts.
    stepsRef.current = tourSteps.map((s) => ({ ...s }));

    setRun(false);
    setStepIndex(0);
    stepIndexRef.current = 0;
    onStepChange?.(0);

    let checks = 0;
    const waitForFirstTarget = () => {
      const firstStepSelector = getStepSelector(0);
      const firstTargetReady = firstStepSelector
        ? isTargetVisible(firstStepSelector)
        : true;

      if (firstTargetReady || checks >= STARTUP_MAX_READY_CHECKS) {
        setRun(true);
        return;
      }

      checks += 1;
      startupTimerRef.current = setTimeout(
        waitForFirstTarget,
        STARTUP_READY_CHECK_INTERVAL_MS
      );
    };

    startupTimerRef.current = setTimeout(waitForFirstTarget, 0);

    return () => {
      clearStartupTimer();
      resetTargetRetry();
    };
  }, [isActive, clearStartupTimer, resetTargetRetry, onStepChange]);

  const handleJoyrideCallback = useCallback(
    (data: CallBackProps) => {
      const { status, type, index, action } = data;
      const currentIndex = clampStep(index ?? 0);

      if (type === EVENTS.STEP_BEFORE) {
        onStepChange?.(currentIndex);
      }

      if (type === EVENTS.STEP_AFTER) {
        resetTargetRetry();

        // Defer step transitions until after Joyride settles the previous tooltip/spotlight
        // to avoid transient null-target errors during fast panel remounts.
        if (action === ACTIONS.NEXT) {
          const nextStep = currentIndex + 1;

          // In controlled mode, avoid clamping the last "Next" back onto the final step.
          // Close immediately so "Get Started" completes reliably.
          if (nextStep >= tourSteps.length) {
            clearStartupTimer();
            resetTargetRetry();
            setRun(false);
            onClose();
            return;
          }

          // Prime host UI state for the next step before Joyride evaluates the new target.
          onStepChange?.(nextStep);
          requestAnimationFrame(() => setTourStep(currentIndex + 1));
        } else if (action === ACTIONS.PREV) {
          const prevStep = currentIndex - 1;
          onStepChange?.(prevStep);
          requestAnimationFrame(() => setTourStep(currentIndex - 1));
        }
      }

      if (type === EVENTS.TARGET_NOT_FOUND) {
        onStepChange?.(currentIndex);
        scheduleMissingTargetRetry(currentIndex);
      }

      if (
        status === STATUS.FINISHED ||
        status === STATUS.SKIPPED ||
        action === ACTIONS.CLOSE
      ) {
        clearStartupTimer();
        resetTargetRetry();
        setRun(false);
        onClose();
      }
    },
    [
      clearStartupTimer,
      onClose,
      onStepChange,
      resetTargetRetry,
      scheduleMissingTargetRetry,
      setTourStep,
    ]
  );

  if (!isActive) return null;

  const isDarkTheme =
    typeof document !== "undefined" &&
    document.documentElement.classList.contains("dark");

  return (
    <Joyride
      steps={stepsRef.current}
      run={run}
      stepIndex={stepIndex}
      continuous
      showSkipButton
      showProgress
      disableScrolling
      disableScrollParentFix
      spotlightClicks
      disableOverlay
      disableOverlayClose
      callback={handleJoyrideCallback}
      tooltipComponent={CustomTooltip}
      floaterProps={{
        disableAnimation: true,
        offset: 16,
      }}
      styles={{
        options: {
          zIndex: 10000,
          arrowColor: isDarkTheme ? "#1e293b" : "#fff",
          backgroundColor: isDarkTheme ? "#1e293b" : "#fff",
          overlayColor: "rgba(15, 23, 42, 0.4)",
          primaryColor: "#0ea5e9",
          spotlightShadow: "0 0 0 3px rgba(14, 165, 233, 0.5), 0 0 20px rgba(14, 165, 233, 0.2)",
        },
        spotlight: {
          borderRadius: 12,
          backgroundColor: "transparent",
        },
      }}
      locale={{
        back: "Back",
        close: "Close",
        last: "Get Started",
        next: "Next",
        skip: "Skip tour",
      }}
    />
  );
}

// Demo mode toggle button for the header
interface DemoToggleProps {
  isActive: boolean;
  onToggle: () => void;
  /** When true, the toggle appears greyed/slashed and is non-interactive (e.g., settings modal open) */
  disabled?: boolean;
}

export function DemoToggle({ isActive, onToggle, disabled = false }: DemoToggleProps) {
  return (
    <button
      onClick={disabled ? undefined : onToggle}
      disabled={disabled}
      aria-disabled={disabled}
      className={cn(
        "relative flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
        disabled
          ? "bg-gray-300 text-gray-500 cursor-not-allowed opacity-60 shadow-none"
          : isActive
            ? "bg-clinical-500 text-white shadow-lg shadow-clinical-500/30"
            : "bg-gradient-to-r from-clinical-500 to-clinical-600 text-white hover:from-clinical-600 hover:to-clinical-700 shadow-md"
      )}
      title={disabled ? "Demo mode unavailable while settings is open" : isActive ? "Exit demo mode" : "Start guided tour"}
    >
      {/* Diagonal strike-through overlay when disabled */}
      {disabled && (
        <span
          aria-hidden="true"
          className="absolute inset-0 pointer-events-none overflow-hidden rounded-lg"
        >
          <span className="absolute top-1/2 left-0 w-full h-0.5 bg-gray-500/70 -rotate-12 origin-center" />
        </span>
      )}
      {isActive ? (
        <>
          <X className="w-4 h-4" />
          <span className="hidden sm:inline">Exit Demo</span>
        </>
      ) : (
        <>
          <Play className="w-4 h-4" />
          <span className="hidden sm:inline">Demo Mode</span>
        </>
      )}
    </button>
  );
}

// Welcome modal for first-time users
interface WelcomeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStartDemo: () => void;
}

export function WelcomeModal({ isOpen, onClose, onStartDemo }: WelcomeModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-navy-900/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white dark:bg-navy-800 border border-gray-100 dark:border-navy-600 rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden animate-scale-in">
        {/* Gradient header */}
        <div className="bg-gradient-to-r from-clinical-500 via-clinical-600 to-violet-600 p-8 text-white">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-14 h-14 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
              <Microscope className="w-8 h-8" />
            </div>
            <div>
              <h2 className="text-2xl font-bold">Welcome to Enso Atlas</h2>
              <p className="text-clinical-100 text-sm">
                Pathology Evidence Engine
              </p>
            </div>
          </div>
          <p className="text-white/90 leading-relaxed">
            An AI-powered platform for predicting cancer treatment response from 
            histopathology images using Google&apos;s MedGemma.
          </p>
        </div>

        {/* Features */}
        <div className="p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {[
              { icon: Brain, label: "MedGemma AI", desc: "Vision-language model" },
              { icon: Microscope, label: "WSI Analysis", desc: "Gigapixel images" },
              { icon: Layers, label: "Evidence Maps", desc: "Explainable AI" },
              { icon: FileText, label: "Clinical Reports", desc: "PDF export" },
            ].map((feature, i) => (
              <div
                key={i}
                className="flex items-start gap-3 p-3 rounded-lg bg-gray-50 dark:bg-navy-700/60"
              >
                <div className="w-10 h-10 rounded-lg bg-clinical-100 dark:bg-clinical-900/40 flex items-center justify-center shrink-0">
                  <feature.icon className="w-5 h-5 text-clinical-600 dark:text-clinical-300" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-gray-100 text-sm">
                    {feature.label}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">{feature.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 pb-6 flex items-center justify-between">
          <button
            onClick={onClose}
            className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 text-sm transition-colors"
          >
            Skip intro
          </button>
          <button
            onClick={() => {
              onClose();
              onStartDemo();
            }}
            className={cn(
              "flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all",
              "bg-gradient-to-r from-clinical-500 to-clinical-600 text-white",
              "hover:from-clinical-600 hover:to-clinical-700 shadow-lg hover:shadow-xl"
            )}
          >
            <Play className="w-5 h-5" />
            Start Guided Tour
          </button>
        </div>
      </div>
    </div>
  );
}
