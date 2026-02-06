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
      className="bg-white rounded-2xl shadow-2xl border border-gray-100 max-w-md overflow-hidden animate-scale-in"
    >
      {/* Progress bar */}
      <div className="h-1 bg-gray-100">
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
              <span className="text-xs font-semibold text-clinical-600 bg-clinical-50 px-2 py-0.5 rounded-full">
                Step {index + 1} of {size}
              </span>
            </div>
            {step.title && (
              <h3 className="text-lg font-bold text-gray-900 leading-tight">
                {step.title}
              </h3>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 pb-4">
        <div className="text-gray-600 text-sm leading-relaxed">
          {step.content}
        </div>
      </div>

      {/* Feature highlights if present */}
      {(step.data as { features?: string[] })?.features && (
        <div className="px-6 pb-4">
          <div className="bg-gray-50 rounded-lg p-3 space-y-2">
            {(step.data as { features: string[] }).features.map((feature, i) => (
              <div key={i} className="flex items-center gap-2 text-sm text-gray-700">
                <div className="w-5 h-5 rounded-full bg-clinical-100 flex items-center justify-center shrink-0">
                  <Zap className="w-3 h-3 text-clinical-600" />
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
          className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
        >
          Skip tour
        </button>
        <div className="flex items-center gap-2">
          {index > 0 && (
            <button
              {...backProps}
              className="flex items-center gap-1 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
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
        <p className="text-clinical-600 font-medium">
          The system supports gigapixel-scale images at multiple magnification levels.
        </p>
      </div>
    ),
    placement: "right" as const,
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
    target: '[data-demo="analyze-button"]',
    title: "Run AI Analysis",
    content: (
      <div className="space-y-2">
        <p>
          Click <strong>Analyze Slide</strong> to run our MedGemma-powered pathology AI. 
          The system will process thousands of tissue patches to predict treatment response.
        </p>
        <p className="text-amber-600 text-sm">
          ⚡ Analysis typically completes in 10–30 seconds.
        </p>
      </div>
    ),
    placement: "right" as const,
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
        <p className="text-sm text-gray-500">
          Toggle heatmap visibility, zoom to evidence patches, or enter fullscreen mode.
        </p>
      </div>
    ),
    placement: "left" as const,
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
    target: '[data-demo="prediction-panel"]',
    title: "Treatment Response Prediction",
    content: (
      <div className="space-y-2">
        <p>
          View the AI&apos;s prediction with confidence scores. The model classifies patients 
          as <span className="text-green-600 font-semibold">Sensitive</span> or{" "}
          <span className="text-red-600 font-semibold">Resistant</span> to platinum-based chemotherapy.
        </p>
        <p className="text-sm text-gray-500">
          Includes quality metrics and uncertainty quantification.
        </p>
      </div>
    ),
    placement: "left" as const,
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
    target: '[data-demo="evidence-panel"]',
    title: "Evidence Patches",
    content: (
      <div className="space-y-2">
        <p>
          Examine the most influential tissue regions that drove the AI&apos;s decision. 
          Each patch shows its contribution score and can be zoomed for detailed inspection.
        </p>
        <p className="text-clinical-600 font-medium">
          Click any patch to navigate the viewer directly to that region.
        </p>
      </div>
    ),
    placement: "left" as const,
    data: {
      icon: <Layers className="w-6 h-6 text-white" />,
      features: [
        "Top contributing regions",
        "Attention scores",
        "Click-to-navigate",
      ],
    },
  },
  {
    target: '[data-demo="similar-cases"]',
    title: "Similar Historical Cases",
    content: (
      <div className="space-y-2">
        <p>
          The AI retrieves morphologically similar cases from the database, 
          showing their outcomes to provide clinical context and validation.
        </p>
        <p className="text-sm text-gray-500">
          Click any case to view that patient&apos;s slide and analysis.
        </p>
      </div>
    ),
    placement: "left" as const,
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
    target: '[data-demo="report-panel"]',
    title: "Clinical Report Generation",
    content: (
      <div className="space-y-2">
        <p>
          Generate a comprehensive clinical report summarizing the AI analysis, 
          evidence patches, and recommendations. Export to PDF for clinical workflows.
        </p>
        <p className="text-green-600 font-medium">
          Reports are structured for easy integration with EMR systems.
        </p>
      </div>
    ),
    placement: "left" as const,
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

export function DemoMode({ isActive, onClose, onStepChange }: DemoModeProps) {
  const [stepIndex, setStepIndex] = useState(0);
  const [run, setRun] = useState(false);

  useEffect(() => {
    if (isActive) {
      setStepIndex(0);
      setRun(true);
    } else {
      setRun(false);
    }
  }, [isActive]);

  const handleJoyrideCallback = useCallback(
    (data: CallBackProps) => {
      const { status, type, index, action } = data;

      if (type === EVENTS.STEP_AFTER) {
        if (action === ACTIONS.NEXT) {
          setStepIndex(index + 1);
          onStepChange?.(index + 1);
        } else if (action === ACTIONS.PREV) {
          setStepIndex(index - 1);
          onStepChange?.(index - 1);
        }
      }

      if (
        status === STATUS.FINISHED ||
        status === STATUS.SKIPPED ||
        action === ACTIONS.CLOSE
      ) {
        setRun(false);
        onClose();
      }
    },
    [onClose, onStepChange]
  );

  if (!isActive) return null;

  return (
    <Joyride
      steps={tourSteps}
      run={run}
      stepIndex={stepIndex}
      continuous
      showSkipButton
      showProgress
      scrollToFirstStep
      spotlightClicks
      disableOverlayClose
      callback={handleJoyrideCallback}
      tooltipComponent={CustomTooltip}
      floaterProps={{
        disableAnimation: false,
      }}
      styles={{
        options: {
          zIndex: 10000,
          arrowColor: "#fff",
          backgroundColor: "#fff",
          overlayColor: "rgba(15, 23, 42, 0.75)",
          primaryColor: "#0ea5e9",
          spotlightShadow: "0 0 20px rgba(14, 165, 233, 0.5)",
        },
        spotlight: {
          borderRadius: 12,
        },
        overlay: {
          mixBlendMode: undefined,
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
}

export function DemoToggle({ isActive, onToggle }: DemoToggleProps) {
  return (
    <button
      onClick={onToggle}
      className={cn(
        "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
        isActive
          ? "bg-clinical-500 text-white shadow-lg shadow-clinical-500/30"
          : "bg-gradient-to-r from-clinical-500 to-clinical-600 text-white hover:from-clinical-600 hover:to-clinical-700 shadow-md"
      )}
      title={isActive ? "Exit demo mode" : "Start guided tour"}
    >
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
      <div className="relative bg-white rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden animate-scale-in">
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
                className="flex items-start gap-3 p-3 rounded-lg bg-gray-50"
              >
                <div className="w-10 h-10 rounded-lg bg-clinical-100 flex items-center justify-center shrink-0">
                  <feature.icon className="w-5 h-5 text-clinical-600" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900 text-sm">
                    {feature.label}
                  </div>
                  <div className="text-xs text-gray-500">{feature.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 pb-6 flex items-center justify-between">
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-sm transition-colors"
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
