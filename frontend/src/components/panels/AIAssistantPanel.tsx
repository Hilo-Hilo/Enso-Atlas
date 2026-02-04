"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import {
  Bot,
  Brain,
  FileText,
  Loader2,
  Search,
  Send,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Clock,
  ChevronDown,
  ChevronUp,
  MessageSquare,
  Lightbulb,
  Target,
  Activity,
  Copy,
  Download,
  Eye,
  MapPin,
} from "lucide-react";
import { cn } from "@/lib/utils";

// API base URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://100.111.126.23:8003";

// Types for agent workflow
interface AgentStepData {
  step: string;
  status: "pending" | "running" | "complete" | "skipped" | "error";
  message: string;
  reasoning?: string;
  data?: Record<string, unknown>;
  timestamp?: string;
  duration_ms?: number;
}

interface AgentPrediction {
  model_name: string;
  label: string;
  score: number;
  confidence: number;
}

interface EvidencePatch {
  rank: number;
  patch_index: number;
  attention_weight: number;
  coordinates?: [number, number];
}

interface SimilarCase {
  slide_id: string;
  similarity_score: number;
  label?: string;
}

interface AIAssistantPanelProps {
  slideId: string | null;
  clinicalContext?: string;
  onAnalysisComplete?: (report: Record<string, unknown>) => void;
  onHighlightRegion?: (x: number, y: number, weight: number) => void;
  className?: string;
}

// Step icon mapping
const STEP_ICONS: Record<string, React.ReactNode> = {
  initialize: <Activity className="h-4 w-4" />,
  analyze: <Brain className="h-4 w-4" />,
  retrieve: <Search className="h-4 w-4" />,
  compare: <Target className="h-4 w-4" />,
  reason: <Lightbulb className="h-4 w-4" />,
  report: <FileText className="h-4 w-4" />,
  complete: <CheckCircle2 className="h-4 w-4" />,
  error: <AlertCircle className="h-4 w-4" />,
};

// Step display names
const STEP_NAMES: Record<string, string> = {
  initialize: "Loading Data",
  analyze: "Running Models",
  retrieve: "Finding Similar Cases",
  compare: "Comparing Cases",
  reason: "Generating Reasoning",
  report: "Creating Report",
  complete: "Analysis Complete",
  error: "Error",
};

// Evidence patch thumbnail component
function EvidencePatchCard({
  patch,
  slideId,
  onClick,
}: {
  patch: EvidencePatch;
  slideId: string;
  onClick?: () => void;
}) {
  const coords = patch.coordinates || [0, 0];
  const intensity = Math.min(1, patch.attention_weight * 3);
  
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative flex flex-col items-center p-2 rounded-lg border transition-all",
        "hover:border-indigo-400 hover:shadow-md",
        "bg-gradient-to-br from-indigo-50 to-purple-50"
      )}
      title={`Patch #${patch.patch_index} - Attention: ${(patch.attention_weight * 100).toFixed(1)}%`}
    >
      <div 
        className="w-14 h-14 rounded bg-gray-200 flex items-center justify-center relative overflow-hidden"
        style={{
          boxShadow: `inset 0 0 0 2px rgba(99, 102, 241, ${intensity})`,
        }}
      >
        <MapPin 
          className="h-6 w-6 text-indigo-500" 
          style={{ opacity: 0.3 + intensity * 0.7 }}
        />
        <div 
          className="absolute inset-0 bg-indigo-500"
          style={{ opacity: intensity * 0.2 }}
        />
      </div>
      <div className="text-xs mt-1 font-medium text-gray-600">
        #{patch.rank}
      </div>
      <div className="text-xs text-gray-400">
        {(patch.attention_weight * 100).toFixed(0)}%
      </div>
    </button>
  );
}

// Workflow step component
function WorkflowStep({
  step,
  isExpanded,
  onToggle,
  slideId,
  onHighlightRegion,
}: {
  step: AgentStepData;
  isExpanded: boolean;
  onToggle: () => void;
  slideId?: string;
  onHighlightRegion?: (x: number, y: number, weight: number) => void;
}) {
  const statusColors: Record<string, string> = {
    pending: "text-gray-400",
    running: "text-blue-500 animate-pulse",
    complete: "text-green-500",
    skipped: "text-yellow-500",
    error: "text-red-500",
  };

  const statusBg: Record<string, string> = {
    pending: "bg-gray-100",
    running: "bg-blue-50 border-blue-200",
    complete: "bg-green-50 border-green-200",
    skipped: "bg-yellow-50 border-yellow-200",
    error: "bg-red-50 border-red-200",
  };
  
  const topEvidence = (step.data?.top_evidence as EvidencePatch[]) || [];

  return (
    <div className={cn("border rounded-lg transition-all", statusBg[step.status])}>
      <button
        onClick={onToggle}
        className="w-full p-3 flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-3">
          <span className={statusColors[step.status]}>
            {step.status === "running" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              STEP_ICONS[step.step] || <Activity className="h-4 w-4" />
            )}
          </span>
          <div>
            <div className="font-medium text-sm">
              {STEP_NAMES[step.step] || step.step}
            </div>
            <div className="text-xs text-gray-500">{step.message}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {step.duration_ms !== undefined && step.duration_ms > 0 && (
            <span className="text-xs text-gray-400">
              {(step.duration_ms / 1000).toFixed(1)}s
            </span>
          )}
          {(step.reasoning || topEvidence.length > 0) && (
            isExpanded ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )
          )}
        </div>
      </button>
      
      {isExpanded && (step.reasoning || step.data) && (
        <div className="px-3 pb-3 pt-1 border-t border-gray-200/50 space-y-3">
          {/* Reasoning */}
          {step.reasoning && (
            <div className="text-xs text-gray-600 whitespace-pre-wrap font-mono bg-white/50 rounded p-2">
              {step.reasoning}
            </div>
          )}
          
          {/* Evidence Patches */}
          {topEvidence.length > 0 && slideId && (
            <div>
              <div className="text-xs font-medium text-gray-500 mb-2 flex items-center gap-1">
                <Eye className="h-3 w-3" />
                High Attention Regions
              </div>
              <div className="flex flex-wrap gap-2">
                {topEvidence.slice(0, 5).map((patch, i) => (
                  <EvidencePatchCard
                    key={i}
                    patch={patch}
                    slideId={slideId}
                    onClick={() => {
                      if (onHighlightRegion && patch.coordinates) {
                        onHighlightRegion(
                          patch.coordinates[0],
                          patch.coordinates[1],
                          patch.attention_weight
                        );
                      }
                    }}
                  />
                ))}
              </div>
            </div>
          )}
          
          {/* Predictions */}
          {step.data?.predictions && (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(step.data.predictions as Record<string, AgentPrediction>).map(([id, pred]) => (
                <div key={id} className="bg-white rounded p-2 border">
                  <div className="font-medium text-xs truncate">{pred.model_name}</div>
                  <div className={cn(
                    "text-sm font-semibold",
                    pred.label === "responder" || pred.label === "positive" 
                      ? "text-green-600" 
                      : "text-orange-600"
                  )}>
                    {pred.label} ({(pred.score * 100).toFixed(0)}%)
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* Similar Cases */}
          {step.data?.similar_cases && (
            <div>
              <div className="text-xs font-medium text-gray-500 mb-1">Similar Cases:</div>
              <div className="space-y-1">
                {(step.data.similar_cases as SimilarCase[]).slice(0, 3).map((c, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-gray-600">
                    <span className="truncate font-mono">{c.slide_id}</span>
                    <span className={cn(
                      "px-1.5 py-0.5 rounded text-xs",
                      c.label === "responder" ? "bg-green-100 text-green-700" : "bg-orange-100 text-orange-700"
                    )}>
                      {c.label || "unknown"}
                    </span>
                    <span className="text-gray-400">
                      {(c.similarity_score * 100).toFixed(0)}% match
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Chat message component with evidence support
function ChatMessage({
  message,
  isUser,
  timestamp,
  evidencePatches,
  onHighlightRegion,
}: {
  message: string;
  isUser: boolean;
  timestamp?: string;
  evidencePatches?: EvidencePatch[];
  onHighlightRegion?: (x: number, y: number, weight: number) => void;
}) {
  return (
    <div className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
          <Bot className="h-4 w-4 text-indigo-600" />
        </div>
      )}
      <div className={cn("max-w-[85%]", isUser ? "" : "")}>
        <div
          className={cn(
            "rounded-lg px-4 py-2",
            isUser
              ? "bg-indigo-600 text-white"
              : "bg-gray-100 text-gray-800"
          )}
        >
          <div className="text-sm whitespace-pre-wrap">{message}</div>
        </div>
        
        {/* Evidence patches in response */}
        {!isUser && evidencePatches && evidencePatches.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {evidencePatches.slice(0, 4).map((patch, i) => (
              <button
                key={i}
                onClick={() => {
                  if (onHighlightRegion && patch.coordinates) {
                    onHighlightRegion(
                      patch.coordinates[0],
                      patch.coordinates[1],
                      patch.attention_weight
                    );
                  }
                }}
                className="px-2 py-1 text-xs bg-indigo-50 text-indigo-600 rounded-full hover:bg-indigo-100 flex items-center gap-1"
              >
                <MapPin className="h-3 w-3" />
                Region #{patch.rank}
              </button>
            ))}
          </div>
        )}
        
        {timestamp && (
          <div className={cn(
            "text-xs mt-1",
            isUser ? "text-right text-gray-400" : "text-gray-400"
          )}>
            {new Date(timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center flex-shrink-0">
          <MessageSquare className="h-4 w-4 text-white" />
        </div>
      )}
    </div>
  );
}

export function AIAssistantPanel({
  slideId,
  clinicalContext = "",
  onAnalysisComplete,
  onHighlightRegion,
  className,
}: AIAssistantPanelProps) {
  // State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [steps, setSteps] = useState<AgentStepData[]>([]);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [chatMessages, setChatMessages] = useState<Array<{
    message: string;
    isUser: boolean;
    timestamp: string;
    evidencePatches?: EvidencePatch[];
  }>>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isFollowupLoading, setIsFollowupLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [topEvidence, setTopEvidence] = useState<EvidencePatch[]>([]);
  const [copySuccess, setCopySuccess] = useState(false);
  
  // Refs
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages, steps]);

  // Start agent analysis
  const startAnalysis = useCallback(async () => {
    if (!slideId) return;
    
    setIsAnalyzing(true);
    setError(null);
    setSteps([]);
    setSessionId(null);
    setReport(null);
    setChatMessages([]);
    setExpandedSteps(new Set());
    setTopEvidence([]);
    
    try {
      const response = await fetch(`${API_BASE}/api/agent/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slide_id: slideId,
          clinical_context: clinicalContext,
          questions: [],
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      
      // Process SSE stream
      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");
      
      const decoder = new TextDecoder();
      let buffer = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6)) as AgentStepData;
              
              // Update steps
              setSteps((prev) => {
                const existing = prev.findIndex((s) => s.step === data.step);
                if (existing >= 0) {
                  const updated = [...prev];
                  updated[existing] = data;
                  return updated;
                }
                return [...prev, data];
              });
              
              // Auto-expand running and completed steps with reasoning
              if (data.reasoning && (data.status === "running" || data.status === "complete")) {
                setExpandedSteps((prev) => new Set(Array.from(prev).concat(data.step)));
              }
              
              // Extract session ID
              if (data.data?.session_id) {
                setSessionId(data.data.session_id as string);
              }
              
              // Extract top evidence patches
              if (data.data?.top_evidence) {
                setTopEvidence(data.data.top_evidence as EvidencePatch[]);
              }
              
              // Extract report on completion
              if (data.step === "report" && data.status === "complete" && data.data?.report) {
                setReport(data.data.report as Record<string, unknown>);
                onAnalysisComplete?.(data.data.report as Record<string, unknown>);
              }
              
              // Handle errors
              if (data.status === "error") {
                setError(data.message);
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  }, [slideId, clinicalContext, onAnalysisComplete]);
  
  // Send follow-up question
  const sendFollowup = useCallback(async () => {
    if (!sessionId || !inputMessage.trim()) return;
    
    const question = inputMessage.trim();
    setInputMessage("");
    setIsFollowupLoading(true);
    
    // Add user message
    setChatMessages((prev) => [
      ...prev,
      { message: question, isUser: true, timestamp: new Date().toISOString() },
    ]);
    
    try {
      const response = await fetch(`${API_BASE}/api/agent/followup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Follow-up failed: ${response.statusText}`);
      }
      
      // Process SSE stream
      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");
      
      const decoder = new TextDecoder();
      let buffer = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.reasoning) {
                // Check if response mentions regions/attention
                const mentionsRegions = /region|attention|area|patch/i.test(data.reasoning);
                
                setChatMessages((prev) => [
                  ...prev,
                  {
                    message: data.reasoning,
                    isUser: false,
                    timestamp: data.timestamp || new Date().toISOString(),
                    evidencePatches: mentionsRegions ? topEvidence.slice(0, 4) : undefined,
                  },
                ]);
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (e) {
      setChatMessages((prev) => [
        ...prev,
        {
          message: `Error: ${e instanceof Error ? e.message : "Failed to get response"}`,
          isUser: false,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsFollowupLoading(false);
      inputRef.current?.focus();
    }
  }, [sessionId, inputMessage, topEvidence]);
  
  // Handle Enter key in input
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendFollowup();
      }
    },
    [sendFollowup]
  );
  
  // Toggle step expansion
  const toggleStep = useCallback((stepName: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepName)) {
        next.delete(stepName);
      } else {
        next.add(stepName);
      }
      return next;
    });
  }, []);
  
  // Copy conversation to clipboard
  const copyConversation = useCallback(async () => {
    const lines: string[] = [];
    lines.push(`AI Analysis - Slide ${slideId}`);
    lines.push(`Generated: ${new Date().toISOString()}`);
    lines.push("");
    
    // Add step summaries
    steps.forEach(step => {
      lines.push(`[${STEP_NAMES[step.step]}] ${step.message}`);
      if (step.reasoning) {
        lines.push(step.reasoning);
      }
      lines.push("");
    });
    
    // Add chat messages
    if (chatMessages.length > 0) {
      lines.push("--- Follow-up Questions ---");
      chatMessages.forEach(msg => {
        lines.push(`${msg.isUser ? "You" : "AI"}: ${msg.message}`);
      });
    }
    
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  }, [slideId, steps, chatMessages]);
  
  // Export as JSON
  const exportConversation = useCallback(() => {
    const data = {
      slideId,
      sessionId,
      timestamp: new Date().toISOString(),
      steps: steps.map(s => ({
        step: s.step,
        status: s.status,
        message: s.message,
        reasoning: s.reasoning,
      })),
      report,
      conversation: chatMessages,
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-${slideId}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [slideId, sessionId, steps, report, chatMessages]);
  
  // Context-aware suggested questions based on predictions
  const suggestedQuestions = React.useMemo(() => {
    const questions: string[] = [];
    
    // Always include these basics
    questions.push("What is the predicted treatment response?");
    
    // Add based on predictions
    const analyzeStep = steps.find(s => s.step === "analyze");
    if (analyzeStep?.data?.predictions) {
      const preds = analyzeStep.data.predictions as Record<string, AgentPrediction>;
      
      if (preds.platinum_sensitivity) {
        questions.push("Why was this platinum response predicted?");
        if (preds.platinum_sensitivity.label === "non-responder") {
          questions.push("What alternative treatments might work?");
        }
      }
      
      if (preds.survival_5y) {
        questions.push("What is the 5-year survival prognosis?");
      }
    }
    
    // Add region questions if we have evidence
    if (topEvidence.length > 0) {
      questions.push("What are the high-attention regions?");
      questions.push("Show me the concerning areas");
    }
    
    // Add similar cases question if we have them
    const retrieveStep = steps.find(s => s.step === "retrieve");
    if (retrieveStep?.data?.similar_cases) {
      questions.push("How does this compare to similar cases?");
    }
    
    return questions.slice(0, 4);
  }, [steps, topEvidence]);

  return (
    <div className={cn("flex flex-col h-full bg-white rounded-lg shadow-sm", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-indigo-100 rounded-lg">
            <Sparkles className="h-5 w-5 text-indigo-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">AI Assistant</h3>
            <p className="text-xs text-gray-500">
              {isAnalyzing ? "Analyzing..." : sessionId ? "Analysis complete" : "Multi-step analysis"}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Export buttons */}
          {sessionId && (
            <>
              <button
                onClick={copyConversation}
                className={cn(
                  "p-2 rounded-lg transition-colors",
                  copySuccess ? "bg-green-100 text-green-600" : "hover:bg-gray-100 text-gray-500"
                )}
                title="Copy to clipboard"
              >
                {copySuccess ? <CheckCircle2 className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              </button>
              <button
                onClick={exportConversation}
                className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors"
                title="Export as JSON"
              >
                <Download className="h-4 w-4" />
              </button>
            </>
          )}
          
          <button
            onClick={startAnalysis}
            disabled={!slideId || isAnalyzing}
            className={cn(
              "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
              slideId && !isAnalyzing
                ? "bg-indigo-600 text-white hover:bg-indigo-700"
                : "bg-gray-100 text-gray-400 cursor-not-allowed"
            )}
          >
            {isAnalyzing ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <Bot className="h-4 w-4" />
                {sessionId ? "Re-analyze" : "Start Analysis"}
              </span>
            )}
          </button>
        </div>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {!slideId ? (
          <div className="flex-1 flex items-center justify-center text-gray-400 p-8 text-center">
            <div>
              <Bot className="h-12 w-12 mx-auto mb-3 opacity-30" />
              <p>Select a slide to begin AI-powered analysis</p>
            </div>
          </div>
        ) : steps.length === 0 && !isAnalyzing ? (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-500 p-8">
            <div className="text-center max-w-sm">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-indigo-50 flex items-center justify-center">
                <Bot className="h-8 w-8 text-indigo-500" />
              </div>
              <h4 className="font-medium text-gray-900 mb-2">
                Ready for Analysis
              </h4>
              <p className="text-sm mb-4">
                I will analyze slide <span className="font-mono text-indigo-600">{slideId}</span> using
                multiple AI models, find similar cases, and generate a comprehensive report.
              </p>
              <div className="text-xs text-gray-400 space-y-1">
                <p>• Multi-model TransMIL predictions</p>
                <p>• Similar case retrieval</p>
                <p>• Evidence-based reasoning</p>
                <p>• Structured clinical report</p>
              </div>
            </div>
          </div>
        ) : (
          <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Workflow Steps */}
            {steps.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-gray-500 uppercase tracking-wide flex items-center gap-2">
                  <Activity className="h-3 w-3" />
                  Analysis Progress
                </div>
                {steps.map((step) => (
                  <WorkflowStep
                    key={step.step}
                    step={step}
                    isExpanded={expandedSteps.has(step.step)}
                    onToggle={() => toggleStep(step.step)}
                    slideId={slideId}
                    onHighlightRegion={onHighlightRegion}
                  />
                ))}
              </div>
            )}
            
            {/* Error */}
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center gap-2 text-red-600">
                  <AlertCircle className="h-4 w-4" />
                  <span className="font-medium">Error</span>
                </div>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            )}
            
            {/* Chat Messages */}
            {chatMessages.length > 0 && (
              <div className="space-y-2 pt-4 border-t">
                <div className="text-xs font-medium text-gray-500 uppercase tracking-wide flex items-center gap-2">
                  <MessageSquare className="h-3 w-3" />
                  Follow-up Questions
                </div>
                <div className="space-y-3">
                  {chatMessages.map((msg, i) => (
                    <ChatMessage
                      key={i}
                      message={msg.message}
                      isUser={msg.isUser}
                      timestamp={msg.timestamp}
                      evidencePatches={msg.evidencePatches}
                      onHighlightRegion={onHighlightRegion}
                    />
                  ))}
                  {isFollowupLoading && (
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center">
                        <Loader2 className="h-4 w-4 animate-spin text-indigo-600" />
                      </div>
                      <div className="bg-gray-100 rounded-lg px-4 py-2">
                        <div className="flex items-center gap-2 text-gray-500 text-sm">
                          <span className="animate-pulse">Thinking</span>
                          <span className="animate-bounce">.</span>
                          <span className="animate-bounce" style={{ animationDelay: "0.1s" }}>.</span>
                          <span className="animate-bounce" style={{ animationDelay: "0.2s" }}>.</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* Suggested Questions */}
            {sessionId && chatMessages.length === 0 && !isAnalyzing && (
              <div className="pt-4 border-t">
                <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-2">
                  <Lightbulb className="h-3 w-3" />
                  Ask a Question
                </div>
                <div className="flex flex-wrap gap-2">
                  {suggestedQuestions.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setInputMessage(q);
                        inputRef.current?.focus();
                      }}
                      className="text-xs px-3 py-1.5 bg-indigo-50 text-indigo-600 rounded-full hover:bg-indigo-100 transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Input */}
      {sessionId && (
        <div className="p-4 border-t bg-gray-50">
          <div className="flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about predictions, regions, or similar cases..."
              disabled={isFollowupLoading}
              className={cn(
                "flex-1 px-4 py-2 border rounded-lg text-sm bg-white",
                "focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent",
                isFollowupLoading && "bg-gray-100"
              )}
            />
            <button
              onClick={sendFollowup}
              disabled={!inputMessage.trim() || isFollowupLoading}
              className={cn(
                "p-2 rounded-lg transition-colors",
                inputMessage.trim() && !isFollowupLoading
                  ? "bg-indigo-600 text-white hover:bg-indigo-700"
                  : "bg-gray-200 text-gray-400"
              )}
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AIAssistantPanel;
