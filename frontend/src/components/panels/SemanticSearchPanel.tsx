"use client";

import React, { useState, useCallback, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { cn } from "@/lib/utils";
import {
  Search,
  X,
  ZoomIn,
  Crosshair,
  Lightbulb,
  AlertCircle,
} from "lucide-react";
import type { SemanticSearchResult, PatchCoordinates } from "@/types";
import { getPatchUrl } from "@/lib/api";

interface SemanticSearchPanelProps {
  slideId: string | null;
  projectId?: string;
  isAnalyzed: boolean;
  onSearch: (query: string, topK: number) => Promise<void>;
  results: SemanticSearchResult[];
  isSearching: boolean;
  error?: string | null;
  onPatchClick?: (coords: PatchCoordinates) => void;
  onPatchDeselect?: () => void;
  selectedPatchId?: string;
  inputRef?: React.RefObject<HTMLInputElement>;
  onClearResults?: () => void;
}

const EXAMPLE_QUERIES = [
  "tumor cells",
  "stromal tissue",
  "necrosis",
  "inflammatory infiltrate",
  "glandular structures",
];

export function SemanticSearchPanel({
  slideId,
  projectId,
  isAnalyzed,
  onSearch,
  results,
  isSearching,
  error,
  onPatchClick,
  onPatchDeselect,
  selectedPatchId,
  inputRef,
  onClearResults,
}: SemanticSearchPanelProps) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [hasSubmittedQuery, setHasSubmittedQuery] = useState(false);

  useEffect(() => {
    setHasSubmittedQuery(false);
  }, [slideId]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = query.trim();
      if (!trimmed) return;
      setHasSubmittedQuery(true);
      await onSearch(trimmed, topK);
    },
    [query, topK, onSearch]
  );

  const handleExampleClick = useCallback(
    async (exampleQuery: string) => {
      setQuery(exampleQuery);
      setHasSubmittedQuery(true);
      await onSearch(exampleQuery, topK);
    },
    [topK, onSearch]
  );

  const clearQuery = useCallback(() => {
    setQuery("");
    setHasSubmittedQuery(false);
    onClearResults?.();
  }, [onClearResults]);

  // Not ready to search - no slide analyzed
  if (!slideId || !isAnalyzed) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-4 w-4 text-gray-400" />
            MedSigLIP Semantic Search
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-gray-500">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gray-100 flex items-center justify-center">
              <Search className="h-6 w-6 text-gray-400" />
            </div>
            <p className="text-sm font-medium text-gray-600">
              Search unavailable
            </p>
            <p className="text-xs mt-1.5 text-gray-500 max-w-[200px] mx-auto">
              Run analysis on a slide to enable MedSigLIP semantic search by description.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Search className="h-4 w-4 text-clinical-600" />
          MedSigLIP Semantic Search
          {results.length > 0 && (
            <Badge variant="info" size="sm" className="font-mono">
              {results.length}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        {/* Search Form */}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search patches by description..."
              className={cn(
                "w-full pl-3 pr-10 py-2 text-sm rounded-lg border transition-colors",
                "border-gray-300 focus:border-clinical-500 focus:ring-2 focus:ring-clinical-200",
                "placeholder:text-gray-400"
              )}
              disabled={isSearching}
            />
            {query && (
              <button
                type="button"
                onClick={clearQuery}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                aria-label="Clear search query"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600">Results:</label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="text-xs border border-gray-300 rounded px-2 py-1 focus:border-clinical-500 focus:ring-1 focus:ring-clinical-200"
              disabled={isSearching}
            >
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
            </select>
            <div className="flex-1" />
            <Button
              type="submit"
              size="sm"
              disabled={!query.trim() || isSearching}
              className="px-4"
            >
              {isSearching ? (
                <>
                  <Spinner size="sm" className="mr-1.5" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="h-3.5 w-3.5 mr-1.5" />
                  Search
                </>
              )}
            </Button>
          </div>
        </form>

        {/* Example Queries */}
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 text-xs text-gray-500">
            <Lightbulb className="h-3 w-3" />
            <span>Try:</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLE_QUERIES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => handleExampleClick(example)}
                disabled={isSearching}
                className={cn(
                  "text-xs px-2 py-1 rounded-full border transition-colors",
                  "border-gray-200 bg-gray-50 text-gray-600",
                  "hover:border-clinical-400 hover:bg-clinical-50 hover:text-clinical-700",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-xs text-red-700">{error}</p>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-2 pt-2 border-t border-gray-100">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span className="font-medium">Matching Patches</span>
              <span>Similarity Score</span>
            </div>
            {selectedPatchId && (
              <p className="text-2xs text-clinical-700">
                Tip: click the selected patch again to deselect.
              </p>
            )}
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {results.map((result, index) => {
                // Match selection based on coordinates (what handlePatchClick sets)
                const coordsMatch = result.coordinates
                  ? `${result.coordinates[0]}_${result.coordinates[1]}`
                  : null;
                const isSelected = coordsMatch !== null && selectedPatchId === coordsMatch;
                return (
                  <SearchResultItem
                    key={`${result.patch_index}_${index}`}
                    result={result}
                    rank={index + 1}
                    slideId={slideId}
                    projectId={projectId}
                    isSelected={isSelected}
                    onClick={() => {
                      if (!result.coordinates) return;

                      // Clicking an already selected semantic-search patch deselects it.
                      if (isSelected) {
                        onPatchDeselect?.();
                        return;
                      }

                      onPatchClick?.({
                        x: result.coordinates[0],
                        y: result.coordinates[1],
                        width: 224,
                        height: 224,
                        level: 0,
                      });
                    }}
                  />
                );
              })}
            </div>
          </div>
        )}

        {/* No Results State */}
        {!isSearching &&
          hasSubmittedQuery &&
          query.trim().length > 0 &&
          results.length === 0 &&
          !error && (
            <div className="text-center py-4 text-gray-500">
              <p className="text-sm">No matching patches found.</p>
              <p className="text-xs mt-1">Try a different description.</p>
            </div>
          )}
      </CardContent>
    </Card>
  );
}

// Search Result Item Component
interface SearchResultItemProps {
  result: SemanticSearchResult;
  rank: number;
  slideId: string;
  projectId?: string;
  isSelected: boolean;
  onClick: () => void;
}

function SearchResultItem({
  result,
  rank,
  slideId,
  projectId,
  isSelected,
  onClick,
}: SearchResultItemProps) {
  const similarityPercent = Math.round(result.similarity * 100);
  const similarityColor =
    similarityPercent >= 70
      ? "bg-green-500"
      : similarityPercent >= 40
      ? "bg-amber-500"
      : "bg-blue-500";

  const patchUrl = getPatchUrl(slideId, `patch_${result.patch_index}`, {
    projectId,
    coordinates: result.coordinates,
    patchSize: result.patch_size,
  });
  const canNavigate = !!result.coordinates;

  return (
    <button
      onClick={onClick}
      disabled={!canNavigate}
      className={cn(
        "w-full flex items-center gap-3 p-2.5 rounded-lg border transition-all text-left group",
        "hover:border-clinical-500 hover:bg-clinical-50/50 hover:shadow-clinical",
        "focus:outline-none focus:ring-2 focus:ring-clinical-500",
        isSelected
          ? "border-clinical-600 bg-clinical-50 ring-1 ring-clinical-200"
          : "border-gray-200 bg-white",
        !canNavigate && "opacity-70 cursor-not-allowed hover:border-gray-200 hover:bg-white hover:shadow-none"
      )}
      title={
        canNavigate
          ? isSelected
            ? "Selected. Click again to deselect"
            : "Jump to this patch"
          : "Coordinates unavailable for this patch"
      }
    >
      {/* Thumbnail */}
      <div className="relative w-12 h-12 rounded-lg overflow-hidden shrink-0 border border-gray-200 group-hover:border-clinical-300">
        <img
          src={patchUrl}
          alt={`Patch ${result.patch_index}`}
          className="w-full h-full object-cover"
          onError={(e) => {
            // Fallback if image fails to load
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
        <div className="absolute top-0 left-0 bg-navy-900/80 text-white text-2xs font-bold px-1 py-0.5 rounded-br shadow">
          #{rank}
        </div>
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-medium text-gray-900">
            Patch {result.patch_index}
          </span>
          <div
            className={cn(
              "px-1.5 py-0.5 rounded text-2xs font-bold text-white",
              similarityColor
            )}
          >
            {similarityPercent}%
          </div>
        </div>

        {result.coordinates ? (
          <div className="flex items-center gap-1.5 text-2xs text-gray-500">
            <Crosshair className="h-3 w-3" />
            <span className="font-mono">
              ({result.coordinates[0].toLocaleString()}, {result.coordinates[1].toLocaleString()})
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 text-2xs text-amber-700">
            <AlertCircle className="h-3 w-3" />
            <span>Coordinates unavailable</span>
          </div>
        )}
      </div>

      {/* Navigate indicator */}
      <div className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
        <ZoomIn className="h-4 w-4 text-clinical-600" />
      </div>
    </button>
  );
}
