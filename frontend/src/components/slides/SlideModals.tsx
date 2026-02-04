"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import { X, Palette, FolderOpen } from "lucide-react";

// Color palette for tags
const TAG_COLOR_OPTIONS = [
  { name: "red", class: "bg-red-500" },
  { name: "orange", class: "bg-orange-500" },
  { name: "amber", class: "bg-amber-500" },
  { name: "yellow", class: "bg-yellow-500" },
  { name: "lime", class: "bg-lime-500" },
  { name: "green", class: "bg-green-500" },
  { name: "emerald", class: "bg-emerald-500" },
  { name: "teal", class: "bg-teal-500" },
  { name: "cyan", class: "bg-cyan-500" },
  { name: "sky", class: "bg-sky-500" },
  { name: "blue", class: "bg-blue-500" },
  { name: "indigo", class: "bg-indigo-500" },
  { name: "violet", class: "bg-violet-500" },
  { name: "purple", class: "bg-purple-500" },
  { name: "fuchsia", class: "bg-fuchsia-500" },
  { name: "pink", class: "bg-pink-500" },
  { name: "rose", class: "bg-rose-500" },
];

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title: string;
  icon?: React.ReactNode;
}

function Modal({ isOpen, onClose, children, title, icon }: ModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal content */}
      <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-modal-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-gray-50 to-white">
          <div className="flex items-center gap-3">
            {icon}
            <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
}

// Create Tag Modal
interface CreateTagModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreateTag: (name: string, color: string) => void;
  existingTags: string[];
  isLoading?: boolean;
}

export function CreateTagModal({
  isOpen,
  onClose,
  onCreateTag,
  existingTags,
  isLoading,
}: CreateTagModalProps) {
  const [name, setName] = useState("");
  const [color, setColor] = useState("blue");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const trimmedName = name.trim().toLowerCase();
    if (!trimmedName) {
      setError("Tag name is required");
      return;
    }
    if (existingTags.includes(trimmedName)) {
      setError("A tag with this name already exists");
      return;
    }

    onCreateTag(trimmedName, color);
    setName("");
    setColor("blue");
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Create New Tag"
      icon={<Palette className="h-5 w-5 text-clinical-500" />}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Tag name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">
            Tag Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setError(null);
            }}
            placeholder="e.g., high-priority, review-needed"
            className={cn(
              "w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 transition-all",
              error
                ? "border-red-300 focus:ring-red-500/20 focus:border-red-500"
                : "border-gray-200 focus:ring-clinical-500/20 focus:border-clinical-500"
            )}
            autoFocus
          />
          {error && <p className="mt-1.5 text-sm text-red-500">{error}</p>}
        </div>

        {/* Color picker */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Color
          </label>
          <div className="flex flex-wrap gap-2">
            {TAG_COLOR_OPTIONS.map((option) => (
              <button
                key={option.name}
                type="button"
                onClick={() => setColor(option.name)}
                className={cn(
                  "w-8 h-8 rounded-full transition-all",
                  option.class,
                  color === option.name
                    ? "ring-2 ring-offset-2 ring-gray-900 scale-110"
                    : "hover:scale-110"
                )}
                title={option.name}
              />
            ))}
          </div>
        </div>

        {/* Preview */}
        {name && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              Preview
            </label>
            <div
              className={cn(
                "inline-flex px-2.5 py-1 text-sm font-medium rounded-full",
                `bg-${color}-100 text-${color}-800 border border-${color}-200`
              )}
              style={{
                backgroundColor: `var(--${color}-100, #dbeafe)`,
                color: `var(--${color}-800, #1e40af)`,
              }}
            >
              {name.trim().toLowerCase() || "tag-name"}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3 pt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" isLoading={isLoading}>
            Create Tag
          </Button>
        </div>
      </form>
    </Modal>
  );
}

// Create Group Modal
interface CreateGroupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreateGroup: (name: string, description?: string) => void;
  existingGroups: string[];
  isLoading?: boolean;
}

export function CreateGroupModal({
  isOpen,
  onClose,
  onCreateGroup,
  existingGroups,
  isLoading,
}: CreateGroupModalProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const trimmedName = name.trim();
    if (!trimmedName) {
      setError("Group name is required");
      return;
    }
    if (existingGroups.some((g) => g.toLowerCase() === trimmedName.toLowerCase())) {
      setError("A group with this name already exists");
      return;
    }

    onCreateGroup(trimmedName, description.trim() || undefined);
    setName("");
    setDescription("");
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Create New Group"
      icon={<FolderOpen className="h-5 w-5 text-clinical-500" />}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Group name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">
            Group Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setError(null);
            }}
            placeholder="e.g., Batch 2024-01, Study Cohort A"
            className={cn(
              "w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 transition-all",
              error
                ? "border-red-300 focus:ring-red-500/20 focus:border-red-500"
                : "border-gray-200 focus:ring-clinical-500/20 focus:border-clinical-500"
            )}
            autoFocus
          />
          {error && <p className="mt-1.5 text-sm text-red-500">{error}</p>}
        </div>

        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">
            Description <span className="text-gray-400">(optional)</span>
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Brief description of this group..."
            rows={3}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-clinical-500/20 focus:border-clinical-500 transition-all resize-none"
          />
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 pt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" isLoading={isLoading}>
            Create Group
          </Button>
        </div>
      </form>
    </Modal>
  );
}

// Confirm Delete Modal
interface ConfirmDeleteModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  count: number;
  isLoading?: boolean;
}

export function ConfirmDeleteModal({
  isOpen,
  onClose,
  onConfirm,
  count,
  isLoading,
}: ConfirmDeleteModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Confirm Deletion">
      <div className="space-y-4">
        <p className="text-gray-600">
          Are you sure you want to delete{" "}
          <span className="font-semibold text-gray-900">{count} slide{count > 1 ? "s" : ""}</span>?
          This action cannot be undone.
        </p>

        <div className="bg-red-50 border border-red-100 rounded-lg p-3">
          <p className="text-sm text-red-700">
            ⚠️ All associated data including embeddings, annotations, and analysis results will be
            permanently removed.
          </p>
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="danger"
            onClick={onConfirm}
            isLoading={isLoading}
          >
            Delete {count} Slide{count > 1 ? "s" : ""}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

// Add to Group Modal (for single slide)
interface AddToGroupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAddToGroup: (groupId: string) => void;
  groups: { id: string; name: string; slideIds: string[] }[];
  slideId: string;
  isLoading?: boolean;
}

export function AddToGroupModal({
  isOpen,
  onClose,
  onAddToGroup,
  groups,
  slideId,
  isLoading,
}: AddToGroupModalProps) {
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);

  const handleSubmit = () => {
    if (selectedGroup) {
      onAddToGroup(selectedGroup);
      setSelectedGroup(null);
      onClose();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Add to Group"
      icon={<FolderOpen className="h-5 w-5 text-clinical-500" />}
    >
      <div className="space-y-4">
        <p className="text-sm text-gray-600">
          Select a group to add this slide to:
        </p>

        <div className="space-y-2 max-h-60 overflow-y-auto">
          {groups.map((group) => {
            const alreadyInGroup = group.slideIds.includes(slideId);
            return (
              <button
                key={group.id}
                onClick={() => !alreadyInGroup && setSelectedGroup(group.id)}
                disabled={alreadyInGroup}
                className={cn(
                  "w-full flex items-center justify-between px-3 py-2.5 rounded-lg border transition-all text-left",
                  alreadyInGroup
                    ? "bg-gray-50 border-gray-200 text-gray-400 cursor-not-allowed"
                    : selectedGroup === group.id
                    ? "bg-clinical-50 border-clinical-300 text-clinical-700"
                    : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
                )}
              >
                <span className="font-medium">{group.name}</span>
                <span className="text-xs">
                  {alreadyInGroup ? "Already added" : `${group.slideIds.length} slides`}
                </span>
              </button>
            );
          })}
        </div>

        {groups.length === 0 && (
          <p className="text-sm text-gray-500 text-center py-4">
            No groups available. Create a group first.
          </p>
        )}

        <div className="flex justify-end gap-3 pt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="primary"
            onClick={handleSubmit}
            disabled={!selectedGroup}
            isLoading={isLoading}
          >
            Add to Group
          </Button>
        </div>
      </div>
    </Modal>
  );
}
