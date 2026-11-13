"use client";

import React from "react";
import { ShieldCheck, Cpu, ArrowUpRight, Sparkles } from "lucide-react";

interface NavbarProps {
  onScrollToWorkspace?: () => void;
  blindMode?: boolean;
  onToggleBlindMode?: () => void;
}

export function Navbar({
  onScrollToWorkspace,
  blindMode = false,
  onToggleBlindMode,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-zinc-700/60 bg-zinc-900 text-zinc-100 shadow-sm">
            <Cpu className="h-5 w-5 text-blue-500" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="text-base font-semibold tracking-tight text-zinc-100">
                ResumeRank
              </span>
              <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-zinc-400">
                ENTERPRISE
              </span>
            </div>
            <span className="text-[11px] text-zinc-500">
              Hybrid Candidate Intelligence Engine
            </span>
          </div>
        </div>

        {/* Center Links */}
        <nav className="hidden items-center gap-8 md:flex">
          <button
            onClick={onScrollToWorkspace}
            className="text-sm font-medium text-zinc-300 transition-colors hover:text-white"
          >
            Workspace
          </button>
          <a
            href="#architecture"
            className="text-sm font-medium text-zinc-400 transition-colors hover:text-zinc-200"
          >
            Architecture
          </a>
          <a
            href="#benchmarks"
            className="text-sm font-medium text-zinc-400 transition-colors hover:text-zinc-200"
          >
            Verification
          </a>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-sm font-medium text-zinc-400 transition-colors hover:text-zinc-200"
          >
            API Docs
            <ArrowUpRight className="h-3.5 w-3.5 text-zinc-500" />
          </a>
        </nav>

        {/* Right Actions */}
        <div className="flex items-center gap-3">
          {onToggleBlindMode && (
            <button
              onClick={onToggleBlindMode}
              className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all ${
                blindMode
                  ? "border-blue-600/60 bg-blue-950/40 text-blue-400"
                  : "border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
              }`}
              title="Blind Hiring Mode redacts candidate names, locations, and institutions"
            >
              <ShieldCheck
                className={`h-4 w-4 ${blindMode ? "text-blue-400" : "text-zinc-500"}`}
              />
              <span className="hidden sm:inline">Blind Screening:</span>
              <span>{blindMode ? "Active" : "Off"}</span>
            </button>
          )}

          <button
            onClick={onScrollToWorkspace}
            className="flex items-center gap-2 rounded-lg bg-zinc-100 px-3.5 py-1.5 text-xs font-semibold text-zinc-950 shadow-sm transition-all hover:bg-white active:scale-95"
          >
            <Sparkles className="h-3.5 w-3.5 text-blue-600" />
            Launch Screening
          </button>
        </div>
      </div>
    </header>
  );
}
