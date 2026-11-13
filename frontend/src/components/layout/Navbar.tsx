"use client";

import React from "react";
import { ShieldCheck, Cpu, ArrowUpRight, Sparkles, Sun, Moon } from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

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
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-50 h-16 w-full border-b border-zinc-200/80 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md transition-colors">
      <div className="mx-auto flex h-full max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-zinc-200 dark:border-zinc-700/60 bg-zinc-100 dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 shadow-sm">
            <Cpu className="h-5 w-5 text-blue-600 dark:text-blue-500" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="text-base font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
                People Lens
              </span>
              <span className="rounded bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-zinc-600 dark:text-zinc-400">
                ENTERPRISE
              </span>
            </div>
            <span className="text-[11px] text-zinc-500">
              Candidate Intelligence &amp; Pipeline Platform
            </span>
          </div>
        </div>

        {/* Center Links */}
        <nav className="hidden items-center gap-8 md:flex">
          <button
            onClick={onScrollToWorkspace}
            className="text-sm font-medium text-zinc-600 dark:text-zinc-300 transition-colors hover:text-zinc-950 dark:hover:text-white"
          >
            Workspace
          </button>
          <a
            href="#pipeline"
            onClick={onScrollToWorkspace}
            className="text-sm font-medium text-zinc-500 dark:text-zinc-400 transition-colors hover:text-zinc-900 dark:hover:text-zinc-200"
          >
            Pipeline
          </a>
          <a
            href="#architecture"
            className="text-sm font-medium text-zinc-500 dark:text-zinc-400 transition-colors hover:text-zinc-900 dark:hover:text-zinc-200"
          >
            Architecture
          </a>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-sm font-medium text-zinc-500 dark:text-zinc-400 transition-colors hover:text-zinc-900 dark:hover:text-zinc-200"
          >
            API Docs
            <ArrowUpRight className="h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
          </a>
        </nav>

        {/* Right Actions */}
        <div className="flex items-center gap-2.5">
          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-100/80 dark:bg-zinc-900/60 text-zinc-600 dark:text-zinc-400 transition-all hover:bg-zinc-200/70 dark:hover:bg-zinc-800 hover:text-zinc-900 dark:hover:text-zinc-100"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4 text-amber-400" />
            ) : (
              <Moon className="h-4 w-4 text-zinc-700" />
            )}
          </button>

          {onToggleBlindMode && (
            <button
              onClick={onToggleBlindMode}
              className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all ${
                blindMode
                  ? "border-blue-500/60 bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400"
                  : "border-zinc-200 dark:border-zinc-800 bg-zinc-100/80 dark:bg-zinc-900/60 text-zinc-600 dark:text-zinc-400 hover:border-zinc-300 dark:hover:border-zinc-700 hover:text-zinc-900 dark:hover:text-zinc-200"
              }`}
              title="Blind Hiring Mode redacts candidate names, locations, and institutions"
            >
              <ShieldCheck
                className={`h-4 w-4 ${blindMode ? "text-blue-600 dark:text-blue-400" : "text-zinc-400 dark:text-zinc-500"}`}
              />
              <span className="hidden sm:inline">Blind Screening:</span>
              <span>{blindMode ? "Active" : "Off"}</span>
            </button>
          )}

          <button
            onClick={onScrollToWorkspace}
            className="flex items-center gap-2 rounded-lg bg-zinc-900 dark:bg-zinc-100 px-3.5 py-1.5 text-xs font-semibold text-white dark:text-zinc-950 shadow-sm transition-all hover:bg-zinc-800 dark:hover:bg-white active:scale-95"
          >
            <Sparkles className="h-3.5 w-3.5 text-blue-500 dark:text-blue-600" />
            Launch Screening
          </button>
        </div>
      </div>
    </header>
  );
}
