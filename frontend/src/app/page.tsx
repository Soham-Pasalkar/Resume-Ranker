"use client";

import React, { useState } from "react";
import { Navbar } from "@/components/layout/Navbar";
import { HeroSection } from "@/components/hero/HeroSection";
import { RecruiterWorkspace } from "@/components/workspace/RecruiterWorkspace";
import { ArchitectureSection } from "@/components/landing/ArchitectureSection";
import { Footer } from "@/components/layout/Footer";

export default function Home() {
  const [blindMode, setBlindMode] = useState(false);

  const scrollToWorkspace = () => {
    const el = document.getElementById("workspace");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <Navbar
        onScrollToWorkspace={scrollToWorkspace}
        blindMode={blindMode}
        onToggleBlindMode={() => setBlindMode(!blindMode)}
      />

      <main className="flex-1">
        <HeroSection
          onGetStarted={scrollToWorkspace}
          onLoadSample={scrollToWorkspace}
        />

        <RecruiterWorkspace
          blindMode={blindMode}
          onToggleBlindMode={() => setBlindMode(!blindMode)}
        />

        <ArchitectureSection />
      </main>

      <Footer />
    </div>
  );
}
