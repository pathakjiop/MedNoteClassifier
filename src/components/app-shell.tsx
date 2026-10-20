import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Extractor" },
  { to: "/dataset", label: "Dataset" },
  { to: "/models", label: "Models" },
  { to: "/about", label: "Model card" },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="min-h-dvh flex flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:bg-primary focus:text-primary-fg focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="border-b border-border bg-bg-elevated">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <Link to="/" className="flex items-baseline gap-2 no-underline">
            <span className="font-display text-xl font-semibold tracking-tight text-fg">
              MedNoteClassifier
            </span>
            <span className="hidden text-xs uppercase tracking-[0.14em] text-subtle sm:inline">
              Clinical NLP
            </span>
          </Link>
          <nav className="flex flex-wrap gap-1" aria-label="Primary">
            {NAV.map((item) => {
              const active = pathname === item.to;
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={cn(
                    "rounded-sm px-3 py-2 text-sm no-underline transition-colors duration-150",
                    active
                      ? "bg-primary text-primary-fg"
                      : "text-muted hover:bg-surface hover:text-fg",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <div className="border-b border-border bg-surface px-4 py-2 text-center text-xs text-muted sm:px-6">
        Research model trained on de-identified USMLE exam notes. Not a medical device. Not for
        diagnosis or care.
      </div>
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
        {children}
      </main>
      <footer className="border-t border-border px-4 py-6 text-center text-xs text-subtle sm:px-6">
        NBME Score Clinical Patient Notes · character-span extraction · seed 42
      </footer>
    </div>
  );
}
