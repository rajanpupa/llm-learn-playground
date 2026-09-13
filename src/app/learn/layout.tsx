"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  LESSONS,
  getLesson,
  getNextLesson,
  getPrevLesson,
  formatLessonNumber,
} from "@/lib/lessons";

const GROUPS = ["Foundations", "The Architecture", "Learning", "Inference & Wrap-up"];

export default function LearnLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const slug = pathname.split("/").filter(Boolean).pop() ?? "";
  const lesson = getLesson(slug);
  const [menuOpen, setMenuOpen] = useState(false);

  const index = lesson ? lesson.number - 1 : 0;
  const prev = getPrevLesson(slug);
  const next = getNextLesson(slug);

  return (
    <div className="lesson-layout">
      {/* Mobile header */}
      <div className="mobile-bar">
        <Link className="mobile-home" href="/">
          ← Home
        </Link>
        <button className="btn btn-ghost btn-sm" onClick={() => setMenuOpen((o) => !o)}>
          {menuOpen ? "Close" : "Lessons"}
        </button>
      </div>

      <aside className={`lesson-sidebar ${menuOpen ? "open" : ""}`}>
        <Link href="/" className="toc-link">
          <span className="toc-num">←</span> Back to home
        </Link>
        <Link href="/playground" className="toc-link" style={{ marginBottom: 14 }}>
          <span className="toc-num">🧪</span> <strong>Playground</strong>
        </Link>

        <div className="progress-wrap">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${((index + 1) / LESSONS.length) * 100}%` }}
            />
          </div>
          <span className="progress-text">
            {index + 1} / {LESSONS.length}
          </span>
        </div>

        {GROUPS.map((group) => {
          const items = LESSONS.filter((l) => l.category === group);
          if (items.length === 0) return null;
          return (
            <div className="toc-group" key={group}>
              <div className="toc-group-label">{group}</div>
              {items.map((l) => (
                <Link
                  key={l.slug}
                  href={`/learn/${l.slug}`}
                  className={`toc-link ${l.slug === slug ? "active" : ""}`}
                  onClick={() => setMenuOpen(false)}
                >
                  <span className="toc-num">{formatLessonNumber(l.number)}</span>
                  <span>{l.title}</span>
                </Link>
              ))}
            </div>
          );
        })}
      </aside>

      <div className="lesson-main">
        {lesson && (
          <header style={{ marginBottom: 28 }}>
            <p className="lesson-eyebrow">
              Lesson {formatLessonNumber(lesson.number)} of {LESSONS.length} · {lesson.category}
            </p>
            <h1 className="lesson-title">
              {formatLessonNumber(lesson.number)}. {lesson.title}
            </h1>
            <p className="lesson-summary">{lesson.summary}</p>
          </header>
        )}

        {children}

        <nav className="lesson-nav">
          {prev ? (
            <Link className="nav-card" href={`/learn/${prev.slug}`}>
              <div className="nav-label">← Previous</div>
              <div className="nav-title">
                {formatLessonNumber(prev.number)}. {prev.title}
              </div>
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link className="nav-card right" href={`/learn/${next.slug}`}>
              <div className="nav-label">Next →</div>
              <div className="nav-title">
                {formatLessonNumber(next.number)}. {next.title}
              </div>
            </Link>
          ) : (
            <Link className="nav-card right" href="/">
              <div className="nav-label">Done →</div>
              <div className="nav-title">Back to the start</div>
            </Link>
          )}
        </nav>
      </div>
    </div>
  );
}
