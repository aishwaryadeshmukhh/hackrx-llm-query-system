'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import styles from './page.module.css'

const DEMO_STEPS = [
  { delay: 0,    type: 'q',    text: 'Is cataract surgery covered after 18 months on this policy?' },
  { delay: 900,  type: 'tool', text: 'check_waiting_period({ "benefit_type": "cataract" })' },
  { delay: 1900, type: 'obs',  text: 'Specified disease waiting period: 24 months (Code-Excl02). Cataract listed under specified conditions.' },
  { delay: 3000, type: 'tool', text: 'lookup_exclusions({ "procedure_or_condition": "cataract surgery" })' },
  { delay: 4000, type: 'obs',  text: 'Cataract surgery excluded until expiry of 24 months of continuous coverage.' },
  { delay: 5200, type: 'ans',  text: 'not_covered — 18 months < 24-month specified disease waiting period (Code-Excl02).' },
]

const FEATURES = [
  {
    title: 'Reads the fine print so you don\'t have to',
    body: 'Upload any health insurance PDF and ask a question in plain English. The system retrieves the exact clause that applies — section name, page number, and the precise policy language.',
  },
  {
    title: 'Shows its work, step by step',
    body: 'A ReAct agent reasons through waiting periods, exclusions, and conditional coverage in sequence. You see every tool call and what it found, not just a final answer.',
  },
  {
    title: 'Gives a decision, not a hedge',
    body: 'Covered, not covered, partial, or unclear — with a confidence score. Decision logic is applied strictly: if the waiting period hasn\'t elapsed, it says so.',
  },
]

export default function Home() {
  const [visibleSteps, setVisibleSteps] = useState([])
  const [done, setDone] = useState(false)
  const timerRef = useRef([])

  function runDemo() {
    timerRef.current.forEach(clearTimeout)
    timerRef.current = []
    setVisibleSteps([])
    setDone(false)

    DEMO_STEPS.forEach((step, i) => {
      const t = setTimeout(() => {
        setVisibleSteps(prev => [...prev, step])
        if (i === DEMO_STEPS.length - 1) setDone(true)
      }, step.delay + 400)
      timerRef.current.push(t)
    })
  }

  useEffect(() => {
    runDemo()
    return () => timerRef.current.forEach(clearTimeout)
  }, [])

  useEffect(() => {
    if (done) {
      const t = setTimeout(runDemo, 4000)
      timerRef.current.push(t)
    }
  }, [done])

  return (
    <main className={styles.landing}>

      {/* Nav */}
      <nav className={styles.nav}>
        <div className={styles.navInner}>
          <span className={styles.navWordmark}>PolicyMind</span>
          <div className={styles.navRight}>
            <span className={styles.navBadge}>HackRx 2024</span>
            <Link href="/analyze" className={styles.navCta}>Try it</Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <div className={styles.heroText}>
            <p className={styles.eyebrow}>Agentic RAG · Insurance</p>
            <h1 className={styles.heroTitle}>
              Know exactly what<br />
              your policy covers —<br />
              <em className={styles.heroEm}>and why.</em>
            </h1>
            <p className={styles.heroSub}>
              Upload a health insurance PDF, ask any coverage question,
              and get an answer that cites the actual clause — not a summary,
              not a guess.
            </p>
            <div className={styles.heroCtas}>
              <Link href="/analyze" className={styles.ctaPrimary}>
                Try it now
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M3 8h10M9 4l4 4-4 4" />
                </svg>
              </Link>
              <a href="#what-it-does" className={styles.ctaGhost}>See what it does</a>
            </div>
          </div>

          {/* Animated agent trace */}
          <div className={styles.terminal} aria-label="Live example of agent reasoning">
            <div className={styles.terminalBar}>
              <span className={styles.termDot} style={{ background: '#F87171' }} />
              <span className={styles.termDot} style={{ background: '#FBBF24' }} />
              <span className={styles.termDot} style={{ background: '#2DD4BF' }} />
              <span className={styles.termTitle}>agent trace</span>
            </div>
            <div className={styles.termBody}>
              {visibleSteps.map((step, i) => (
                <div key={i} className={styles.termLine}>
                  {step.type === 'q' && (
                    <><span className={styles.termPrefix}>?</span><span className={styles.termText}>{step.text}</span></>
                  )}
                  {step.type === 'tool' && (
                    <><span className={styles.termPrefix}>›</span><code className={styles.termCode}>{step.text}</code></>
                  )}
                  {step.type === 'obs' && (
                    <><span className={styles.termPrefix}>·</span><span className={styles.termObs}>{step.text}</span></>
                  )}
                  {step.type === 'ans' && (
                    <><span className={styles.termPrefixAns}>✓</span><span className={styles.termAns}>{step.text}</span></>
                  )}
                </div>
              ))}
              {!done && visibleSteps.length > 0 && (
                <div className={styles.termCursor} aria-hidden="true" />
              )}
            </div>
          </div>
        </div>
      </section>


      {/* Features */}
      <section className={styles.features} id="what-it-does">
        <div className={styles.featuresInner}>
          <p className={styles.sectionEye}>What it does</p>
          <div className={styles.featureGrid}>
            {FEATURES.map((f, i) => (
              <div key={i} className={styles.featureCard}>
                <h3 className={styles.featureTitle}>{f.title}</h3>
                <p className={styles.featureBody}>{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA band */}
      <section className={styles.ctaBand}>
        <div className={styles.ctaBandInner}>
          <h2 className={styles.ctaBandTitle}>Upload a policy. Ask a question.</h2>
          <p className={styles.ctaBandSub}>No configuration needed. Results in under a minute.</p>
          <Link href="/analyze" className={styles.ctaPrimary}>
            Open the analyzer
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 8h10M9 4l4 4-4 4" />
            </svg>
          </Link>
        </div>
      </section>

      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <span className={styles.footerWordmark}>PolicyMind</span>
          <p className={styles.footerNote}>HackRx 2024 · Pinecone · Groq · Gemini</p>
        </div>
      </footer>

    </main>
  )
}
