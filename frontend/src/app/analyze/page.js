'use client'

import { useState, useRef } from 'react'
import Link from 'next/link'
import styles from './analyze.module.css'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const SAMPLE_QUERIES = [
  'Is knee replacement surgery covered in the first year of the policy?',
  'What is the waiting period for pre-existing diseases?',
  'Is cataract surgery covered if the policy is 18 months old?',
  'Is bariatric surgery covered under this policy?',
  'Does the policy cover mental illness inpatient treatment?',
]

const DECISION_CONFIG = {
  covered:     { label: 'Covered',     color: 'var(--covered)',     bg: 'var(--covered-dim)' },
  not_covered: { label: 'Not Covered', color: 'var(--not-covered)', bg: 'var(--not-covered-dim)' },
  partial:     { label: 'Partial',     color: 'var(--partial)',     bg: 'var(--partial-dim)' },
  unclear:     { label: 'Unclear',     color: 'var(--unclear)',     bg: 'var(--unclear-dim)' },
  error:       { label: 'Error',       color: 'var(--unclear)',     bg: 'var(--unclear-dim)' },
}

export default function AnalyzePage() {
  const [pdfFile, setPdfFile]   = useState(null)
  const [question, setQuestion] = useState('')
  const [status, setStatus]     = useState(null)
  const [trace, setTrace]       = useState([])
  const [answer, setAnswer]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [openStep, setOpenStep] = useState(null)
  const fileInputRef            = useRef(null)
  const resultRef               = useRef(null)

  function handleFileChange(e) {
    const f = e.target.files[0]
    if (f && f.type === 'application/pdf') {
      setPdfFile(f); setAnswer(null); setError(null); setTrace([])
    } else {
      setError('Select a valid PDF file.'); setPdfFile(null)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type === 'application/pdf') {
      setPdfFile(f); setAnswer(null); setError(null); setTrace([])
    } else {
      setError('Drop a valid PDF file.')
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!pdfFile || !question.trim()) return

    setLoading(true); setError(null); setAnswer(null)
    setTrace([]); setStatus('Connecting…'); setOpenStep(null)

    try {
      const fd = new FormData()
      fd.append('file', pdfFile)
      fd.append('question', question.trim())

      const res = await fetch(`${API_URL}/hackrx/stream`, { method: 'POST', body: fd })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || err.error || `Server error ${res.status}`)
      }

      const reader = res.body.getReader()
      const dec    = new TextDecoder()
      let buf      = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          let ev
          try { ev = JSON.parse(line.slice(6)) } catch { continue }

          if (ev.type === 'status') {
            setStatus(ev.message)
          } else if (ev.type === 'thought') {
            setTrace(prev => [...prev, {
              step: ev.step, thought: ev.thought,
              action: ev.action, args: ev.args,
              observation: null, pending: true,
            }])
            setOpenStep(ev.step - 1)
            setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
          } else if (ev.type === 'observation') {
            setTrace(prev => prev.map(s =>
              s.step === ev.step ? { ...s, observation: ev.observation, pending: false } : s
            ))
          } else if (ev.type === 'answer') {
            setAnswer(ev); setStatus(null)
            setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
          } else if (ev.type === 'error') {
            setError(ev.message); setStatus(null)
          } else if (ev.type === 'done') {
            setStatus(null)
          }
        }
      }
    } catch (err) {
      setError(err.message); setStatus(null)
    } finally {
      setLoading(false)
    }
  }

  function useSample(q) { setQuestion(q); setAnswer(null); setError(null); setTrace([]) }

  const decision    = answer?.decision || 'unclear'
  const decisionCfg = DECISION_CONFIG[decision] || DECISION_CONFIG.unclear
  const confidence  = answer?.confidence ?? null

  return (
    <div className={styles.page}>
      {/* Sidebar */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarTop}>
          <Link href="/" className={styles.sidebarLogo}>
            <span className={styles.logoWord}>PolicyMind</span>
          </Link>
          <p className={styles.sidebarSub}>Insurance Policy Analyzer</p>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {/* File upload */}
          <div className={styles.field}>
            <label className={styles.label}>Policy Document</label>
            <div
              className={`${styles.dropzone} ${pdfFile ? styles.dropzoneHasFile : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={e => e.preventDefault()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
                className={styles.fileHidden}
              />
              {pdfFile ? (
                <div className={styles.fileChosen}>
                  <svg className={styles.fileIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
                  </svg>
                  <span className={styles.fileName} title={pdfFile.name}>{pdfFile.name}</span>
                  <button
                    type="button"
                    className={styles.fileRemove}
                    onClick={e => { e.stopPropagation(); setPdfFile(null); setAnswer(null) }}
                    aria-label="Remove file"
                  >✕</button>
                </div>
              ) : (
                <div className={styles.dropzonePrompt}>
                  <svg className={styles.uploadIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                  <span className={styles.dzText}>Drop PDF or click to upload</span>
                </div>
              )}
            </div>
          </div>

          {/* Question */}
          <div className={styles.field}>
            <label className={styles.label}>Coverage Question</label>
            <textarea
              className={styles.textarea}
              placeholder="Is knee replacement covered in the first year?"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              rows={4}
              required
            />
          </div>

          <button
            type="submit"
            className={styles.submitBtn}
            disabled={loading || !pdfFile || !question.trim()}
          >
            {loading ? (
              <span className={styles.btnLoading}>
                <span className={styles.spinner} />
                Analyzing…
              </span>
            ) : (
              <>
                Analyze
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M3 8h10M9 4l4 4-4 4" />
                </svg>
              </>
            )}
          </button>
        </form>

        {/* Sample queries */}
        <div className={styles.samples}>
          <span className={styles.samplesLabel}>Sample questions</span>
          <div className={styles.sampleList}>
            {SAMPLE_QUERIES.map((q, i) => (
              <button key={i} className={styles.sampleBtn} onClick={() => useSample(q)} type="button">
                {q}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Main panel */}
      <main className={styles.main}>
        {/* Empty state */}
        {!loading && !trace.length && !answer && !status && !error && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon} aria-hidden="true">
              <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="8" y="4" width="32" height="40" rx="3"/>
                <line x1="16" y1="16" x2="32" y2="16"/>
                <line x1="16" y1="22" x2="32" y2="22"/>
                <line x1="16" y1="28" x2="24" y2="28"/>
                <circle cx="36" cy="36" r="8" fill="none"/>
                <line x1="40" y1="40" x2="44" y2="44"/>
              </svg>
            </div>
            <h2 className={styles.emptyTitle}>Upload a policy to begin</h2>
            <p className={styles.emptyBody}>
              Select a health insurance PDF and ask any coverage question.
              The agent will retrieve the relevant clauses and explain its reasoning.
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className={styles.errorBox} role="alert">
            <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" className={styles.errorIcon}>
              <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm-.75 3.75a.75.75 0 011.5 0v4a.75.75 0 01-1.5 0v-4zm.75 7a.875.875 0 110-1.75.875.875 0 010 1.75z"/>
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Status */}
        {status && (
          <div className={styles.statusBar}>
            <span className={styles.statusDot} />
            <span>{status}</span>
          </div>
        )}

        {/* Results */}
        {(trace.length > 0 || answer) && (
          <div className={styles.results} ref={resultRef}>

            {/* Reasoning trace */}
            {trace.length > 0 && (
              <div className={styles.traceCard}>
                <div className={styles.traceCardHead}>
                  <h3 className={styles.cardLabel}>Reasoning Trace</h3>
                  <span className={styles.traceMeta}>
                    {trace.length} step{trace.length !== 1 ? 's' : ''}
                    {loading && <span className={styles.livePip}> live</span>}
                  </span>
                </div>
                <div className={styles.traceList}>
                  {trace.map((step, i) => (
                    <div key={i} className={styles.traceStep}>
                      <button
                        className={styles.traceToggle}
                        onClick={() => setOpenStep(openStep === i ? null : i)}
                        type="button"
                        aria-expanded={openStep === i}
                      >
                        <span className={styles.stepLeft}>
                          <span className={`${styles.stepNum} ${step.pending ? styles.stepPending : ''}`}>
                            {step.pending
                              ? <span className={styles.stepSpinner} />
                              : step.step
                            }
                          </span>
                          <span className={styles.toolBadge}>{step.action}</span>
                          {step.pending && (
                            <span className={styles.dots} aria-hidden="true">
                              <span/><span/><span/>
                            </span>
                          )}
                        </span>
                        <svg
                          className={`${styles.chevron} ${openStep === i ? styles.chevronOpen : ''}`}
                          viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"
                          strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
                        >
                          <path d="M4 6l4 4 4-4" />
                        </svg>
                      </button>

                      {openStep === i && (
                        <div className={styles.traceBody}>
                          {step.thought && (
                            <div className={styles.traceRow}>
                              <span className={styles.rowLabel}>Thought</span>
                              <p className={styles.rowText}>{step.thought}</p>
                            </div>
                          )}
                          <div className={styles.traceRow}>
                            <span className={styles.rowLabel}>Call</span>
                            <code className={styles.rowCode}>
                              {step.action}({JSON.stringify(step.args || {})})
                            </code>
                          </div>
                          <div className={styles.traceRow}>
                            <span className={styles.rowLabel}>Observation</span>
                            {step.observation
                              ? <p className={styles.rowObs}>{step.observation}</p>
                              : <p className={styles.rowFetching}>Retrieving chunks…</p>
                            }
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Decision */}
            {answer && (
              <>
                <div className={styles.decisionCard} style={{ '--dec-color': decisionCfg.color, '--dec-bg': decisionCfg.bg }}>
                  <div className={styles.decisionLeft}>
                    <span className={styles.decisionBadge}>{decisionCfg.label}</span>
                    {answer.query_type && (
                      <span className={styles.modeBadge}>
                        {answer.query_type === 'complex' ? 'ReAct' : 'Direct'}
                      </span>
                    )}
                  </div>
                  {confidence !== null && (
                    <div className={styles.confBlock}>
                      <span className={styles.confLabel}>Confidence</span>
                      <div className={styles.confBar} role="progressbar" aria-valuenow={Math.round(confidence * 100)} aria-valuemin={0} aria-valuemax={100}>
                        <div className={styles.confFill} style={{ width: `${Math.round(confidence * 100)}%` }} />
                      </div>
                      <span className={styles.confVal}>{Math.round(confidence * 100)}%</span>
                    </div>
                  )}
                </div>

                {answer.answer && (
                  <div className={styles.answerCard}>
                    <h3 className={styles.cardLabel}>Answer</h3>
                    <p className={styles.answerText}>{answer.answer}</p>
                  </div>
                )}

                {answer.justification && (
                  <div className={styles.answerCard}>
                    <h3 className={styles.cardLabel}>Clause Reference</h3>
                    <p className={styles.answerText}>{answer.justification}</p>
                  </div>
                )}

                {answer.relevant_clauses?.length > 0 && (
                  <div className={styles.clausesCard}>
                    <h3 className={styles.cardLabel}>Source Clauses</h3>
                    <div className={styles.clauseList}>
                      {answer.relevant_clauses.map((c, i) => (
                        <div key={i} className={styles.clauseItem}>
                          <div className={styles.clauseHeader}>
                            <span className={styles.clauseSection}>{c.section || c.document || c.doc_name}</span>
                            <div className={styles.clauseMeta}>
                              {c.page && <span className={styles.clausePage}>p. {c.page}</span>}
                              {c.score && <span className={styles.clauseScore}>{Math.round(c.score * 100)}%</span>}
                            </div>
                          </div>
                          {c.content && (
                            <blockquote className={styles.clauseQuote}>{c.content}</blockquote>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
