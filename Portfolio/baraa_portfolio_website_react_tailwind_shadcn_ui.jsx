import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Mail, Github, Linkedin, FileText, BarChart3, LayoutTemplate } from "lucide-react";

const data = {
  name: "Baraa Mohaisen",
  tagline: "Project Overview: Offline LLM + Wikipedia RAG / Distributed Labeling / Calibration / Benchmarks",
  email: "mohaisenbaraa@gmail.com",
  github: "https://github.com/yourhandle",
  linkedin: "https://www.linkedin.com/in/your-handle/",
  resumeUrl: "#",
  reportUrl: "#",
  about: "Skim-first portfolio distilled from the engineering report. Sections mirror the report and call out numbers, methods, and figures.",
};

type TableSpec = { caption: string; columns: string[]; rows: any[][] };

type Section = {
  id: string;
  title: string;
  micro?: string;
  bullets: string[];
  exec?: { intro?: string; points?: string[]; outro?: string };
  tables?: TableSpec[];
  graphs?: { label: string; filename: string; src: string }[];
  fallback?: { title: string; ascii: string[] };
  pie?: { data: { label: string; value: number }[] };
};

const sections: Section[] = [
  {
    id: "mandate-arch",
    title: "1) Project Mandate & System Architecture",
    micro: "Privacy-first, fully offline chatbot on local hardware. Local retrieval plus calibrated labeling delivers accuracy without a cloud dependency.",
    bullets: [
      "Goal: private, fully offline chatbot; replace brittle web-scraping with local retrieval.",
      "Constraint: CPU-bound consumer hardware; reliability prioritized over tricks.",
      "Architecture: Local LLM + offline Wikipedia index (SQL/FAISS) and orchestration for labeling/inference.",
    ],
    exec: {
      intro: "This project builds a reliable, private, fully offline chatbot that answers queries with no internet access. It guarantees privacy by running on local hardware while maintaining accuracy and efficiency.",
      points: [
        "Dynamic Search Classification: a custom classifier decides when external lookup is needed, balancing speed with factual accuracy.",
        "Confidence Calibration Pipeline: a multi-stage process converts noisy lightweight-model outputs into trustworthy, reusable training data.",
        "Distributed Labeling and fine-tuned BERT: a fault-tolerant system labeled and audited 200k+ examples; the BERT classifier reached about 90% accuracy in its first epoch.",
      ],
      outro: "Together, these parts form a blueprint for data-centric AI that is performant, robust across domains, and resilient under real-world constraints.",
    },
    graphs: [
      { label: "System Flow (optional diagram)", filename: "fig_architecture.png", src: "/images/fig_architecture.png" },
    ],
  },
  {
    id: "query-pipeline",
    title: "2) System Architecture - Query Classification Pipeline",
    micro: "User query -> preprocessing -> search-needed classifier. Deterministic path keeps latency low and privacy intact.",
    bullets: [
      "User query input -> normalization and entity extraction -> classification: {search_needed, confidence}.",
      "Preprocessing: spaCy pipeline for tokenization and NER; length and punctuation normalization; safe truncation for short text.",
      "Classifier acts as gatekeeper: search triggers offline retrieval; no-search routes directly to LLM response.",
    ],
    graphs: [
      { label: "Query Pipeline Flowchart", filename: "fig_query_pipeline.png", src: "/images/fig_query_pipeline.png" },
    ],
    fallback: {
      title: "Text Walkthrough (no graphic)",
      ascii: [
        "[User Query]",
        "  |",
        "  v",
        "[Preprocess] - normalize, tokenize, NER (spaCy), strip noise, cap length",
        "  |",
        "  v",
        "[Classifier] -> outputs { search_needed in {0,1}, confidence in [0,1] }",
        "  |-- search_needed = 1 -> [Retrieve (Offline Wikipedia)] -> [Summarize] -> [LLM Answer]",
        "  |-- search_needed = 0 -> [LLM Answer (direct)]",
      ],
    },
  },
  {
    id: "early-iterations",
    title: "2.1-2.3) Early Iterations and Expansion into Transformers",
    micro: "Traditional baselines underfit; BERT chosen for bidirectional context, calibration stability, and CPU-friendly latency.",
    bullets: [
      "Baselines (rules plus shallow ML) struggled with nuance: synonyms/paraphrase, multi-entity prompts, and short-text ambiguity.",
      "Confidence from baselines was unstable -> poor thresholding; transformers offered richer features and well-studied calibration.",
      "BERT outperformed small alternatives (XLNet, ELECTRA, DeBERTa, T5 considered) on accuracy/latency/tooling trade-offs.",
    ],
    fallback: {
      title: "Why BERT (ASCII quick compare)",
      ascii: [
        "Rules/Heuristics  -> brittle; fails on phrasing",
        "TF-IDF + LogReg   -> weak semantics; high FP",
        "RNN/CNN           -> struggles with long dependencies",
        "BERT (bi-transformer) -> stronger context both ways; calibration literature; acceptable CPU latency",
      ],
    },
  },
  {
    id: "processing-for-bert",
    title: "2.4) Processing Input for BERT Classification",
    micro: "spaCy plus rules yield clean, consistent short-text inputs. Sequence constraints avoid truncation artifacts.",
    bullets: [
      "Tools: spaCy for tokenization and NER; lightweight normalization.",
      "Steps: normalization (case/whitespace/punctuation), entity tagging, sequence length management (truncate or pad).",
      "Dependency parsing informs entity grouping for ambiguous multi-span queries.",
    ],
    fallback: {
      title: "Preprocess Details (no chart)",
      ascii: [
        "Normalize -> strip odd unicode, collapse whitespace, fix quotes",
        "spaCy -> tokenize, NER (ORG/LOC/DATE/NUM), dependency heads",
        "Sequence policy -> max_len N; keep salient entities intact; pad or truncate safely",
      ],
    },
  },
  {
    id: "phase1-data",
    title: "3) Phase 1 - Data Sources and Labeling Strategy",
    micro: "Diverse corpora plus exported user history provide scale and realism; hard rules inject domain priors.",
    bullets: [
      "Goal: tens of thousands of examples with calibrated confidences for BERT training.",
      "External sources: Wikipedia Q-sets, medical journals (hard-coded search), Stack Overflow programming Q and A.",
      "Exported user history: real queries with typos and entities; streamed extraction (CSV or JSON).",
      "Labeling rules: Medical -> always search; Math -> always no search. Misc reasoning intent often truncated to no search.",
    ],
    tables: [
      {
        caption: "Dataset Composition (approximate counts)",
        columns: ["Source", "Count", "Notes"],
        rows: [
          ["Wikipedia Q-sets", "~30,000", "Curated QA items"],
          ["Medical journals", "~16,000", "Rule-labeled: search"],
          ["Stack Overflow", "~16,000", "Programming intents"],
          ["Exported user history", "~280,000", "Real-world phrasing, typos"],
          ["Misc (math and reasoning)", "~6,000", "More reasoning intent; often labeled no search"],
        ],
      },
    ],
    pie: {
      data: [
        { label: "Wikipedia Q-sets", value: 30000 },
        { label: "Medical journals", value: 16000 },
        { label: "Stack Overflow", value: 16000 },
        { label: "Exported user history", value: 280000 },
        { label: "Misc (math and reasoning)", value: 6000 },
      ],
    },
  },
  {
    id: "labeling-llms",
    title: "3.2-3.5) Selecting LLMs for Data Labeling and Testing Results",
    micro: "Qwen2.5 0.5b instruct selected. Balanced latency and accuracy with schema adherence. Discrepancies reduced to 15/60 with prompt tweaks.",
    bullets: [
      "Models: Qwen2.5 0.5b instruct; Granite3.3 2b; Phi-4 mini 3.8B; Llama3.2 1b; Falcon 3.1b; Granite3.1-moe 1b.",
      "Testing: core prompt plus short variants; one vs few shot; sweep temperature/top-p/max-tokens/schema strictness/system prompt.",
      "Key finding: raw confidences inflated across models -> calibration required before use.",
      "Qwen2.5 discrepancies: 25/60 -> 15/60 after prompt and score adjustments.",
    ],
    tables: [
      {
        caption: "Summary of Findings (from report)",
        columns: [
          "Model",
          "Params",
          "Avg Latency (s)",
          "CPU Profile",
          "Confidence Behavior",
          "Avg Discrepancies",
          "Domain Strengths",
          "Domain Weaknesses",
        ],
        rows: [
          ["Qwen2.5 0.5b instruct", "0.5B", "~0.6", "Stable, low CPU", "Overconfident (0.9-1.0) but consistent", "~25", "Programming and general", "Math; some health edge cases"],
          ["Granite3.3 2b", "2.0B", "~2.2", "High CPU usage", "Inflated (~0.95)", "~21", "Math domain, stable", "Latency too high for CPU-only"],
          ["Phi-4 mini 3.8B", "3.8B", "~2.6", "Very high CPU", "Overconfident, clustered high", "~27", "Semantic reasoning, nuanced", "Prohibitively slow; memory demand"],
          ["Llama3.2 1b", "1.0B", "~1.2", "Moderate CPU", "Moderately overconfident", "~29", "Mental health queries", "Less consistent in math/programming"],
          ["Falcon 3.1b", "1.0B", "~1.5", "Spiky CPU demand", "Overconfident, unreliable", "~39", "-", "Highest discrepancies; schema issues"],
          ["Granite3.1-moe 1b", "1.0B", "~1.1", "Low CPU overhead", "Unstable, inconsistent", "~32", "Short factual queries (occasional)", "Unstable across domains"],
        ],
      },
    ],
    graphs: [
      { label: "Figure 1 - Avg Discrepancies per Model", filename: "discrepancies.png", src: "/images/discrepancies.png" },
      { label: "Figure 2 - Latency vs CPU Time", filename: "latency.png", src: "/images/latency.png" },
      { label: "Figure 3 - Avg Confidence vs Avg Discrepancies", filename: "confidence_vs_discrepancies.png", src: "/images/confidence.png" },
      { label: "Figure 4 - Domain-Level Discrepancy (stacked)", filename: "domain_discrepancy.png", src: "/images/domain_discrepancy.png" },
    ],
  },
  {
    id: "phase2-calibration",
    title: "4) Phase 2 - Multi-Stage Calibration Pipeline",
    micro: "Shrinkage plus down-only temperature scaling and guardrails -> calibrated confidences (about 0.71-0.73 average).",
    bullets: [
      "Shrinkage: per-domain gamma (Programming 2.9 -> about 0.64 avg; General 1.8 -> about 0.56).",
      "Temperature scaling (T >= 1): decreases toward 0.5; never increases raw.",
      "Guardrails: never-increase; preserve trivial 1.0; per-domain calibrators.json.",
    ],
    tables: [
      {
        caption: "Raw vs Calibrated Confidence (report)",
        columns: ["Domain", "Raw Avg", "Calibrated Avg"],
        rows: [
          ["Programming", "~0.85", "~0.732"],
          ["General", "~0.80", "~0.712"],
        ],
      },
    ],
  },
  {
    id: "phase3-scaling",
    title: "5) Phase 3 - Production-Grade Scaling (FastAPI + Docker)",
    micro: "Linear scaling, durable replication, quick failover. Telemetry surfaces bottlenecks on commodity hardware.",
    bullets: [
      "FastAPI cluster: sharding, leases, leader election, 2-ack replication, reconciliation, crash safety.",
      "Telemetry: per-node CPU/RAM/latency plus aggregated queue depth/throughput/backlog.",
      "Linux-first Docker: single base image, host-mounted datasets, Ollama REST on host; Raspberry Pi head-node option.",
    ],
    tables: [
      {
        caption: "Throughput and Reliability (from report/resume)",
        columns: ["Metric", "Value", "Notes"],
        rows: [
          ["Datapoints processed", "~327k", "~20 hours"],
          ["Leader failover", "<2s", "Lease-based"],
          ["Scaling", "Near-linear", "Replication later bottleneck"],
          ["Durability", "2x replica acks", "No single-node loss"],
        ],
      },
    ],
  },
  {
    id: "phase3-results",
    title: "5.3) Results - Dataset Quality Insights",
    micro: "Calibrated confidence tracks difficulty; domain stratification and length stats validate label quality.",
    bullets: [
      "Confidence distribution by domain aligns with expected difficulty after calibration.",
      "Domain-aware stratification highlights where search decisions cluster (e.g., medical).",
      "Text length by label: search trends longer/denser (entity-rich).",
    ],
    graphs: [
      { label: "Confidence Distribution by Domain", filename: "fig_conf_by_domain.png", src: "/images/fig_conf_by_domain.png" },
      { label: "Domain-Aware Stratification", filename: "fig_domain_stratification.png", src: "/images/fig_domain_stratification.png" },
      { label: "Text Length by Label", filename: "fig_text_length_by_label.png", src: "/images/fig_text_length_by_label.png" },
    ],
  },
  {
    id: "phase4-finetune",
    title: "6) Phase 4 - Fine-Tuning the BERT Classifier",
    micro: "Binary classifier hit about 90% in 1 epoch; next: confidence-aware training for graded decisions.",
    bullets: [
      "Model: BERT fine-tuned on calibrated labels; entity tags provide richer supervision.",
      "Result: about 90% accuracy (binary) in a single epoch.",
      "Roadmap: confidence-aware classification going forward.",
    ],
    fallback: {
      title: "Projected Path (ASCII)",
      ascii: [
        "Epochs: 1 -> 3 -> 5 (projected)",
        "Accuracy: 0.90 -> 0.92 -> 0.93 (if more data plus curriculum plus confidence-aware loss)",
      ],
    },
  },
  {
    id: "conclusion",
    title: "7) Conclusion and References",
    micro: "Qwen2.5 0.5b instruct selected; calibration solved overconfidence; distributed infra enabled practical scale.",
    bullets: [
      "Model selection: Qwen2.5 0.5b instruct; discrepancies reduced to 15/60 via prompt and score tweaks.",
      "Calibration: shrinkage plus temperature scaling plus guardrails produced trustworthy confidences for training.",
      "Infra: coordination, replication, and observability made it reliable at scale.",
    ],
  },
];

export default function Portfolio() {
  const fade = { hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } };
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  useEffect(() => {
    try {
      console.assert(Array.isArray(sections), "sections must be an array");
      console.assert(typeof data.tagline === "string" && data.tagline.length > 0, "data.tagline required");
      console.assert(sections.length > 0, "at least one section required");
      sections.forEach((s, idx) => {
        console.assert(Boolean(s.id), `section[${idx}] missing id`);
        console.assert(Boolean(s.title), `section[${idx}] missing title`);
        console.assert(Array.isArray(s.bullets), `section[${idx}] bullets must be array`);
        if (s.tables) {
          s.tables.forEach((t, ti) => {
            console.assert(Array.isArray(t.columns), `section[${idx}].tables[${ti}].columns must be array`);
            t.rows.forEach((r, ri) => {
              console.assert(r.length === t.columns.length, `tables[${ti}].rows[${ri}] length must equal columns length`);
            });
          });
        }
        if (s.pie) {
          const total = s.pie.data.reduce((sum, d) => sum + (d.value || 0), 0);
          console.assert(total > 0, `section[${idx}].pie total must be > 0`);
        }
      });
    } catch (e) {
      console.warn("Section validation warning", e);
    }
  }, []);

  const rootClass = "min-h-screen bg-gradient-to-br from-white to-gray-50 text-gray-900";
  const noteBox = "mt-4 text-xs text-gray-600 border rounded-lg p-3 bg-white/60";
  const microText = "text-gray-700";

  return (
    <div className={rootClass}>
      <section className="max-w-6xl mx-auto px-4 py-10">
        <motion.div initial="hidden" animate="show" variants={fade}>
          <h1 className="text-3xl md:text-5xl font-bold tracking-tight">{data.tagline}</h1>
          <p className={`mt-3 max-w-3xl ${microText}`}>{data.about}</p>
          <div className={noteBox}>
            <strong>Note:</strong> Sections mirror the original engineering report. For details, see the full PDF in the repo. {" "}
            {data.reportUrl !== "#" ? (
              <a className="underline" href={data.reportUrl} target="_blank" rel="noreferrer">Open report</a>
            ) : (
              <span className="opacity-70">(Set <code>data.reportUrl</code> to link your PDF)</span>
            )}
          </div>
          <div className="mt-6 flex flex-wrap gap-3 items-center">
            <IconLink href={`mailto:${data.email}`} icon={<Mail size={16} />}>Email</IconLink>
            <IconLink href={data.github} icon={<Github size={16} />}>GitHub</IconLink>
            <IconLink href={data.linkedin} icon={<Linkedin size={16} />}>LinkedIn</IconLink>
            <IconLink href={data.resumeUrl} icon={<FileText size={16} />}>Resume</IconLink>
          </div>
        </motion.div>
      </section>

      <section className="max-w-6xl mx-auto px-4 pb-6">
        <div className="grid grid-cols-1 gap-6">
          {sections.map((s) => (
            <Card key={s.id} className="rounded-2xl shadow-sm">
              <CardContent className="p-5">
                <div className="flex items-center gap-2 mb-2">
                  <LayoutTemplate size={18} />
                  <h2 className="text-lg font-semibold">{s.title}</h2>
                </div>
                {s.micro && <p className={`-mt-1 mb-2 text-sm ${microText}`}>{s.micro}</p>}
                <ul className="text-sm list-disc pl-5 space-y-1">
                  {s.bullets.map((b, i) => <li key={i}>{b}</li>)}
                </ul>

                {s.exec && (
                  <div className="mt-4 border rounded-lg p-4 bg-white/50">
                    <div className="text-sm font-semibold mb-2">Executive Summary</div>
                    {s.exec.intro && <p className={`text-sm ${microText}`}>{s.exec.intro}</p>}
                    {s.exec.points && (
                      <ol className="mt-2 text-sm list-decimal pl-5 space-y-1">
                        {s.exec.points.map((p: string, i: number) => <li key={i}>{p}</li>)}
                      </ol>
                    )}
                    {s.exec.outro && <p className={`mt-2 text-sm ${microText}`}>{s.exec.outro}</p>}
                  </div>
                )}

                {s.fallback && (
                  <div className="mt-3 text-xs font-mono whitespace-pre-wrap border rounded-lg p-3 bg-gray-50">
                    <div className="font-semibold mb-1">{s.fallback.title}</div>
                    {s.fallback.ascii.join("\n")}
                  </div>
                )}

                {s.tables && s.tables.length > 0 && (
                  <div className="mt-4 grid grid-cols-1 gap-4">
                    {s.tables.map((t: TableSpec, ti: number) => (
                      <div key={ti} className="overflow-x-auto border rounded-xl bg-white">
                        <div className="text-xs px-3 py-2 text-gray-600 border-b">{t.caption}</div>
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="bg-gray-50">
                              {t.columns.map((c: string, ci: number) => (
                                <th key={ci} className="text-left px-3 py-2 border-b">{c}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {t.rows.map((r: any[], ri: number) => (
                              <tr key={ri} className="odd:bg-white even:bg-gray-50">
                                {r.map((cell: any, ci: number) => (
                                  <td key={ci} className="px-3 py-2 align-top border-b">{cell}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ))}
                  </div>
                )}

                {s.pie && s.pie.data && (
                  <div className="mt-4 grid grid-cols-1 md:grid-cols-[220px,1fr] gap-4">
                    <div className="flex items-center justify-center">
                      <PieChart data={s.pie.data} size={200} />
                    </div>
                    <div>
                      <div className={`text-xs mb-2 ${microText}`}>Dataset composition (approximate)</div>
                      <ul className="text-sm grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {s.pie.data.map((d: any, i: number) => (
                          <li key={i} className="flex items-center gap-2">
                            <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: getPieColor(i) }} />
                            <span className="truncate">{d.label}</span>
                            <span className={`ml-auto tabular-nums ${microText}`}>{formatPct(d.value, s.pie.data)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {s.graphs && s.graphs.length > 0 && (
                  <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                    {s.graphs.map((g: any, i: number) => (
                      <figure key={i} className="border rounded-xl p-3 bg-white">
                        <div className={`text-xs mb-2 flex items-center gap-2 ${microText}`}><BarChart3 size={14} /> {g.label}</div>
                        <img
                          src={g.src}
                          alt={g.label}
                          loading="lazy"
                          className="rounded-md cursor-zoom-in"
                          onClick={(e) => setLightboxSrc((e.target as HTMLImageElement).src)}
                          onError={(e) => {
                            const el = e.target as HTMLImageElement;
                            el.alt = `Graph not found. Place an image at /public/images/${g.filename} or set g.src to a full URL.`;
                            el.style.opacity = "0.5";
                          }}
                        />
                        <figcaption className="mt-2 text-xs text-gray-500">
                          Drop file: <code className="bg-gray-50 px-1 py-0.5 rounded border">/public/images/{g.filename}</code>
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <footer className="max-w-6xl mx-auto px-4 py-10 text-sm text-gray-500">
        (c) {new Date().getFullYear()} {data.name}. Built for skim-speed; numbers up front.
      </footer>

      {lightboxSrc && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center" onClick={() => setLightboxSrc(null)}>
          <img src={lightboxSrc} className="max-h-[90vh] max-w-[92vw] rounded-lg" />
        </div>
      )}
    </div>
  );
}

function getPieColor(i: number): string {
  const COLORS = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#f87171", "#10b981", "#f59e0b"];
  return COLORS[i % COLORS.length];
}

function formatPct(value: number, data: { value: number }[]): string {
  const total = data.reduce((s, d) => s + (d.value || 0), 0) || 1;
  const pct = (value / total) * 100;
  if (pct > 1) return `${pct.toFixed(1)}%`;
  return `${pct.toFixed(2)}%`;
}

function PieChart({ data, size = 200 }: { data: { label: string; value: number }[]; size?: number }) {
  const total = data.reduce((s, d) => s + (d.value || 0), 0) || 1;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 2;
  let acc = 0;
  const arcs = data.map((d, i) => {
    const start = (acc / total) * Math.PI * 2 - Math.PI / 2;
    acc += d.value || 0;
    const end = (acc / total) * Math.PI * 2 - Math.PI / 2;
    const x1 = cx + r * Math.cos(start);
    const y1 = cy + r * Math.sin(start);
    const x2 = cx + r * Math.cos(end);
    const y2 = cy + r * Math.sin(end);
    const large = end - start > Math.PI ? 1 : 0;
    const dAttr = `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`;
    return <path key={i} d={dAttr} fill={getPieColor(i)} />;
  });
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="drop-shadow-sm">
      {arcs}
      <circle cx={cx} cy={cy} r={r} fill="transparent" stroke="rgba(0,0,0,0.04)" />
    </svg>
  );
}

function IconLink({ href, icon, children }: { href: string; icon: React.ReactNode; children: React.ReactNode }) {
  const cls = "inline-flex items-center gap-2 px-3 py-1.5 rounded-md border text-sm bg-gray-100 text-gray-900 border-gray-200 hover:bg-gray-200";
  return (
    <a href={href} target="_blank" rel="noreferrer" className={cls}>
      {icon}
      <span>{children}</span>
    </a>
  );
}
