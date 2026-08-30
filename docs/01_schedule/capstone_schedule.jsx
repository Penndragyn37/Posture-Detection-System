import { useState, Fragment } from "react";

const CURRENT_WEEK = 8;

const phases = [
  {
    id: "P1",
    level2: "Initiation & Planning",
    weeks: [1, 2],
    color: "#4361EE",
    light: "#EEF1FD",
    workPackages: [
      {
        id: "WP1.1",
        level3: "Project Setup",
        weeks: [1],
        activities: [
          {
            id: "A1.1.1",
            level4: "Finalize capstone proposal with advisor",
            week: 1,
            tasks: ["Confirm perception-only scope", "Get advisor signature on proposal form", "Upload to CEAS portal"],
          },
          {
            id: "A1.1.2",
            level4: "Configure development environment",
            week: 1,
            tasks: ["Install Python, OpenCV, MediaPipe", "Set up Ollama for local VLM inference", "Create project repo and folder structure"],
          },
        ],
      },
      {
        id: "WP1.2",
        level3: "Evaluation Framework Design",
        weeks: [2],
        activities: [
          {
            id: "A1.2.1",
            level4: "Define success metrics",
            week: 2,
            tasks: ["Accuracy, precision, recall per approach", "Latency measurement methodology", "Hardware resource monitoring plan"],
          },
          {
            id: "A1.2.2",
            level4: "Dataset protocol design",
            week: 2,
            tasks: ["Define posture classes: sitting, standing, absent", "Camera placement and distance standards", "Lighting and background variation plan"],
          },
        ],
      },
    ],
  },
  {
    id: "P2",
    level2: "Research & Architecture",
    weeks: [2, 3],
    color: "#7209B7",
    light: "#F3EEF9",
    workPackages: [
      {
        id: "WP2.0",
        level3: "System Requirements Specification",
        weeks: [2, 3],
        activities: [
          {
            id: "A2.0.1",
            level4: "Functional and non-functional requirements",
            week: 2,
            tasks: ["Define 36 requirements across FR and NFR groups (IEEE 830 / ISO 29148)", "Document safety invariant: desk never adjusts without visual countdown", "Specify performance, accuracy, privacy, and portability constraints per approach"],
          },
          {
            id: "A2.0.2",
            level4: "Interface requirements and traceability matrix",
            week: 3,
            tasks: ["Define user, hardware, software, and JSON communication interfaces", "Build requirements traceability matrix linking each FR/NFR to design document and test", "Baseline SRS as reference for architecture, FMEA, and final report"],
          },
        ],
      },
      {
        id: "WP2.1",
        level3: "Literature Review",
        weeks: [2, 3],
        activities: [
          {
            id: "A2.1.1",
            level4: "Survey pose estimation methods",
            week: 2,
            tasks: ["Review MediaPipe Pose documentation and benchmarks", "Compare BlazePose vs OpenPose vs MoveNet", "Note keypoint sets and accuracy trade-offs"],
          },
          {
            id: "A2.1.2",
            level4: "Survey VLM vision capabilities",
            week: 3,
            tasks: ["Review Claude vision API documentation", "Evaluate Qwen2-VL vs Gemma 3 benchmarks", "Assess prompt strategies for structured classification output"],
          },
        ],
      },
      {
        id: "WP2.2",
        level3: "System Architecture",
        weeks: [3],
        activities: [
          {
            id: "A2.2.1",
            level4: "Design unified system architecture",
            week: 3,
            tasks: ["Define shared input pipeline (frame capture)", "Specify JSON output schema for all approaches", "Design modular approach-swappable architecture"],
          },
          {
            id: "A2.2.2",
            level4: "Data collection execution",
            week: 3,
            tasks: ["Record standing/sitting/absent video clips across 5 lighting × 3 background × 3 distance conditions", "Label 270 frames (90 per class, ~6 per condition combination) with ground truth annotations", "Partition into 210-frame development set and 60-frame held-out test set — no training split needed (zero-shot VLM)"],
          },
        ],
      },
      {
        id: "WP2.3",
        level3: "Fault Tree Analysis (FTA)",
        weeks: [3],
        activities: [
          {
            id: "A2.3.1",
            level4: "FTA top-down safety analysis",
            week: 3,
            tasks: ["Define top undesired event: desk adjusts unexpectedly", "Map E1 branch (AND gate): safety invariant violated", "Map E2 branch (OR gate): countdown fires without prevention"],
          },
          {
            id: "A2.3.2",
            level4: "Minimum cut sets and WP2.3 document",
            week: 3,
            tasks: ["Identify 4 minimum cut sets (MCS-1 to MCS-4)", "Confirm MCS-1 near-impossible by structural code enforcement", "Cross-reference with WP3.0 FMEA: FTA top-down, FMEA bottom-up"],
          },
        ],
      },
    ],
  },
  {
    id: "P3",
    level2: "Implementation: Gemma 4 (Primary) + Claude API (PoC)",
    weeks: [4, 5, 6, 7],
    color: "#F77F00",
    light: "#FEF4E8",
    workPackages: [
      {
        id: "WP3.1",
        level3: "WP3.1  Gemma 4 Server Deployment (Primary)",
        weeks: [4, 5],
        activities: [
          {
            id: "A3.1.1",
            level4: "Gemma 4 26B MoE local deployment",
            week: 4,
            tasks: ["Install Ollama and pull gemma4:26b model", "Configure Q8_0 quantization for MoE routing accuracy", "Validate GPU availability and VRAM headroom (~10GB)"],
          },
          {
            id: "A3.1.2",
            level4: "Gemma 4 detection pipeline (server)",
            week: 5,
            tasks: ["Implement Gemma4Detector class extending PostureDetector ABC", "Set image token budget to 280 tokens via config", "Parse and validate JSON output schema on each response"],
          },
        ],
      },
      {
        id: "WP3.2",
        level3: "WP3.2  Claude API Proof of Concept (Low-End PC)",
        weeks: [5],
        activities: [
          {
            id: "A3.2.1",
            level4: "Low-end PC and webcam setup",
            week: 5,
            tasks: ["Confirm PC spec: any machine with webcam and internet connection", "Install Python, OpenCV, and anthropic SDK (no GPU required)", "Validate Claude API key and network connectivity from test machine"],
          },
          {
            id: "A3.2.2",
            level4: "Claude PoC pipeline and cloud risk assessment",
            week: 5,
            tasks: ["Implement ClaudeDetector using same PostureDetector interface as Gemma 4", "Test connectivity failure modes: offline, slow network, API rate limit, outage", "Record latency, per-frame cost, and accuracy vs Gemma 4 on same test set"],
          },
        ],
      },
      {
        id: "WP3.3",
        level3: "WP3.3  Gemma 4 Mobile Extension (E4B)",
        weeks: [5, 6],
        activities: [
          {
            id: "A3.3.1",
            level4: "E4B deployment on target device",
            week: 5,
            tasks: ["Deploy Gemma 4 E4B via LiteRT-LM on Android or iOS", "Confirm 8-15 tok/sec inference speed on device", "Validate memory footprint under 5GB device RAM"],
          },
          {
            id: "A3.3.2",
            level4: "Mobile pipeline integration",
            week: 6,
            tasks: ["Replicate server pipeline on mobile using same config structure", "Test 280 vs 560 image token budget on mobile hardware", "Compare accuracy and latency vs server variant on same test frames"],
          },
        ],
      },
      {
        id: "WP3.4",
        level3: "WP3.4  Integration and System Testing (Gemma 4 Focus)",
        weeks: [7],
        activities: [
          {
            id: "A3.4.1",
            level4: "Unified output interface validation",
            week: 7,
            tasks: ["Confirm server and mobile variants conform to shared JSON schema", "Validate evaluation logger captures all required fields", "Document output interface for desk actuation integration"],
          },
          {
            id: "A3.4.2",
            level4: "Gemma 4 robustness testing",
            week: 7,
            tasks: ["Test server and mobile variants on full labeled test set", "Validate under varied lighting, backgrounds, and distances", "Confirm graceful error handling on model timeout or malformed output"],
          },
        ],
      },
    ],
  },
  {
    id: "P4",
    level2: "Evaluation & Analysis",
    weeks: [6, 7, 8, 9],
    color: "#2D6A4F",
    light: "#EAF4EF",
    workPackages: [
      {
        id: "WP4.1",
        level3: "Benchmarking",
        weeks: [6, 7],
        activities: [
          {
            id: "A4.1.1",
            level4: "Accuracy benchmarking",
            week: 6,
            tasks: ["Claude API 99.4% (macro F1 0.994) — NFR-005/006 ✓", "Gemma 4 26B 98.5% (macro F1 0.987) — NFR-005/006 ✓. Fastest approach, meets NFR-001", "Gemma 4 E4B 96.4% (macro F1 0.967) — NFR-005/006 ✓. Near-camera Gemma4 pattern identified and documented"],
          },
          {
            id: "A4.1.2",
            level4: "Performance benchmarking",
            week: 6,
            tasks: ["Claude API 2,681ms mean (NFR-003 ≤3,000ms ✓); p95 3,447ms (NFR-003 ≤5,000ms ✓)", "Gemma 4 26B 1,855ms mean (NFR-001 ≤2,000ms ✓) — fastest; run on AWS g5.2xlarge, NVIDIA A10G, Ubuntu 24.04", "Gemma 4 E4B 5,593ms mean (NFR-002 ✗ — misses 3,000ms target); p95 6,385ms"],
          },
        ],
      },
      {
        id: "WP4.2",
        level3: "Comparative Analysis",
        weeks: [9],
        activities: [
          {
            id: "A4.2.1",
            level4: "Edge case and robustness testing",
            week: 9,
            tasks: ["Test under varied lighting conditions", "Test partial occlusion scenarios", "Test at different camera distances and angles"],
          },
          {
            id: "A4.2.2",
            level4: "Results synthesis",
            week: 9,
            tasks: ["Compare server vs mobile Gemma 4 variants across all metrics", "Identify optimal deployment configuration for smart desk scenario", "Draft deployment recommendation and note alternative paths available"],
          },
        ],
      },
    ],
  },
  {
    id: "P5",
    level2: "Documentation & Presentation",
    weeks: [8, 9, 10],
    color: "#C1121F",
    light: "#FAEAEB",
    workPackages: [
      {
        id: "WP5.0",
        level3: "Validation Plan (UAT)",
        weeks: [8, 9],
        activities: [
          {
            id: "A5.0.1",
            level4: "UAT scenario execution",
            week: 8,
            tasks: ["VAL-01: Natural standing transition at real desk", "VAL-02: Transient event (shoe-tying) ignored", "VAL-03: User cancels pending adjustment"],
          },
          {
            id: "A5.0.2",
            level4: "Validation outcomes and report",
            week: 9,
            tasks: ["VAL-04: Absent detection and desk hold", "Complete acceptance criteria table (Accept/Partial/Reject)", "Document verification vs validation outcomes in final report"],
          },
        ],
      },
      {
        id: "WP5.1",
        level3: "Final Report",
        weeks: [8, 9],
        activities: [
          {
            id: "A5.1.1",
            level4: "Draft report sections (concurrent with analysis)",
            week: 8,
            tasks: ["Write Abstract, Introduction, and Background", "Write Methods section covering all three approaches", "Include system architecture diagram"],
          },
          {
            id: "A5.1.2",
            level4: "Complete and finalize report",
            week: 9,
            tasks: ["Write Results, Discussion, and Conclusion", "Add bibliography and appendices (code, figures)", "Review against MEng report rubric and 10-page limit"],
          },
        ],
      },
      {
        id: "WP5.2",
        level3: "Presentation & Submission",
        weeks: [9, 10],
        activities: [
          {
            id: "A5.2.1",
            level4: "Build presentation slides",
            week: 9,
            tasks: ["Structure: problem, approaches, results, recommendation", "Include benchmark comparison charts and visuals", "Prepare live demo or recorded demo video"],
          },
          {
            id: "A5.2.2",
            level4: "Final submission",
            week: 10,
            tasks: ["Rehearse presentation (target 20 minutes)", "Advisor review and final edits", "Upload final report to CEAS Graduate Portal", "Submit all materials by July 22 deadline"],
          },
        ],
      },
    ],
  },
];

const weeks = [
  { num: 1, dates: "May 11-17" },
  { num: 2, dates: "May 18-24" },
  { num: 3, dates: "May 25-31" },
  { num: 4, dates: "Jun 1-7" },
  { num: 5, dates: "Jun 8-14" },
  { num: 6, dates: "Jun 15-21" },
  { num: 7, dates: "Jun 22-28" },
  { num: 8, dates: "Jun 29-Jul 5" },
  { num: 9, dates: "Jul 6-12" },
  { num: 10, dates: "Jul 13-22" },
];

export default function CapstoneSchedule() {
  const [expanded, setExpanded] = useState({ P1: true, P2: true, P3: true, P4: true, P5: true });
  const [expandedWP, setExpandedWP] = useState({ WP1_1: true, WP1_2: true, WP2_0: true, WP2_1: true, WP2_2: true, WP2_3: true, WP3_1: true, WP3_2: true, WP3_3: true, WP3_4: true, WP4_1: true, WP4_2: true, WP5_0: true, WP5_1: true, WP5_2: true });
  const [view, setView] = useState("wbs"); // gantt | wbs

  const togglePhase = (id) => setExpanded((p) => ({ ...p, [id]: !p[id] }));
  const toggleWP = (id) => setExpandedWP((p) => ({ ...p, [id]: !p[id] }));

  return (
    <div style={{ fontFamily: "'Inter', 'Segoe UI', sans-serif", background: "#F7F8FA", minHeight: "100vh", padding: "0 0 48px 0" }}>
      {/* Header */}
      <div style={{ background: "#0F1923", color: "#fff", padding: "28px 32px 22px" }}>
        <div style={{ fontSize: 11, letterSpacing: 3, color: "#7B8FA1", textTransform: "uppercase", marginBottom: 6 }}>MEng Capstone · Robotics & Intelligent Autonomous Systems</div>
        <div style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>Posture Detection System</div>
        <div style={{ fontSize: 13, color: "#A0B0C0" }}>May 11 – July 22, 2026 &nbsp;·&nbsp; 10 Weeks &nbsp;·&nbsp; 3 Credit Hours</div>
        <div style={{ display: "flex", gap: 12, marginTop: 18, flexWrap: "wrap" }}>
          {[
            { label: "L1 Project", color: "#4361EE" },
            { label: "L2 Phase", color: "#7209B7" },
            { label: "L3 Work Package", color: "#F77F00" },
            { label: "L4 Activity", color: "#2D6A4F" },
            { label: "L5 Task", color: "#C1121F" },
          ].map((l) => (
            <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#A0B0C0" }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: l.color }} />
              {l.label}
            </div>
          ))}
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            {["gantt", "wbs"].map((v) => (
              <button key={v} onClick={() => setView(v)} style={{ padding: "4px 14px", borderRadius: 6, border: "1px solid #2A3A4A", background: view === v ? "#4361EE" : "transparent", color: view === v ? "#fff" : "#7B8FA1", fontSize: 12, cursor: "pointer", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>
                {v === "gantt" ? "Gantt" : "WBS"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Current week banner */}
      <div style={{ background: "#FFF8E7", borderBottom: "1px solid #FFE082", padding: "10px 32px", display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}>
        <span style={{ background: "#F77F00", color: "#fff", borderRadius: 4, padding: "2px 8px", fontWeight: 700, fontSize: 11 }}>CURRENT</span>
        <span style={{ color: "#6D4C00" }}>Week 7 complete (Jun 23–29). All three approaches benchmarked: Claude 99.4% (NFR-003 ✓), Gemma 4 26B 98.5% (NFR-001 ✓), Gemma 4 E4B 96.4% (NFR-002 ✗ latency). T1–T8 reliability all PASS. VAL-01–04 UAT all Accept. Final Report v6 delivered. Code rev14 (CAM-01/02/03). <strong>Week 8 (Jun 30–Jul 6): final report polish + presentation deck</strong></span>
      </div>

      <div style={{ padding: "24px 32px" }}>
        {/* Level 1 Banner */}
        <div style={{ background: "#0F1923", color: "#fff", borderRadius: 10, padding: "14px 20px", marginBottom: 20, display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ background: "#4361EE", color: "#fff", borderRadius: 6, padding: "4px 10px", fontSize: 11, fontWeight: 700, letterSpacing: 1 }}>LEVEL 1</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15 }}>Camera-Based Posture Detection System — Comparative Evaluation</div>
            <div style={{ fontSize: 12, color: "#7B8FA1", marginTop: 2 }}>Gemma 4 (Primary) vs. Claude API (PoC) · Final Report & Presentation due July 22</div>
          </div>
        </div>

        {view === "gantt" ? (
          <GanttView phases={phases} weeks={weeks} currentWeek={CURRENT_WEEK} expanded={expanded} togglePhase={togglePhase} />
        ) : (
          <WBSView phases={phases} expanded={expanded} expandedWP={expandedWP} togglePhase={togglePhase} toggleWP={toggleWP} />
        )}
      </div>
    </div>
  );
}

function GanttView({ phases, weeks, currentWeek, expanded, togglePhase }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 900 }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "8px 12px", fontSize: 11, color: "#7B8FA1", fontWeight: 600, width: 240, borderBottom: "2px solid #E2E8F0" }}>LEVEL 2–3 / WORK PACKAGE</th>
            {weeks.map((w) => (
              <th key={w.num} style={{ padding: "6px 4px", fontSize: 10, color: w.num === currentWeek ? "#F77F00" : "#7B8FA1", fontWeight: w.num === currentWeek ? 700 : 500, textAlign: "center", borderBottom: `2px solid ${w.num === currentWeek ? "#F77F00" : "#E2E8F0"}`, minWidth: 64, background: w.num === currentWeek ? "#FFF8E7" : "transparent" }}>
                <div>W{w.num}</div>
                <div style={{ fontSize: 9, fontWeight: 400 }}>{w.dates}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {phases.map((phase) => (
            <Fragment key={phase.id}>
              {/* Phase row — L2 */}
              <tr onClick={() => togglePhase(phase.id)} style={{ cursor: "pointer" }}>
                <td style={{ padding: "10px 12px", background: phase.color, color: "#fff", borderRadius: "6px 0 0 6px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 11, opacity: 0.7 }}>L2</span>
                    <span style={{ fontWeight: 700, fontSize: 13 }}>{phase.level2}</span>
                    <span style={{ marginLeft: "auto", fontSize: 14 }}>{expanded[phase.id] ? "▾" : "▸"}</span>
                  </div>
                </td>
                {weeks.map((w) => {
                  const active = phase.weeks.includes(w.num);
                  return (
                    <td key={w.num} style={{ background: w.num === currentWeek ? "#FFF8E7" : "#fff", padding: "4px", textAlign: "center", borderBottom: "1px solid #F0F0F0" }}>
                      {active && <div style={{ background: phase.color, borderRadius: 4, height: 18, opacity: 0.85 }} />}
                    </td>
                  );
                })}
              </tr>

              {/* Work Package rows — L3 */}
              {expanded[phase.id] &&
                phase.workPackages.map((wp) => (
                  <Fragment key={wp.id}>
                    <tr>
                      <td style={{ padding: "7px 12px 7px 28px", background: phase.light, fontSize: 12, color: "#333", borderLeft: `3px solid ${phase.color}` }}>
                        <span style={{ fontSize: 10, color: phase.color, fontWeight: 700, marginRight: 6 }}>L3</span>
                        {wp.level3}
                      </td>
                      {weeks.map((w) => {
                        const active = wp.weeks.includes(w.num);
                        return (
                          <td key={w.num} style={{ background: w.num === currentWeek ? "#FFF8E7" : phase.light, padding: "4px", textAlign: "center", borderBottom: "1px solid #F0F0F0" }}>
                            {active && <div style={{ background: phase.color, borderRadius: 3, height: 12, opacity: 0.45 }} />}
                          </td>
                        );
                      })}
                    </tr>

                    {/* Activity rows — L4 */}
                    {wp.activities.map((act) => (
                      <Fragment key={act.id}>
                        <tr>
                          <td style={{ padding: "5px 12px 5px 44px", background: "#fff", fontSize: 11, color: "#555", borderLeft: `3px solid ${phase.color}` }}>
                            <span style={{ fontSize: 9, color: "#999", fontWeight: 700, marginRight: 6 }}>L4</span>
                            {act.level4}
                          </td>
                          {weeks.map((w) => {
                            const active = act.week === w.num;
                            return (
                              <td key={w.num} style={{ background: w.num === currentWeek ? "#FFF8E7" : "#fff", padding: "3px 4px", borderBottom: "1px solid #F7F7F7" }}>
                                {active && <div style={{ background: phase.color, borderRadius: 2, height: 8, opacity: 0.3, margin: "0 4px" }} />}
                              </td>
                            );
                          })}
                        </tr>

                        {/* L5 Task rows */}
                        {act.tasks.map((task, ti) => (
                          <tr key={ti}>
                            <td style={{ padding: "3px 12px 3px 64px", background: "#fff", fontSize: 10, color: "#888", borderLeft: `3px solid ${phase.color}`, fontStyle: "italic" }}>
                              <span style={{ color: phase.color, marginRight: 5 }}>→</span>{task}
                            </td>
                            {weeks.map((w) => (
                              <td key={w.num} style={{ background: w.num === currentWeek ? "#FFF8E7" : "#fff", padding: "2px 4px", borderBottom: "1px solid #FAFAFA" }}>
                                {act.week === w.num && <div style={{ background: phase.color, borderRadius: 1, height: 4, opacity: 0.15, margin: "0 6px" }} />}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </Fragment>
                    ))}
                  </Fragment>
                ))}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WBSView({ phases, expanded, expandedWP, togglePhase, toggleWP }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {phases.map((phase) => (
        <div key={phase.id} style={{ borderRadius: 10, overflow: "hidden", border: `1px solid ${phase.color}22` }}>
          {/* L2 Phase header */}
          <div onClick={() => togglePhase(phase.id)} style={{ background: phase.color, color: "#fff", padding: "12px 18px", cursor: "pointer", display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 10, background: "rgba(255,255,255,0.25)", borderRadius: 4, padding: "2px 7px", fontWeight: 700, letterSpacing: 1 }}>LEVEL 2</span>
            <span style={{ fontWeight: 700, fontSize: 14, flex: 1 }}>{phase.level2}</span>
            <span style={{ fontSize: 12, opacity: 0.8 }}>Weeks {phase.weeks[0]}{phase.weeks.length > 1 ? `–${phase.weeks[phase.weeks.length - 1]}` : ""}</span>
            <span style={{ fontSize: 16 }}>{expanded[phase.id] ? "▾" : "▸"}</span>
          </div>

          {expanded[phase.id] && (
            <div style={{ background: "#fff" }}>
              {phase.workPackages.map((wp) => (
                <div key={wp.id} style={{ borderTop: `1px solid ${phase.light}` }}>
                  {/* L3 Work Package */}
                  <div onClick={() => toggleWP(wp.id)} style={{ background: phase.light, padding: "9px 18px 9px 32px", cursor: "pointer", display: "flex", alignItems: "center", gap: 10, borderLeft: `4px solid ${phase.color}` }}>
                    <span style={{ fontSize: 10, color: phase.color, fontWeight: 700, letterSpacing: 1 }}>LEVEL 3</span>
                    <span style={{ fontWeight: 600, fontSize: 13, color: "#222", flex: 1 }}>{wp.level3}</span>
                    <span style={{ fontSize: 11, color: "#888" }}>W{wp.weeks.join("–")}</span>
                    <span style={{ fontSize: 14, color: "#aaa" }}>{expandedWP[wp.id] ? "▾" : "▸"}</span>
                  </div>

                  {expandedWP[wp.id] &&
                    wp.activities.map((act) => (
                      <div key={act.id} style={{ borderLeft: `4px solid ${phase.color}` }}>
                        {/* L4 Activity */}
                        <div style={{ padding: "8px 18px 8px 52px", display: "flex", alignItems: "center", gap: 10, background: "#FAFAFA", borderTop: "1px solid #F0F0F0" }}>
                          <span style={{ fontSize: 9, color: phase.color, fontWeight: 700, letterSpacing: 1, background: phase.light, padding: "2px 6px", borderRadius: 3 }}>LEVEL 4</span>
                          <span style={{ fontSize: 12, color: "#333", flex: 1, fontWeight: 600 }}>{act.level4}</span>
                          <span style={{ fontSize: 11, color: "#aaa", background: "#F0F0F0", padding: "2px 8px", borderRadius: 10 }}>Week {act.week}</span>
                        </div>

                        {/* L5 Tasks — always visible */}
                        <div style={{ background: "#fff", padding: "8px 18px 12px 72px", borderTop: "1px solid #F5F5F5" }}>
                          <div style={{ fontSize: 9, color: "#bbb", fontWeight: 700, letterSpacing: 1.5, marginBottom: 6, textTransform: "uppercase" }}>Level 5 — Tasks</div>
                          {act.tasks.map((task, i) => (
                            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "5px 0", fontSize: 12, color: "#444", borderBottom: i < act.tasks.length - 1 ? "1px dashed #F0F0F0" : "none" }}>
                              <span style={{ color: phase.color, fontWeight: 700, marginTop: 1, flexShrink: 0 }}>→</span>
                              {task}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
