import { useEffect, useState, useRef, useCallback } from "react";
import WaveSurfer from "wavesurfer.js";
import {
  getStemLibrary,
  stackShuffle,
  stackPreview,
  stackExportUrl,
  submitTransformFeedback,
  type StemLibraryEntry,
  type StackPreviewResponse,
} from "../api";

const CATEGORIES = ["drums", "bass", "vocals", "piano", "synth", "other"] as const;
type Category = (typeof CATEGORIES)[number];

const CATEGORY_LABELS: Record<Category, string> = {
  drums: "Drums",
  bass: "Bass",
  vocals: "Vocals",
  piano: "Piano",
  synth: "Synth",
  other: "Other",
};

const KEYS = [
  "C major", "C minor", "C# major", "C# minor",
  "D major", "D minor", "D# major", "D# minor",
  "E major", "E minor",
  "F major", "F minor", "F# major", "F# minor",
  "G major", "G minor", "G# major", "G# minor",
  "A major", "A minor", "A# major", "A# minor",
  "B major", "B minor",
];

const HP_MIN = 20;
const HP_MAX = 2000;
const LP_MIN = 200;
const LP_MAX = 20000;
const fmtFreq = (hz: number) =>
  hz >= 1000 ? `${(hz / 1000).toFixed(1)}k` : `${Math.round(hz)}`;

export default function Stacker() {
  const [library, setLibrary] = useState<StemLibraryEntry[]>([]);
  const [slots, setSlots] = useState<Record<string, string>>({});
  const [slotBackends, setSlotBackends] = useState<Record<string, string>>({});
  const [slotBpms, setSlotBpms] = useState<Record<string, number>>({});
  // 0 = full track, 4/8/16/32 = loop length in bars
  const [slotLoops, setSlotLoops] = useState<Record<string, number>>({});
  const [targetBpm, setTargetBpm] = useState(128);
  const [targetKey, setTargetKey] = useState("A minor");
  const [preview, setPreview] = useState<StackPreviewResponse | null>(null);
  // Per-slot transform vote: "good" | "bad" once submitted for the current preview
  const [transformVotes, setTransformVotes] = useState<Record<string, "good" | "bad">>({});
  const [transformBusy, setTransformBusy] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [volumes, setVolumes] = useState<Record<string, number>>({});   // 0–100
  const [soloCategory, setSoloCategory] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  // Filters: hp = high-pass cutoff Hz (20=off), lp = low-pass cutoff Hz (20000=off)
  type FilterState = { hp: number; lp: number };
  const [slotFilters, setSlotFilters] = useState<Record<string, FilterState>>({});

  const mixWsRef = useRef<WaveSurfer | null>(null);
  const mixContainerRef = useRef<HTMLDivElement>(null);
  const stemWsRefs = useRef<Record<string, WaveSurfer>>({});
  const stemContainerRefs = useRef<Record<string, HTMLDivElement | null>>({});

  type AudioNodes = { ctx: AudioContext; hpf: BiquadFilterNode; lpf: BiquadFilterNode };
  const audioNodesRef = useRef<Record<string, AudioNodes>>({});

  useEffect(() => {
    getStemLibrary()
      .then((entries) => {
        setLibrary(entries);
        if (entries.length > 0 && !entries.some((e) => Object.values(slots).includes(e.name))) {
          handleShuffle(entries);
        }
      })
      .catch(() => {});
  }, []);

  const handleShuffle = useCallback(async (_lib?: StemLibraryEntry[]) => {
    try {
      setError(null);
      const resp = await stackShuffle();
      const newSlots: Record<string, string> = {};
      for (const [cat, info] of Object.entries(resp.slots)) {
        newSlots[cat] = info.name;
      }
      setSlots(newSlots);

      const first = Object.values(resp.slots)[0];
      if (first?.bpm) setTargetBpm(Math.round(first.bpm));
      if (first?.key) setTargetKey(first.key);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const handleSlotChange = (cat: string, name: string) => {
    setSlots((prev) => ({ ...prev, [cat]: name }));
    // Pre-fill BPM override with the song's detected BPM
    const entry = library.find((e) => e.name === name);
    if (entry?.bpm) {
      setSlotBpms((prev) => ({ ...prev, [cat]: entry.bpm! }));
    }
  };

  const handleSlotShuffle = (cat: string) => {
    if (library.length === 0) return;
    const current = slots[cat];
    const pool = library.filter((e) => e.name !== current);
    const pick = pool.length > 0 ? pool[Math.floor(Math.random() * pool.length)] : library[0];
    setSlots((prev) => ({ ...prev, [cat]: pick.name }));
    if (pick.bpm) setSlotBpms((prev) => ({ ...prev, [cat]: pick.bpm! }));
  };

  // Refs so the auto-build effect always calls with latest state
  const latestBuildArgs = useRef({ slots, targetBpm, targetKey, slotBackends, slotBpms, slotLoops });
  useEffect(() => {
    latestBuildArgs.current = { slots, targetBpm, targetKey, slotBackends, slotBpms, slotLoops };
  });

  const handlePreview = async (keepExisting = false) => {
    const args = latestBuildArgs.current;
    const filled = Object.entries(args.slots).filter(([, v]) => v);
    if (filled.length === 0) return;

    setBuilding(true);
    setError(null);
    // keepExisting=true: leave old waveforms visible while rebuilding (auto-trigger)
    if (!keepExisting) {
      setPreview(null);
      destroyAllWavesurfers();
    }

    try {
      const loopArg = Object.keys(args.slotLoops).length ? args.slotLoops : undefined;
      const resp = await stackPreview(
        args.slots, args.targetBpm, args.targetKey,
        args.slotBackends, args.slotBpms, loopArg,
      );
      setPreview(resp);
      setTransformVotes({});
    } catch (e) {
      setError(String(e));
    } finally {
      setBuilding(false);
    }
  };

  const handleTransformVote = async (cat: string, verdict: "good" | "bad") => {
    if (!preview) return;
    const info = preview.slots_info[cat];
    if (!info) return;
    setTransformBusy(cat);
    try {
      await submitTransformFeedback({
        category: cat,
        verdict,
        semitones: info.semitones ?? null,
        stretch_ratio: info.stretch_ratio ?? null,
        song: info.name,
        stack_id: preview.stack_id,
      });
      setTransformVotes((prev) => ({ ...prev, [cat]: verdict }));
    } catch (e) {
      setError(String(e));
    } finally {
      setTransformBusy(null);
    }
  };

  // Auto-rebuild when slots or loop lengths change (500ms debounce)
  const autoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hasMountedRef = useRef(false);
  useEffect(() => {
    if (!hasMountedRef.current) { hasMountedRef.current = true; return; }
    if (Object.keys(slots).length === 0) return;

    if (autoTimerRef.current) clearTimeout(autoTimerRef.current);
    autoTimerRef.current = setTimeout(() => handlePreview(true), 500);
    return () => {
      if (autoTimerRef.current) clearTimeout(autoTimerRef.current);
    };
  }, [slots, slotLoops]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleExport = async () => {
    const filled = Object.entries(slots).filter(([, v]) => v);
    if (filled.length === 0) return;

    setExporting(true);
    setError(null);
    try {
      const loopArgExp = Object.keys(slotLoops).length ? slotLoops : undefined;
      const blob = await stackExportUrl(slots, targetBpm, targetKey, slotBackends, slotBpms, loopArgExp);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `stack_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(String(e));
    } finally {
      setExporting(false);
    }
  };

  const destroyAllWavesurfers = () => {
    mixWsRef.current?.destroy();
    mixWsRef.current = null;
    for (const ws of Object.values(stemWsRefs.current)) {
      ws.destroy();
    }
    stemWsRefs.current = {};
    for (const nodes of Object.values(audioNodesRef.current)) {
      nodes.ctx.close().catch(() => {});
    }
    audioNodesRef.current = {};
    setPlaying(false);
  };

  // Create wavesurfer instances when preview arrives
  useEffect(() => {
    if (!preview) return;

    // Mix waveform — visual transport only, audio comes from per-stem players
    if (mixContainerRef.current) {
      const ws = WaveSurfer.create({
        container: mixContainerRef.current,
        waveColor: "#4a5568",
        progressColor: "#667eea",
        cursorColor: "#e53e3e",
        cursorWidth: 2,
        height: 64,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        interact: true,
        url: preview.mix_url,
      });
      ws.setVolume(0);
      ws.on("play", () => setPlaying(true));
      ws.on("pause", () => setPlaying(false));
      ws.on("finish", () => setPlaying(false));
      mixWsRef.current = ws;
    }

    // Per-stem waveforms + Web Audio filter chain
    for (const [cat, url] of Object.entries(preview.stem_urls)) {
      const container = stemContainerRefs.current[cat];
      if (!container) continue;

      const ws = WaveSurfer.create({
        container,
        waveColor: "#3a3d48",
        progressColor: "#667eea",
        cursorColor: "#e53e3e",
        cursorWidth: 1,
        height: 36,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        interact: false,
        url,
      });

      // Build HPF → LPF chain once the media element is available
      ws.on("ready", () => {
        const el = ws.getMediaElement();
        if (!el || audioNodesRef.current[cat]) return;
        try {
          const ctx = new AudioContext();
          const src = ctx.createMediaElementSource(el);
          const hpf = ctx.createBiquadFilter();
          hpf.type = "highpass";
          hpf.frequency.value = slotFilters[cat]?.hp ?? 20;
          hpf.Q.value = 0.5;
          const lpf = ctx.createBiquadFilter();
          lpf.type = "lowpass";
          lpf.frequency.value = slotFilters[cat]?.lp ?? 20000;
          lpf.Q.value = 0.5;
          src.connect(hpf);
          hpf.connect(lpf);
          lpf.connect(ctx.destination);
          audioNodesRef.current[cat] = { ctx, hpf, lpf };
        } catch {
          // Fallback: no filter (e.g. if media element already captured)
        }
      });

      stemWsRefs.current[cat] = ws;
    }

    return () => {
      destroyAllWavesurfers();
      setSlotFilters({});
    };
  }, [preview]);

  // Sync stem playback with mix transport
  useEffect(() => {
    const mix = mixWsRef.current;
    if (!mix) return;

    const syncPositions = () => {
      const time = mix.getCurrentTime();
      for (const [, ws] of Object.entries(stemWsRefs.current)) {
        const wsDur = ws.getDuration();
        if (wsDur <= 0) continue;
        ws.seekTo(Math.min(time / wsDur, 1));
      }
    };

    const onPlay = () => {
      syncPositions();
      for (const [cat, ws] of Object.entries(stemWsRefs.current)) {
        const vol = soloCategory
          ? (cat === soloCategory ? (volumes[cat] ?? 100) / 100 : 0)
          : (volumes[cat] ?? 100) / 100;
        ws.setVolume(vol);
        ws.play();
      }
    };

    const onPause = () => {
      for (const ws of Object.values(stemWsRefs.current)) {
        ws.pause();
      }
    };

    const onSeeking = () => syncPositions();

    const onFinish = () => {
      for (const ws of Object.values(stemWsRefs.current)) {
        ws.pause();
      }
    };

    mix.on("play", onPlay);
    mix.on("pause", onPause);
    mix.on("seeking", onSeeking);
    mix.on("finish", onFinish);

    return () => {
      mix.un("play", onPlay);
      mix.un("pause", onPause);
      mix.un("seeking", onSeeking);
      mix.un("finish", onFinish);
    };
  }, [preview, volumes, soloCategory]);

  // Update stem volumes live when slider or solo changes
  useEffect(() => {
    for (const [cat, ws] of Object.entries(stemWsRefs.current)) {
      const vol = soloCategory
        ? (cat === soloCategory ? (volumes[cat] ?? 100) / 100 : 0)
        : (volumes[cat] ?? 100) / 100;
      ws.setVolume(vol);
    }
  }, [volumes, soloCategory]);

  // Push filter changes live to Web Audio nodes
  useEffect(() => {
    for (const [cat, nodes] of Object.entries(audioNodesRef.current)) {
      const f = slotFilters[cat];
      if (!f) continue;
      nodes.hpf.frequency.setTargetAtTime(f.hp, nodes.ctx.currentTime, 0.01);
      nodes.lpf.frequency.setTargetAtTime(f.lp, nodes.ctx.currentTime, 0.01);
    }
  }, [slotFilters]);

  const handleVolume = (cat: string, val: number) => {
    setVolumes((prev) => ({ ...prev, [cat]: val }));
  };

  const handleFilter = (cat: string, type: "hp" | "lp", val: number) => {
    setSlotFilters((prev) => {
      const cur: FilterState = prev[cat] ?? { hp: HP_MIN, lp: LP_MAX };
      return { ...prev, [cat]: { ...cur, [type]: val } };
    });
  };


  const toggleSolo = (cat: string) => {
    setSoloCategory((prev) => (prev === cat ? null : cat));
  };

  const handlePlayPause = () => {
    const ws = mixWsRef.current;
    if (!ws) return;
    ws.playPause();
  };

  const getEntryForSlot = (cat: string): StemLibraryEntry | undefined => {
    return library.find((e) => e.name === slots[cat]);
  };

  const getBackendsForSlot = (cat: string): string[] => {
    const entry = getEntryForSlot(cat);
    if (!entry?.backends) return [];
    return Object.keys(entry.backends).filter(
      (b) => entry.backends[b]?.stems?.includes(cat) ?? false,
    );
  };

  const handleBackendToggle = (cat: string, backend: string) => {
    setSlotBackends((prev) => ({ ...prev, [cat]: backend }));
  };

  const filledCount = Object.values(slots).filter(Boolean).length;

  return (
    <div className="stacker">
      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      <div className="stacker-slots">
        {CATEGORIES.map((cat) => {
          const entry = getEntryForSlot(cat);
          const backends = getBackendsForSlot(cat);
          const currentBackend = slotBackends[cat] || (backends.includes("lalal") ? "lalal" : backends[0] || "");
          return (
            <div key={cat} className="stacker-slot">
              <div className="slot-header">
                <span className="slot-label">{CATEGORY_LABELS[cat]}</span>
                <button
                  className="btn-icon"
                  onClick={() => handleSlotShuffle(cat)}
                  title={`Shuffle ${cat}`}
                >
                  ↻
                </button>
              </div>
              <select
                className="slot-select"
                value={slots[cat] || ""}
                onChange={(e) => handleSlotChange(cat, e.target.value)}
              >
                <option value="">— none —</option>
                {library.map((e) => (
                  <option key={e.name} value={e.name}>
                    {e.name}
                  </option>
                ))}
              </select>
              {entry && (
                <>
                  <div className="slot-meta-row">
                    <label className="slot-bpm-label">
                      BPM
                      <input
                        type="number"
                        className="slot-bpm-input"
                        step="0.001"
                        min="40"
                        max="300"
                        value={slotBpms[cat] ?? entry.bpm ?? ""}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          if (!isNaN(v)) setSlotBpms((prev) => ({ ...prev, [cat]: v }));
                        }}
                        title="Source BPM — edit to fix detection errors"
                      />
                    </label>
                    <span className="slot-key">
                      {entry.key || "—"}{entry.camelot ? ` (${entry.camelot})` : ""}
                    </span>
                  </div>
                  <div className="slot-loop-row">
                    <span className="slot-loop-label">Loop</span>
                    {([0, 4, 8, 16, 32] as const).map((bars) => (
                      <button
                        key={bars}
                        className={`btn-loop ${(slotLoops[cat] ?? 0) === bars ? "active" : ""}`}
                        onClick={() =>
                          setSlotLoops((prev) => ({ ...prev, [cat]: bars }))
                        }
                        title={bars === 0 ? "Full track" : `${bars}-bar loop`}
                      >
                        {bars === 0 ? "Full" : `${bars}`}
                      </button>
                    ))}
                    {(slotLoops[cat] ?? 0) > 0 && preview?.slots_info?.[cat]?.loop_start_sec != null && (
                      <span className="slot-loop-pos" title="Loop start position">
                        @{preview.slots_info[cat].loop_start_sec!.toFixed(1)}s
                      </span>
                    )}
                  </div>
                </>
              )}
              {backends.length > 1 && (
                <div className="slot-backend-toggle">
                  {backends.map((b) => (
                    <button
                      key={b}
                      className={`btn-backend ${currentBackend === b ? "active" : ""}`}
                      onClick={() => handleBackendToggle(cat, b)}
                    >
                      {b === "lalal" ? "Lalal" : "Demucs"}
                    </button>
                  ))}
                </div>
              )}
              {preview && preview.stem_urls[cat] && (
                <div className="slot-waveform">
                  <div className="slot-waveform-actions">
                    <button
                      className={`btn-solo ${soloCategory === cat ? "active" : ""}`}
                      onClick={() => toggleSolo(cat)}
                      title="Solo"
                    >
                      S
                    </button>
                    <span className="slot-vol-pct">
                      {Math.round(volumes[cat] ?? 100)}
                    </span>
                  </div>
                  <div className="slot-waveform-right">
                    <div
                      className="slot-waveform-container"
                      ref={(el) => { stemContainerRefs.current[cat] = el; }}
                    />
                    <input
                      type="range"
                      className="slot-vol-slider"
                      min={0}
                      max={100}
                      value={volumes[cat] ?? 100}
                      onChange={(e) => handleVolume(cat, Number(e.target.value))}
                      title={`Volume: ${Math.round(volumes[cat] ?? 100)}%`}
                    />
                    <div className="slot-filters">
                      <label className="slot-filter-label">HP</label>
                      <input
                        type="range"
                        className="slot-filter-slider hp-slider"
                        min={HP_MIN}
                        max={HP_MAX}
                        step={10}
                        value={slotFilters[cat]?.hp ?? HP_MIN}
                        onChange={(e) => handleFilter(cat, "hp", Number(e.target.value))}
                        title={`High-pass: ${fmtFreq(slotFilters[cat]?.hp ?? HP_MIN)} Hz`}
                      />
                      <span className="slot-filter-val">
                        {(slotFilters[cat]?.hp ?? HP_MIN) <= HP_MIN
                          ? "off"
                          : `${fmtFreq(slotFilters[cat]?.hp ?? HP_MIN)}Hz`}
                      </span>
                      <label className="slot-filter-label">LP</label>
                      <input
                        type="range"
                        className="slot-filter-slider lp-slider"
                        min={LP_MIN}
                        max={LP_MAX}
                        step={100}
                        value={slotFilters[cat]?.lp ?? LP_MAX}
                        onChange={(e) => handleFilter(cat, "lp", Number(e.target.value))}
                        title={`Low-pass: ${fmtFreq(slotFilters[cat]?.lp ?? LP_MAX)} Hz`}
                      />
                      <span className="slot-filter-val">
                        {(slotFilters[cat]?.lp ?? LP_MAX) >= LP_MAX
                          ? "off"
                          : `${fmtFreq(slotFilters[cat]?.lp ?? LP_MAX)}Hz`}
                      </span>
                    </div>
                    {(() => {
                      const info = preview.slots_info[cat];
                      if (!info) return null;
                      const semi = info.semitones ?? 0;
                      const ratio = info.stretch_ratio ?? 1;
                      const shifted = Math.abs(semi) > 0 || Math.abs(ratio - 1) > 0.001;
                      if (!shifted && !info.warning) return null;
                      const voted = transformVotes[cat];
                      return (
                        <div className="slot-transform-feedback">
                          <span className="slot-transform-meta">
                            {semi !== 0 && `${semi > 0 ? "+" : ""}${semi} st`}
                            {semi !== 0 && Math.abs(ratio - 1) > 0.001 && " · "}
                            {Math.abs(ratio - 1) > 0.001 && `${ratio.toFixed(3)}×`}
                          </span>
                          {info.warning && (
                            <span className="slot-transform-warn" title={info.warning}>
                              past your limit
                            </span>
                          )}
                          {voted ? (
                            <span className={`slot-vote-done vote-${voted}`}>
                              {voted === "good" ? "clean" : "artifacts"}
                            </span>
                          ) : (
                            <span className="slot-vote-actions">
                              <button
                                className="btn-vote-mini btn-vote-good"
                                disabled={transformBusy === cat}
                                onClick={() => handleTransformVote(cat, "good")}
                                title="Sounds clean"
                              >
                                Clean
                              </button>
                              <button
                                className="btn-vote-mini btn-vote-bad"
                                disabled={transformBusy === cat}
                                onClick={() => handleTransformVote(cat, "bad")}
                                title="Artifacts / degraded"
                              >
                                Artifacts
                              </button>
                            </span>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="stacker-controls">
        <div className="stacker-target">
          <label className="target-field">
            <span className="target-label">BPM</span>
            <input
              type="number"
              className="target-input"
              value={targetBpm}
              min={60}
              max={200}
              onChange={(e) => setTargetBpm(Number(e.target.value))}
            />
          </label>
          <label className="target-field">
            <span className="target-label">Key</span>
            <select
              className="target-input target-select"
              value={targetKey}
              onChange={(e) => setTargetKey(e.target.value)}
            >
              {KEYS.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="stacker-actions">
          <button className="btn-secondary" onClick={() => handleShuffle()}>
            Shuffle All
          </button>
          <button
            className="btn-primary"
            onClick={() => handlePreview(false)}
            disabled={building || filledCount === 0}
          >
            {building ? "Building…" : "Preview"}
          </button>
          <button
            className="btn-secondary"
            onClick={handleExport}
            disabled={exporting || filledCount === 0}
          >
            {exporting ? "Exporting…" : "Export"}
          </button>
        </div>
      </div>

      {preview && (
        <div className={`stacker-preview${building ? " stacker-preview--rebuilding" : ""}`}>
          {building && <div className="rebuild-banner">Rebuilding…</div>}
          <div className="preview-header">
            <span className="preview-label">Mix Preview</span>
            <button className="btn-icon" onClick={handlePlayPause} disabled={building}>
              {playing ? "⏸" : "▶"}
            </button>
          </div>
          <div className="preview-waveform" ref={mixContainerRef} />
        </div>
      )}

      {library.length === 0 && (
        <div className="stacker-empty">
          No stems in library yet. Use the Trim Picker to process and trim
          songs first.
        </div>
      )}
    </div>
  );
}
