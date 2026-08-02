import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import WaveSurfer from "wavesurfer.js";
import type { Candidate } from "../api";

const SCOPE_ZOOM = 800; // px/s  → ~1.25 ms per pixel at 640px container

interface Props {
  audioUrl: string;
  candidates: Candidate[];
  allDownbeats: Candidate[];
  bpm: number;
  onPick: (trimSec: number) => void;
  picking: boolean;
}

const PREVIEW_DURATION = 4;
const NEARBY_WINDOW = 20;
const NEARBY_MAX = 8;
const ZOOM_STEPS = [0, 10, 25, 50, 100, 200];

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

function energyBar(pct: number): string {
  const filled = Math.round((pct / 100) * 10);
  return "█".repeat(filled) + "░".repeat(10 - filled);
}

function findNearby(allDownbeats: Candidate[], timeSec: number): Candidate[] {
  const nearby = allDownbeats.filter(
    (d) => Math.abs(d.time_sec - timeSec) <= NEARBY_WINDOW
  );
  nearby.sort((a, b) => b.energy_pct - a.energy_pct);
  const top = nearby.slice(0, NEARBY_MAX);
  top.sort((a, b) => a.time_sec - b.time_sec);
  return top;
}

export default function WaveformPicker({
  audioUrl,
  candidates,
  allDownbeats,
  bpm,
  onPick,
  picking,
}: Props) {
  const waveRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const scopeRef = useRef<HTMLDivElement>(null);
  const scopeWsRef = useRef<WaveSurfer | null>(null);
  const previewTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [selected, setSelected] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [scopeReady, setScopeReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [zoomIdx, setZoomIdx] = useState(0);

  // "granular" = waveform click sets trim directly; "snap" = find nearby downbeats
  const [pickMode, setPickMode] = useState<"granular" | "snap">("granular");

  const [candidateMode, setCandidateMode] = useState<"top" | "nearby">("top");
  const [activeCandidates, setActiveCandidates] = useState<Candidate[]>(candidates);
  const [clickTime, setClickTime] = useState<number | null>(null);
  const [trimPoint, setTrimPoint] = useState<number | null>(null);
  const [msInput, setMsInput] = useState<string>("");

  useEffect(() => {
    if (candidateMode === "top") setActiveCandidates(candidates);
  }, [candidates, candidateMode]);

  const autoPickIdx = activeCandidates.reduce(
    (best, c, i) => (c.energy_pct > activeCandidates[best].energy_pct ? i : best),
    0
  );
  const activeIdx = selected ?? autoPickIdx;
  const currentTrim = trimPoint ?? activeCandidates[activeIdx]?.time_sec ?? 0;

  // Create waveform
  useEffect(() => {
    if (!waveRef.current) return;

    const ws = WaveSurfer.create({
      container: waveRef.current,
      waveColor: "#4a5568",
      progressColor: "#667eea",
      cursorColor: "#e53e3e",
      cursorWidth: 2,
      height: 128,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      autoScroll: true,
      interact: true,
      url: audioUrl,
    });

    ws.on("ready", () => {
      setReady(true);
      setDuration(ws.getDuration());
    });
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));

    wsRef.current = ws;
    return () => { ws.destroy(); };
  }, [audioUrl]);

  // Scope waveform — high-zoom detail strip
  useEffect(() => {
    if (!scopeRef.current) return;
    const ws = WaveSurfer.create({
      container: scopeRef.current,
      waveColor: "#2a3050",
      progressColor: "#4a5580",
      cursorColor: "#fc8181",
      cursorWidth: 2,
      height: 80,
      barWidth: 1,
      barGap: 0,
      barRadius: 0,
      autoScroll: true,
      interact: true,
      url: audioUrl,
    });
    ws.once("ready", () => {
      setScopeReady(true);
      ws.zoom(SCOPE_ZOOM);
    });
    scopeWsRef.current = ws;
    return () => { ws.destroy(); };
  }, [audioUrl]);

  // Apply zoom level
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || !ready) return;
    ws.zoom(ZOOM_STEPS[zoomIdx]);
  }, [zoomIdx, ready]);

  // Scroll-wheel zoom
  useEffect(() => {
    const el = waveRef.current;
    if (!el) return;
    const handleWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      setZoomIdx((prev) => {
        if (e.deltaY < 0) return Math.min(prev + 1, ZOOM_STEPS.length - 1);
        return Math.max(prev - 1, 0);
      });
    };
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, []);

  const seekTo = useCallback((timeSec: number, autoPlay: boolean) => {
    const ws = wsRef.current;
    if (!ws) return;
    const dur = ws.getDuration();
    if (dur <= 0) return;
    ws.seekTo(Math.min(Math.max(0, timeSec) / dur, 1));
    if (autoPlay) {
      if (previewTimer.current) clearTimeout(previewTimer.current);
      ws.play();
      previewTimer.current = setTimeout(() => ws.pause(), PREVIEW_DURATION * 1000);
    }
    // Sync scope cursor + center it (only if not triggered by the scope sync effect itself)
    const scopeWs = scopeWsRef.current;
    if (scopeWs) {
      const scopeDur = scopeWs.getDuration();
      if (scopeDur > 0) {
        scopeWs.seekTo(Math.min(Math.max(0, timeSec / scopeDur), 1));
        requestAnimationFrame(() => {
          const scrollEl = scopeWsRef.current?.getWrapper();
          if (scrollEl) {
            const cursorX = timeSec * SCOPE_ZOOM;
            scrollEl.scrollLeft = Math.max(0, cursorX - scrollEl.clientWidth / 2);
          }
        });
      }
    }
  }, []);

  const applyTrim = useCallback((timeSec: number) => {
    const t = Math.max(0, timeSec);
    setTrimPoint(t);
    setMsInput((t * 1000).toFixed(1));
    seekTo(t, false);
  }, [seekTo]);

  // Scope click → pixel-accurate time from scroll position + click offset
  useEffect(() => {
    const container = scopeRef.current;
    if (!container) return;

    const handleMouseDown = (e: MouseEvent) => {
      const scrollEl = scopeWsRef.current?.getWrapper();
      if (!scrollEl) return;
      const rect = scrollEl.getBoundingClientRect();
      const clickX = e.clientX - rect.left + scrollEl.scrollLeft;
      applyTrim(Math.max(0, clickX / SCOPE_ZOOM));
    };

    container.addEventListener("mousedown", handleMouseDown);
    return () => container.removeEventListener("mousedown", handleMouseDown);
  }, [applyTrim]);

  // When scope becomes ready, sync it to the current trim point
  useEffect(() => {
    if (!scopeReady) return;
    seekTo(currentTrim, false);
  // Only fire when scopeReady flips true; currentTrim is captured at that moment
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeReady]);

  // Waveform click — behaviour depends on pickMode
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws) return;

    const handleClick = () => {
      const dur = ws.getDuration();
      if (dur <= 0) return;
      const time = ws.getCurrentTime();

      if (pickMode === "granular") {
        applyTrim(time);
      } else {
        // snap: find nearby downbeats
        if (allDownbeats.length === 0) return;
        setClickTime(time);
        const nearby = findNearby(allDownbeats, time);
        if (nearby.length > 0) {
          setActiveCandidates(nearby);
          setCandidateMode("nearby");
          let closestIdx = 0;
          let closestDist = Infinity;
          nearby.forEach((c, i) => {
            const dist = Math.abs(c.time_sec - time);
            if (dist < closestDist) { closestDist = dist; closestIdx = i; }
          });
          setSelected(closestIdx);
          applyTrim(nearby[closestIdx].time_sec);
        }
      }
    };

    ws.on("click", handleClick);
    return () => { ws.un("click", handleClick); };
  }, [allDownbeats, pickMode, applyTrim]);

  // Auto-seek on initial load
  const initialSeekDone = useRef(false);
  useEffect(() => {
    if (ready && activeCandidates.length > 0 && !initialSeekDone.current) {
      initialSeekDone.current = true;
      applyTrim(activeCandidates[autoPickIdx].time_sec);
    }
  }, [ready, activeCandidates, autoPickIdx, applyTrim]);

  // Seek when candidate selection changes (snap mode / lock-on)
  useEffect(() => {
    if (ready && selected !== null && activeCandidates[selected]) {
      applyTrim(activeCandidates[selected].time_sec);
    }
  }, [ready, selected, activeCandidates, applyTrim]);

  // Tick marks overlay
  const ticks = useMemo(() => {
    if (duration <= 0 || allDownbeats.length === 0) return [];
    return allDownbeats.map((d) => ({
      left: `${(d.time_sec / duration) * 100}%`,
      opacity: 0.15 + (d.energy_pct / 100) * 0.85,
    }));
  }, [allDownbeats, duration]);

  // Candidate lock-on click
  const handleCandidateClick = (idx: number) => {
    setSelected(idx);
    applyTrim(activeCandidates[idx].time_sec);
  };

  const handleResetToTop = () => {
    setCandidateMode("top");
    setActiveCandidates(candidates);
    setSelected(null);
    setClickTime(null);
  };

  // Direct ms input
  const handleMsChange = (val: string) => {
    setMsInput(val);
    const parsed = parseFloat(val);
    if (!isNaN(parsed) && parsed >= 0) {
      const t = parsed / 1000;
      setTrimPoint(t);
      seekTo(t, false);
    }
  };

  const zoomLabel = zoomIdx === 0 ? "Fit" : `${ZOOM_STEPS[zoomIdx]}px/s`;

  return (
    <div className="waveform-picker">
      <div className="waveform-header">
        <span>{bpm} BPM</span>
        <div className="waveform-controls">
          {/* Pick mode toggle */}
          <div className="pick-mode-toggle">
            <button
              className={`btn-pick-mode ${pickMode === "granular" ? "active" : ""}`}
              onClick={() => setPickMode("granular")}
              title="Click waveform to set exact trim point"
            >
              Granular
            </button>
            <button
              className={`btn-pick-mode ${pickMode === "snap" ? "active" : ""}`}
              onClick={() => setPickMode("snap")}
              title="Click waveform to snap to nearest downbeat"
            >
              Snap
            </button>
          </div>
          <div className="zoom-controls">
            <button
              className="btn-icon"
              onClick={() => setZoomIdx((p) => Math.max(p - 1, 0))}
              disabled={zoomIdx === 0}
              title="Zoom out"
            >
              −
            </button>
            <span className="zoom-label">{zoomLabel}</span>
            <button
              className="btn-icon"
              onClick={() => setZoomIdx((p) => Math.min(p + 1, ZOOM_STEPS.length - 1))}
              disabled={zoomIdx === ZOOM_STEPS.length - 1}
              title="Zoom in"
            >
              +
            </button>
          </div>
          <button
            className="btn-secondary"
            onClick={() => {
              const ws = wsRef.current;
              if (!ws) return;
              if (playing) ws.pause();
              else ws.play();
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
        </div>
      </div>

      {/* Waveform with tick overlay */}
      <div className="waveform-main-wrap">
        <div className="waveform-main-container">
          <div ref={waveRef} className="waveform-main" />
          {zoomIdx === 0 && ticks.length > 0 && (
            <div className="overview-ticks" aria-hidden>
              {ticks.map((t, i) => (
                <div
                  key={i}
                  className="overview-tick"
                  style={{ left: t.left, opacity: t.opacity }}
                />
              ))}
            </div>
          )}
        </div>
        <div className="waveform-hint">
          {pickMode === "granular"
            ? "Click to set trim point · ⌘+scroll to zoom"
            : "Click to snap to nearest downbeat · ⌘+scroll to zoom"}
        </div>
      </div>

      {/* Scope — high-zoom detail strip always centered on trim point */}
      <div className="waveform-scope-wrap">
        <div className="waveform-scope-label">
          Detail · {SCOPE_ZOOM}px/s · click to set trim
        </div>
        <div ref={scopeRef} className="waveform-scope" />
      </div>

      {/* Trim point row */}
      <div className="fine-tune">
        <div className="fine-tune-row">
          <span className="fine-tune-label">Trim point</span>
          <span className="fine-tune-time">{formatTime(currentTrim)}</span>
          <input
            className="fine-tune-ms-input"
            type="number"
            step="0.1"
            min="0"
            value={msInput || (currentTrim * 1000).toFixed(1)}
            onChange={(e) => handleMsChange(e.target.value)}
            title="Edit milliseconds directly"
          />
          <span className="fine-tune-unit">ms</span>
          <button
            className="btn-nudge nudge-play"
            onClick={() => seekTo(currentTrim, true)}
          >
            ▶ preview
          </button>
        </div>
      </div>

      <button
        className="btn-primary pick-btn"
        onClick={() => onPick(currentTrim)}
        disabled={picking || activeCandidates.length === 0}
      >
        {picking ? "Trimming…" : `Pick ${formatTime(currentTrim)}`}
      </button>

      {/* Candidate list header + reset */}
      <div className="candidates-mode">
        {candidateMode === "nearby" && clickTime != null ? (
          <>
            <span className="mode-label">
              Downbeats near {formatTime(clickTime)} — click to lock on
            </span>
            <button className="btn-secondary btn-small" onClick={handleResetToTop}>
              Show top energy
            </button>
          </>
        ) : (
          <span className="mode-label">Top candidates by kick energy — click to lock on</span>
        )}
      </div>

      <div className="candidates-list">
        {activeCandidates.map((c, i) => (
          <button
            key={`${c.time_sec}-${i}`}
            className={`candidate-row ${i === activeIdx ? "selected" : ""} ${i === autoPickIdx ? "auto-pick" : ""}`}
            onClick={() => handleCandidateClick(i)}
          >
            <span className="candidate-num">{i + 1}</span>
            <span className="candidate-time">{formatTime(c.time_sec)}</span>
            <span className="candidate-bar">{energyBar(c.energy_pct)}</span>
            <span className="candidate-pct">{c.energy_pct.toFixed(0)}%</span>
            {i === autoPickIdx && <span className="auto-badge">auto</span>}
          </button>
        ))}
      </div>
    </div>
  );
}
