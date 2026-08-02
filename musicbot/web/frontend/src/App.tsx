import { useState, useEffect, useCallback } from "react";
import DropZone from "./components/DropZone";
import Library from "./components/Library";
import StemLibrary from "./components/StemLibrary";
import ProcessingStatus from "./components/ProcessingStatus";
import WaveformPicker from "./components/WaveformPicker";
import ResultPanel from "./components/ResultPanel";
import Stacker from "./components/Stacker";
import {
  uploadSong,
  getJob,
  getCandidates,
  getDrumsUrl,
  pickTrim,
  getDownloadUrl,
  loadFromLibrary,
  type Candidate,
  type Job,
} from "./api";

type AppState = "idle" | "processing" | "picking" | "trimming" | "done";
type TabId = "trimmer" | "stacker";

function TrimPicker() {
  const [state, setState] = useState<AppState>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [allDownbeats, setAllDownbeats] = useState<Candidate[]>([]);
  const [bpm, setBpm] = useState(0);
  const [analysis, setAnalysis] = useState<{
    bpm: number;
    key: string;
    camelot: string;
  } | null>(null);
  const [trimSec, setTrimSec] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const enterPicking = useCallback(async (jid: string) => {
    const data = await getCandidates(jid);
    setCandidates(data.candidates);
    setAllDownbeats(data.all_downbeats ?? []);
    setBpm(data.bpm);
    setAnalysis(data.analysis);
    setState("picking");
  }, []);

  const handleFile = useCallback(async (file: File) => {
    try {
      setError(null);
      setState("processing");
      const resp = await uploadSong(file);
      setJobId(resp.job_id);
      if (resp.status === "ready") {
        await enterPicking(resp.job_id);
      }
    } catch (e) {
      setError(String(e));
      setState("idle");
    }
  }, [enterPicking]);

  const handleLibrarySelect = useCallback(async (cacheKey: string) => {
    try {
      setError(null);
      setState("processing");
      const resp = await loadFromLibrary(cacheKey);
      setJobId(resp.job_id);
      await enterPicking(resp.job_id);
    } catch (e) {
      setError(String(e));
      setState("idle");
    }
  }, [enterPicking]);

  useEffect(() => {
    if (!jobId || state !== "processing") return;
    let cancelled = false;

    async function poll() {
      let delay = 2000;
      while (!cancelled) {
        try {
          const j = await getJob(jobId!);
          if (cancelled) break;
          setJob(j);

          if (j.status === "ready") {
            const data = await getCandidates(jobId!);
            if (cancelled) break;
            setCandidates(data.candidates);
            setAllDownbeats(data.all_downbeats ?? []);
            setBpm(data.bpm);
            setAnalysis(data.analysis);
            setState("picking");
            return;
          } else if (j.status === "error") {
            setError(j.error ?? "Processing failed");
            setState("idle");
            return;
          }
        } catch {
          // ignore transient fetch errors
        }
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * 1.3, 5000);
      }
    }
    poll();

    return () => { cancelled = true; };
  }, [jobId, state]);

  const handlePick = useCallback(
    async (sec: number) => {
      if (!jobId) return;
      setState("trimming");
      try {
        const result = await pickTrim(jobId, sec);
        setTrimSec(result.trim_sec);
        setState("done");
      } catch (e) {
        setError(String(e));
        setState("picking");
      }
    },
    [jobId]
  );

  const handleReset = () => {
    setState("idle");
    setJobId(null);
    setJob(null);
    setCandidates([]);
    setAllDownbeats([]);
    setAnalysis(null);
    setTrimSec(0);
    setError(null);
  };

  return (
    <>
      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      {state === "idle" && (
        <>
          <DropZone onFileSelected={handleFile} />
          <Library onSelect={handleLibrarySelect} />
          <StemLibrary />
        </>
      )}

      {state === "processing" && job && (
        <ProcessingStatus status={job.status} filename={job.filename} />
      )}

      {state === "processing" && !job && (
        <ProcessingStatus status="queued" filename="Uploading…" />
      )}

      {(state === "picking" || state === "trimming") && jobId && (
        <WaveformPicker
          audioUrl={getDrumsUrl(jobId)}
          candidates={candidates}
          allDownbeats={allDownbeats}
          bpm={bpm}
          onPick={handlePick}
          picking={state === "trimming"}
        />
      )}

      {state === "done" && jobId && (
        <>
          <ResultPanel
            trimSec={trimSec}
            downloadUrl={getDownloadUrl(jobId)}
            analysis={analysis}
            onReset={handleReset}
          />
          <StemLibrary />
        </>
      )}
    </>
  );
}

export default function App() {
  const [tab, setTab] = useState<TabId>("trimmer");

  return (
    <div className="app">
      <header className="app-header">
        <h1>musicbot</h1>
        <nav className="app-tabs">
          <button
            className={`tab-btn ${tab === "trimmer" ? "active" : ""}`}
            onClick={() => setTab("trimmer")}
          >
            Trim Picker
          </button>
          <button
            className={`tab-btn ${tab === "stacker" ? "active" : ""}`}
            onClick={() => setTab("stacker")}
          >
            Stacker
          </button>
        </nav>
      </header>

      <main className="app-main">
        {tab === "trimmer" && <TrimPicker />}
        {tab === "stacker" && <Stacker />}
      </main>
    </div>
  );
}
