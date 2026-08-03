import { useState } from "react";
import { submitKeyFeedback } from "../api";

interface Analysis {
  bpm: number;
  key: string;
  camelot: string;
  key_alternatives?: string[];
}

interface Props {
  trimSec: number;
  downloadUrl: string;
  filename: string;
  analysis: Analysis | null;
  onReset: () => void;
}

type KeyVoteState = "idle" | "picking" | "saving" | "done";

export default function ResultPanel({
  trimSec,
  downloadUrl,
  filename,
  analysis,
  onReset,
}: Props) {
  const [voteState, setVoteState] = useState<KeyVoteState>("idle");
  const [confirmedKey, setConfirmedKey] = useState<string | null>(null);
  const [voteError, setVoteError] = useState<string | null>(null);

  const alternatives = analysis?.key_alternatives ?? [];

  async function voteCorrect() {
    if (!analysis) return;
    setVoteState("saving");
    setVoteError(null);
    try {
      const res = await submitKeyFeedback({
        filename,
        detected_key: analysis.key,
        detected_camelot: analysis.camelot,
        verdict: "correct",
      });
      setConfirmedKey(res.key);
      setVoteState("done");
    } catch (e) {
      setVoteError(String(e));
      setVoteState("idle");
    }
  }

  async function voteWrong(correctedKey: string) {
    if (!analysis) return;
    setVoteState("saving");
    setVoteError(null);
    try {
      const res = await submitKeyFeedback({
        filename,
        detected_key: analysis.key,
        detected_camelot: analysis.camelot,
        verdict: "wrong",
        corrected_key: correctedKey,
      });
      setConfirmedKey(res.key);
      setVoteState("done");
    } catch (e) {
      setVoteError(String(e));
      setVoteState("picking");
    }
  }

  return (
    <div className="result-panel">
      <div className="result-check">Trimmed</div>

      <div className="result-details">
        <div className="result-row">
          <span className="result-label">Trim point</span>
          <span className="result-value">{(trimSec * 1000).toFixed(1)} ms</span>
        </div>
        {analysis && (
          <>
            <div className="result-row">
              <span className="result-label">BPM</span>
              <span className="result-value">{analysis.bpm}</span>
            </div>
            <div className="result-row">
              <span className="result-label">Key</span>
              <span className="result-value">
                {confirmedKey ?? analysis.key} ({analysis.camelot})
              </span>
            </div>
          </>
        )}
      </div>

      {analysis && voteState !== "done" && (
        <div className="feedback-block">
          <div className="feedback-prompt">Does the key sound right?</div>
          {voteState === "idle" && (
            <div className="feedback-actions">
              <button className="btn-vote btn-vote-good" onClick={voteCorrect}>
                Sounds right
              </button>
              <button
                className="btn-vote btn-vote-bad"
                onClick={() => setVoteState("picking")}
              >
                Wrong key
              </button>
            </div>
          )}
          {voteState === "picking" && (
            <div className="feedback-corrections">
              <div className="feedback-hint">Pick the correct key:</div>
              <div className="feedback-chip-row">
                {alternatives.map((key) => (
                  <button
                    key={key}
                    className="btn-chip"
                    onClick={() => voteWrong(key)}
                  >
                    {key}
                  </button>
                ))}
              </div>
              <button
                className="btn-link"
                onClick={() => setVoteState("idle")}
              >
                Cancel
              </button>
            </div>
          )}
          {voteState === "saving" && (
            <div className="feedback-hint">Saving…</div>
          )}
          {voteError && <div className="feedback-error">{voteError}</div>}
        </div>
      )}

      {voteState === "done" && confirmedKey && (
        <div className="feedback-done">
          Key locked in: <strong>{confirmedKey}</strong>
        </div>
      )}

      <div className="result-actions">
        <a href={downloadUrl} className="btn-primary" download>
          Download stems (.zip)
        </a>
        <button className="btn-secondary" onClick={onReset}>
          Process another song
        </button>
      </div>
    </div>
  );
}
