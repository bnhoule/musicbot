interface Props {
  trimSec: number;
  downloadUrl: string;
  analysis: { bpm: number; key: string; camelot: string } | null;
  onReset: () => void;
}

export default function ResultPanel({
  trimSec,
  downloadUrl,
  analysis,
  onReset,
}: Props) {
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
                {analysis.key} ({analysis.camelot})
              </span>
            </div>
          </>
        )}
      </div>

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
