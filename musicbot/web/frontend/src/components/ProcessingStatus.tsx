interface Props {
  status: string;
  filename: string;
}

const LABELS: Record<string, string> = {
  queued: "Queued…",
  separating: "Separating stems with Demucs…",
  detecting: "Detecting downbeats with madmom…",
  ready: "Ready — pick your trim point",
  trimmed: "Stems trimmed",
  error: "Error",
};

export default function ProcessingStatus({ status, filename }: Props) {
  const label = LABELS[status] ?? status;
  const isProcessing = status === "queued" || status === "separating" || status === "detecting";

  return (
    <div className="processing-status">
      <div className="processing-filename">{filename}</div>
      <div className="processing-label">
        {isProcessing && <span className="spinner" />}
        {label}
      </div>
    </div>
  );
}
