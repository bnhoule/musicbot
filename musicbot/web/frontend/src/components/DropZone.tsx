import { useCallback, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function DropZone({ onFileSelected, disabled }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file && (file.name.endsWith(".mp3") || file.name.endsWith(".wav"))) {
        onFileSelected(file);
      }
    },
    [onFileSelected, disabled]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileSelected(file);
      e.target.value = "";
    },
    [onFileSelected]
  );

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <div className="dropzone-content">
        <div className="dropzone-icon">+</div>
        <p>Drag & drop an MP3 or WAV here</p>
        <p className="dropzone-or">or</p>
        <label className="dropzone-btn">
          Browse files
          <input
            type="file"
            accept=".mp3,.wav"
            onChange={handleFileInput}
            hidden
            disabled={disabled}
          />
        </label>
      </div>
    </div>
  );
}
