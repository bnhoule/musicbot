import { useEffect, useState } from "react";
import { getLibrary, type LibrarySong } from "../api";

interface Props {
  onSelect: (cacheKey: string) => void;
  disabled?: boolean;
}

export default function Library({ onSelect, disabled }: Props) {
  const [songs, setSongs] = useState<LibrarySong[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getLibrary()
      .then(setSongs)
      .catch(() => setSongs([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return null;
  if (songs.length === 0) return null;

  return (
    <div className="library">
      <div className="library-header">
        Pre-processed library — click to load instantly
      </div>
      <div className="library-list">
        {songs.map((s) => (
          <button
            key={s.cache_key}
            className="library-row"
            onClick={() => onSelect(s.cache_key)}
            disabled={disabled}
          >
            <span className="library-name">{s.filename}</span>
            <span className="library-meta">
              {s.bpm ? `${s.bpm} BPM` : ""}
              {s.key ? ` · ${s.key}` : ""}
              {s.camelot ? ` (${s.camelot})` : ""}
            </span>
            {s.backends && s.backends.length > 0 && (
              <span className="library-backends">
                {s.backends.map((b) => (
                  <span key={b} className={`backend-badge badge-${b}`}>
                    {b === "lalal" ? "Lalal" : "Demucs"}
                  </span>
                ))}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
