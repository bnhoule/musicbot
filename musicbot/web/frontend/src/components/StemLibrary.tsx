import { useEffect, useState } from "react";
import { getStemLibrary, deleteStemLibraryEntry, type StemLibraryEntry } from "../api";

export default function StemLibrary() {
  const [entries, setEntries] = useState<StemLibraryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    getStemLibrary()
      .then(setEntries)
      .catch(() => setEntries([]))
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = async (name: string) => {
    setDeleting(name);
    setConfirmDelete(null);
    try {
      await deleteStemLibraryEntry(name);
      setEntries((prev) => prev.filter((e) => e.name !== name));
    } catch {
      setDeleting(null);
    }
  };

  if (loading) return null;
  if (entries.length === 0) {
    return (
      <div className="stem-library-empty">
        No stems trimmed yet. Pick trim points above to build your library.
      </div>
    );
  }

  return (
    <div className="stem-library">
      <div className="stem-library-header">
        Stem library — {entries.length} songs trimmed
      </div>
      <div className="stem-library-list">
        {entries.map((e) => (
          <div key={e.name} className="stem-library-row">
            <span className="stem-lib-name">{e.name}</span>
            <span className="stem-lib-meta">
              {e.bpm ? `${e.bpm}` : "—"} BPM
            </span>
            <span className="stem-lib-meta">
              {e.key || "—"}{e.camelot ? ` (${e.camelot})` : ""}
            </span>
            <span className="stem-lib-trim">
              trim {(e.trim_sec * 1000).toFixed(0)}ms
            </span>

            {confirmDelete === e.name ? (
              <div className="delete-confirm">
                <button
                  className="btn-delete-cancel"
                  onClick={() => setConfirmDelete(null)}
                >
                  Cancel
                </button>
                <button
                  className="btn-delete-confirm"
                  onClick={() => handleDelete(e.name)}
                  disabled={deleting === e.name}
                >
                  {deleting === e.name ? "…" : "Delete"}
                </button>
              </div>
            ) : (
              <button
                className="btn-delete-entry"
                onClick={() => setConfirmDelete(e.name)}
                title="Remove from library"
              >
                ✕
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
