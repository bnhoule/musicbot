export interface Job {
  id: string;
  filename: string;
  status: "queued" | "separating" | "detecting" | "ready" | "trimmed" | "error";
  error: string | null;
  bpm: number | null;
}

export interface Candidate {
  time_sec: number;
  energy_pct: number;
}

export interface CandidatesResponse {
  candidates: Candidate[];
  all_downbeats: Candidate[];
  bpm: number;
  analysis: {
    bpm: number;
    key: string;
    camelot: string;
  };
}

export interface PickResponse {
  status: string;
  trim_sec: number;
  download_url: string;
}

export async function uploadSong(file: File): Promise<{ job_id: string; status?: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getCandidates(jobId: string): Promise<CandidatesResponse> {
  const res = await fetch(`/api/candidates/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function getDrumsUrl(jobId: string): string {
  return `/api/audio/${jobId}/drums`;
}

export async function pickTrim(jobId: string, trimSec: number): Promise<PickResponse> {
  const res = await fetch("/api/pick", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, trim_sec: trimSec }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function getDownloadUrl(jobId: string): string {
  return `/api/download/${jobId}`;
}

export interface LibrarySong {
  cache_key: string;
  filename: string;
  bpm: number | null;
  key: string;
  camelot: string;
  backends?: string[];
}

export async function getLibrary(): Promise<LibrarySong[]> {
  const res = await fetch("/api/library");
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.songs;
}

export async function loadFromLibrary(cacheKey: string): Promise<{ job_id: string }> {
  const res = await fetch(`/api/library/load/${encodeURIComponent(cacheKey)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface BackendInfo {
  stems_dir: string;
  stems: string[];
}

export interface StemLibraryEntry {
  name: string;
  filename: string;
  trim_sec: number;
  bpm: number | null;
  key: string;
  camelot: string;
  backends: Record<string, BackendInfo>;
}

export async function getStemLibrary(): Promise<StemLibraryEntry[]> {
  const res = await fetch("/api/stem-library");
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.entries;
}

export async function deleteStemLibraryEntry(name: string): Promise<void> {
  const res = await fetch(`/api/stem-library/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

// ---------------------------------------------------------------------------
// Stack API
// ---------------------------------------------------------------------------

export interface SlotInfo {
  name: string;
  bpm: number | null;
  key: string;
  camelot: string;
}

export interface ShuffleResponse {
  slots: Record<string, SlotInfo>;
}

export interface StackPreviewResponse {
  stack_id: string;
  mix_url: string;
  stem_urls: Record<string, string>;
  slots_info: Record<string, {
    name: string;
    backend: string;
    original_bpm: number;
    detected_bpm: number;
    original_key: string;
    trim_sec: number;
    loop_bars: number | null;
    loop_start_sec: number | null;
  }>;
}

export async function stackShuffle(): Promise<ShuffleResponse> {
  const res = await fetch("/api/stack/shuffle", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stackPreview(
  slots: Record<string, string>,
  targetBpm: number,
  targetKey: string,
  slotBackends?: Record<string, string>,
  slotBpms?: Record<string, number>,
  slotLoops?: Record<string, number>,
): Promise<StackPreviewResponse> {
  const res = await fetch("/api/stack/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      slots,
      target_bpm: targetBpm,
      target_key: targetKey,
      slot_backends: slotBackends || null,
      slot_bpms: slotBpms || null,
      slot_loops: slotLoops || null,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stackExportUrl(
  slots: Record<string, string>,
  targetBpm: number,
  targetKey: string,
  slotBackends?: Record<string, string>,
  slotBpms?: Record<string, number>,
  slotLoops?: Record<string, number>,
): Promise<Blob> {
  const res = await fetch("/api/stack/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      slots,
      target_bpm: targetBpm,
      target_key: targetKey,
      slot_backends: slotBackends || null,
      slot_bpms: slotBpms || null,
      slot_loops: slotLoops || null,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}
