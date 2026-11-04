"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";
import { importAudio, type Song } from "@/lib/api";

const AUDIO_ACCEPT = "audio/*,.mp3,.wav,.flac,.m4a,.aac,.ogg,.opus,.aiff,.aif,.wma,.lrc";

const LRC_EXT = /\.(lrc)$/i;
/** Lower-cased filename without its extension — the sidecar pairing key.
 * `Track.mp3` pairs with `track.LRC`, same rule as the folder importer. */
const stemOf = (name: string) => name.replace(/\.[^.]+$/, "").toLowerCase();

// Mirrors _LEADING_TRACK_NO in backend/app/services/imports/audio.py so the
// prefill the user sees matches what the server would guess. Duplicating
// these 6 lines is cheaper than a round-trip per file; the server value
// wins whenever a field is left empty.
const LEADING_TRACK_NO = /^\s*\d{1,3}\s*[-._)\]]\s*/;

/** Same (title, artist) rule as the backend: strip a leading track number,
 * split on the LAST " - " — "01 - Daft Punk - One More Time" →
 * ("One More Time", "Daft Punk"). */
function guessMetadata(filename: string): { title: string; artist: string } {
  let stem = filename.replace(/\.[^.]+$/, "");
  stem = stem.replace(LEADING_TRACK_NO, "").replace(/_/g, " ").trim();
  const sep = stem.lastIndexOf(" - ");
  if (sep > 0 && sep + 3 < stem.length) {
    return { title: stem.slice(sep + 3).trim(), artist: stem.slice(0, sep).trim() };
  }
  return { title: stem || "Untitled", artist: "" };
}

type FileState = "idle" | "uploading" | "done" | "error";

type FileEntry = {
  id: string;
  file: File;
  /** `.lrc` sidecar paired by filename stem, if the user supplied one. */
  lrc?: File;
  title: string;
  artist: string;
  progress: number;
  state: FileState;
  error?: string;
};

let entrySeq = 0;

export function LocalImport() {
  const qc = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [processNow, setProcessNow] = useState(false);
  const [importing, setImporting] = useState(false);

  const addFiles = useCallback((files: FileList | File[]) => {
    // Sidecars are keyed by filename stem, so `track.mp3` + `track.lrc`
    // arrive as one entry. A `.lrc` dropped on its own attaches to
    // whatever is still pending — "here's the lyrics for that one".
    const lrcByStem = new Map<string, File>();
    const audioFiles: File[] = [];
    for (const file of Array.from(files)) {
      if (LRC_EXT.test(file.name)) lrcByStem.set(stemOf(file.name), file);
      else audioFiles.push(file);
    }

    const next: FileEntry[] = audioFiles.map((file) => {
      const guessed = guessMetadata(file.name);
      return {
        id: `f${++entrySeq}`,
        file,
        lrc: lrcByStem.get(stemOf(file.name)),
        title: guessed.title,
        artist: guessed.artist,
        progress: 0,
        state: "idle",
      };
    });

    setEntries((prev) => {
      const patched = prev.map((e) => {
        const lrc = lrcByStem.get(stemOf(e.file.name));
        return e.state === "idle" && !e.lrc && lrc ? { ...e, lrc } : e;
      });
      return [...patched, ...next];
    });
  }, []);

  const patchEntry = (id: string, patch: Partial<FileEntry>) => {
    setEntries((prev) =>
      prev.map((e) => (e.id === id ? { ...e, ...patch } : e))
    );
  };

  const importAll = async () => {
    setImporting(true);
    // Sequential on purpose: keeps memory and API-side ffmpeg load
    // predictable for a 100-track drop, and the per-row progress reads
    // top-to-bottom instead of interleaving.
    for (const entry of entries) {
      if (entry.state === "done") continue;
      patchEntry(entry.id, { state: "uploading", progress: 0, error: undefined });
      try {
        const song: Song = await importAudio(
          {
            file: entry.file,
            lrcFile: entry.lrc,
            title: entry.title,
            artist: entry.artist || undefined,
            processNow,
          },
          (fraction) => patchEntry(entry.id, { progress: fraction }),
        );
        patchEntry(entry.id, { state: "done", progress: 1 });
        // Library list refreshes immediately; its existing 1 s poll picks
        // up status changes once the song is queued.
        qc.invalidateQueries({ queryKey: ["songs"] });
        void song;
      } catch (err) {
        patchEntry(entry.id, {
          state: "error",
          error: err instanceof Error ? err.message : "upload failed",
        });
      }
    }
    setImporting(false);
  };

  const pending = entries.filter((e) => e.state === "idle" || e.state === "error");
  const allDone = entries.length > 0 && pending.length === 0;

  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-semibold">Import from disk</h2>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
        }}
        className={`border border-dashed rounded p-4 flex flex-col gap-3 items-center text-center transition-colors ${
          dragOver ? "border-blue-500 bg-blue-500/5" : "opacity-80"
        }`}
      >
        <p className="text-sm opacity-70">
          Drop audio files here (mp3 / wav / flac / m4a / ogg …). Drop a
          matching <code>.lrc</code> alongside and its lyrics are used
          instead of a Genius lookup.
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={AUDIO_ACCEPT}
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
          className="text-sm file:mr-3 file:border file:rounded file:px-3 file:py-1 file:bg-transparent file:text-sm hover:file:bg-black/5 dark:hover:file:bg-white/10"
        />
      </div>

      {entries.length > 0 && (
        <div className="flex flex-col gap-2">
          <ul className="flex flex-col gap-2">
            {entries.map((entry) => (
              <li
                key={entry.id}
                className="border rounded p-2 flex flex-col gap-2"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`text-xs px-2 py-1 rounded whitespace-nowrap ${
                      entry.state === "done"
                        ? "bg-green-500/20"
                        : entry.state === "error"
                          ? "bg-red-500/20"
                          : "bg-blue-500/20"
                    }`}
                  >
                    {entry.state === "uploading"
                      ? `${Math.round(entry.progress * 100)}%`
                      : entry.state}
                  </span>
                  {entry.lrc && (
                    <span
                      className="text-xs px-2 py-1 rounded bg-purple-500/20 whitespace-nowrap"
                      title={`${entry.lrc.name} — lyric sidecar`}
                    >
                      lrc
                    </span>
                  )}
                  <div className="flex-1 min-w-0 flex gap-2">
                    <input
                      type="text"
                      value={entry.title}
                      placeholder="Title"
                      disabled={entry.state !== "idle" && entry.state !== "error"}
                      onChange={(e) =>
                        patchEntry(entry.id, { title: e.target.value })
                      }
                      className="flex-1 min-w-0 border rounded px-2 py-1 bg-transparent text-sm"
                    />
                    <input
                      type="text"
                      value={entry.artist}
                      placeholder="Artist"
                      disabled={entry.state !== "idle" && entry.state !== "error"}
                      onChange={(e) =>
                        patchEntry(entry.id, { artist: e.target.value })
                      }
                      className="w-40 shrink-0 border rounded px-2 py-1 bg-transparent text-sm"
                    />
                  </div>
                  <button
                    type="button"
                    aria-label={`Remove ${entry.file.name}`}
                    onClick={() =>
                      setEntries((prev) => prev.filter((e) => e.id !== entry.id))
                    }
                    disabled={entry.state === "uploading"}
                    className="text-sm opacity-50 hover:opacity-100 hover:text-red-500 px-2 disabled:opacity-30"
                  >
                    ×
                  </button>
                </div>
                {entry.state === "uploading" && (
                  <div className="h-1 rounded bg-black/10 dark:bg-white/10 overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-[width]"
                      style={{ width: `${entry.progress * 100}%` }}
                    />
                  </div>
                )}
                {entry.error && (
                  <p className="text-xs text-red-600 truncate">{entry.error}</p>
                )}
              </li>
            ))}
          </ul>

          <div className="flex items-center gap-3 flex-wrap">
            <button
              type="button"
              onClick={() => void importAll()}
              disabled={importing || allDone}
              className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
            >
              {importing
                ? "Importing…"
                : pending.length > 0
                  ? `Import ${pending.length} file${pending.length === 1 ? "" : "s"}`
                  : "All imported"}
            </button>
            <label className="text-sm opacity-80 flex items-center gap-2">
              <input
                type="checkbox"
                checked={processNow}
                disabled={importing}
                onChange={(e) => setProcessNow(e.target.checked)}
              />
              Process immediately (analyze → stems → transcription)
            </label>
            <button
              type="button"
              onClick={() => {
                setEntries([]);
                if (inputRef.current) inputRef.current.value = "";
              }}
              disabled={importing}
              className="text-sm opacity-50 hover:opacity-100 disabled:opacity-30"
            >
              Clear list
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
