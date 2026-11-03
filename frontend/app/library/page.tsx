import { SearchPanel } from "@/components/SearchPanel";
import { LocalImport } from "@/components/LocalImport";
import { DownloadedSongs } from "@/components/DownloadedSongs";
import { NavHeader } from "@/components/NavHeader";

export default function LibraryPage() {
  return (
    <main className="min-h-screen max-w-4xl mx-auto p-8 flex flex-col gap-8 font-mono">
      <NavHeader />
      <SearchPanel />
      <LocalImport />
      <DownloadedSongs />
    </main>
  );
}
