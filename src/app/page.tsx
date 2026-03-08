import Image from "next/image";
import BasicAIChatInput from "@/components/ui/ai-chat-input-block";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-6 py-3">
        <div className="flex items-center gap-3">
          <Image
            src="/cytech-logo.png"
            alt="CY Tech logo"
            width={120}
            height={40}
            priority
            className="object-contain"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Assistant IA Immobilier</span>
        </div>
      </header>

      {/* Chat area */}
      <main className="flex flex-1 flex-col">
        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 py-8">
          <div className="mx-auto flex max-w-2xl flex-col gap-6">
            {/* Welcome message */}
            <div className="flex flex-col items-center gap-3 py-12 text-center">
              <Image
                src="/cytech-logo.png"
                alt="CY Tech"
                width={80}
                height={27}
                className="object-contain opacity-60"
              />
              <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                Assistant Intelligent d&apos;Analyse Immobilière
              </h1>
              <p className="max-w-md text-sm text-muted-foreground">
                Posez vos questions sur le marché immobilier français. Je peux analyser
                les données DVF, résumer des rapports et évaluer des descriptions de biens.
              </p>
            </div>

            {/* Example prompts */}
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {[
                "Quel est le prix moyen au m² à Paris en 2023 ?",
                "Quelles sont les tendances du marché à Lyon ?",
                "Résume ce rapport de marché immobilier",
                "Analyse la description de ce bien immobilier",
              ].map((prompt) => (
                <button
                  key={prompt}
                  className="rounded-card border border-border bg-card px-4 py-3 text-left text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Input area — pinned to bottom */}
        <div className="border-t border-border bg-background px-4 py-4">
          <div className="mx-auto flex max-w-2xl flex-col items-center">
            <BasicAIChatInput />
            <p className="mt-2 text-center text-xs text-muted-foreground">
              CY Tech · Projet 3 — Données DVF & Hugging Face
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
