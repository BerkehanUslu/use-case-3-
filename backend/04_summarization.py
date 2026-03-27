"""
04_summarization.py
-------------------
French abstractive summarization for real estate texts and PDFs.
Model: moussaKam/barthez-orangesum-abstract (BARThez fine-tuned on OrangeSum)
"""

import argparse
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Global cache for the pipeline
# ---------------------------------------------------------------------------
_summarizer = None


def get_summarizer():
    """Lazy-load and cache the BARThez summarizer callable."""
    global _summarizer
    if _summarizer is None:
        from transformers import BarthezTokenizer, MBartForConditionalGeneration
        import torch

        _model_name = "moussaKam/barthez-orangesum-abstract"
        _tok = BarthezTokenizer.from_pretrained(_model_name)
        _mdl = MBartForConditionalGeneration.from_pretrained(_model_name)
        _mdl.training = False

        def _run(text: str, max_length: int, min_length: int) -> str:
            inputs = _tok(
                [text], return_tensors="pt", truncation=True, max_length=1024
            )
            with torch.no_grad():
                output_ids = _mdl.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=max_length,
                    min_length=min_length,
                )
            return _tok.decode(output_ids[0], skip_special_tokens=True)

        _summarizer = _run
    return _summarizer


# ---------------------------------------------------------------------------
# Core summarization helpers
# ---------------------------------------------------------------------------

def _run_summarizer(text: str, max_length: int, min_length: int) -> str:
    """Run the BARThez model on a single text chunk."""
    summarizer = get_summarizer()
    return summarizer(text, max_length=max_length, min_length=min_length)


def _mock_summary(text: str) -> str:
    """Return a canned summary for --mock mode."""
    return (
        "Résumé simulé : Ce bien immobilier présente des caractéristiques attractives "
        "avec une localisation stratégique et un potentiel d'investissement élevé."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_text(
    text: str,
    max_length: int = 150,
    min_length: int = 30,
    mock: bool = False,
) -> dict:
    """
    Summarize French text.

    Returns:
        {
            "summary": str,
            "original_length": int,
            "summary_length": int,
            "ratio": float,
        }
    """
    original_length = len(text)

    if mock:
        summary = _mock_summary(text)
    else:
        # BARThez handles up to ~1024 tokens; for safety we use chunk_and_summarize
        # for very long texts so that this function always works regardless of length.
        if original_length > 3000:
            result = chunk_and_summarize(text, max_length=max_length)
            summary = result["summary"]
        else:
            summary = _run_summarizer(text, max_length=max_length, min_length=min_length)

    summary_length = len(summary)
    ratio = round(summary_length / original_length, 4) if original_length > 0 else 0.0

    return {
        "summary": summary,
        "original_length": original_length,
        "summary_length": summary_length,
        "ratio": ratio,
    }


def summarize_pdf(
    pdf_path: str,
    max_length: int = 200,
    mock: bool = False,
) -> dict:
    """
    Extract text from a PDF using pypdf, then summarize it.

    Returns:
        {
            "summary": str,
            "pages": int,
            "original_length": int,
            "summary_length": int,
        }
    """
    import pypdf  # pip install pypdf

    with open(pdf_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        pages = len(reader.pages)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    full_text = "\n".join(text_parts)
    original_length = len(full_text)

    if mock:
        summary = _mock_summary(full_text)
    else:
        if original_length > 3000:
            result = chunk_and_summarize(full_text, max_length=max_length)
            summary = result["summary"]
        else:
            summary = _run_summarizer(
                full_text, max_length=max_length, min_length=30
            )

    return {
        "summary": summary,
        "pages": pages,
        "original_length": original_length,
        "summary_length": len(summary),
    }


def chunk_and_summarize(
    text: str,
    chunk_size: int = 1000,
    max_length: int = 150,
    mock: bool = False,
) -> dict:
    """
    For long texts: split into chunks, summarize each, then do a final summary pass.

    Returns:
        {
            "summary": str,
            "chunks": int,
            "original_length": int,
        }
    """
    original_length = len(text)

    # Split text into overlapping-free chunks on word boundaries
    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_len + word_len > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            current_words = []
            current_len = 0
        current_words.append(word)
        current_len += word_len

    if current_words:
        chunks.append(" ".join(current_words))

    num_chunks = len(chunks)

    if mock:
        chunk_summaries = [_mock_summary(chunk) for chunk in chunks]
    else:
        chunk_summaries = []
        for chunk in chunks:
            chunk_summary = _run_summarizer(
                chunk,
                max_length=max_length,
                min_length=min(30, max_length - 1),
            )
            chunk_summaries.append(chunk_summary)

    # Combine chunk summaries and do a final summarization pass
    combined = " ".join(chunk_summaries)

    if mock:
        final_summary = _mock_summary(combined)
    else:
        if len(combined) > chunk_size:
            # Recursive: summarize the combined summaries if still too long
            final_result = chunk_and_summarize(
                combined, chunk_size=chunk_size, max_length=max_length
            )
            final_summary = final_result["summary"]
        else:
            final_summary = _run_summarizer(
                combined,
                max_length=max_length,
                min_length=min(30, max_length - 1),
            )

    return {
        "summary": final_summary,
        "chunks": num_chunks,
        "original_length": original_length,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

SAMPLE_REAL_ESTATE_TEXT = """
Annonce immobilière : Appartement de prestige au cœur de Paris 8e arrondissement

Nous avons le plaisir de vous présenter ce magnifique appartement haussmannien de 120 mètres
carrés situé au troisième étage avec ascenseur, en plein cœur du 8e arrondissement de Paris,
à proximité immédiate des Champs-Élysées et de l'Arc de Triomphe.

Ce bien d'exception, entièrement rénové avec des matériaux haut de gamme, dispose de quatre
pièces lumineuses comprenant un vaste salon-salle à manger de 40 m² avec parquet en chêne
massif, une cuisine équipée ouverte sur le séjour, deux chambres dont une suite parentale avec
dressing et salle de bain privative, et une troisième chambre ou bureau, ainsi qu'une salle
de douche supplémentaire et de nombreux rangements intégrés.

Les plafonds haussmanniens de 3,20 mètres de hauteur, les moulures d'époque soigneusement
restaurées et les grandes fenêtres offrant une vue dégagée sur les toits parisiens confèrent à
cet appartement un charme et un cachet incomparables.

Le bien bénéficie d'une excellente exposition plein sud garantissant une luminosité naturelle
tout au long de la journée. La copropriété, en parfait état d'entretien, comprend une cour
intérieure arborée, un gardien à temps plein et un local vélos sécurisé.

Situé à moins de cinq minutes à pied des meilleures boutiques, restaurants étoilés et grands
hôtels internationaux, cet appartement constitue une opportunité d'investissement exceptionnelle,
que ce soit pour une résidence principale, secondaire ou un investissement locatif haut de gamme.
Prix de vente : 1 850 000 euros honoraires d'agence inclus. Charges de copropriété : 450 euros
par mois. Taxe foncière annuelle : 2 200 euros. Diagnostics énergétiques disponibles sur demande.
Visite possible sur rendez-vous uniquement. Dossier de financement exigé avant toute visite.
Contactez notre agence pour plus d'informations et pour planifier une visite privée de ce bien
d'exception au cœur de la capitale française.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="French real estate text summarization (BARThez)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip model loading; return a canned summary.",
    )
    parser.add_argument(
        "--pdf",
        metavar="PATH",
        help="Path to a PDF file to summarize instead of the sample text.",
    )
    args = parser.parse_args()

    print("=== French Real Estate Summarization Demo ===\n")

    if args.pdf:
        print(f"Summarizing PDF: {args.pdf}\n")
        result = summarize_pdf(args.pdf, mock=args.mock)
        print(f"Pages          : {result['pages']}")
        print(f"Original length: {result['original_length']} characters")
        print(f"Summary length : {result['summary_length']} characters")
        print(f"\nSummary:\n{result['summary']}\n")
    else:
        # ----------------------------------------------------------------
        # 1. Basic text summarization
        # ----------------------------------------------------------------
        print("-- summarize_text() --")
        result = summarize_text(SAMPLE_REAL_ESTATE_TEXT, mock=args.mock)
        print(f"Original length : {result['original_length']} characters")
        print(f"Summary length  : {result['summary_length']} characters")
        print(f"Compression ratio: {result['ratio']:.2%}")
        print(f"\nSummary:\n{result['summary']}\n")

        # ----------------------------------------------------------------
        # 2. Chunk-and-summarize (force chunking by using small chunk_size)
        # ----------------------------------------------------------------
        print("-- chunk_and_summarize() --")
        chunk_result = chunk_and_summarize(
            SAMPLE_REAL_ESTATE_TEXT,
            chunk_size=500,
            max_length=100,
            mock=args.mock,
        )
        print(f"Chunks used     : {chunk_result['chunks']}")
        print(f"Original length : {chunk_result['original_length']} characters")
        print(f"\nFinal summary:\n{chunk_result['summary']}\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
