"""
Text-Chunking für RAG-Pipeline
================================

Zerlegt Dokumente in überlappende Chunks für die Vektorisierung.
Optimiert für psychiatrische Fachtexte und Gerichtsakten.
"""

from app.config import get_settings


class TextChunker:
    """Teilt Text in überlappende Chunks mit konfigurierbarer Größe."""

    def __init__(self):
        settings = get_settings()
        self._chunk_size = settings.rag_chunk_size
        self._overlap = settings.rag_chunk_overlap
        # Trennzeichen in Prioritätsreihenfolge
        self._separators = [
            "\n\n\n",     # Abschnittswechsel
            "\n\n",       # Absatz
            "\n",         # Zeilenumbruch
            ". ",         # Satz (Deutsch/Englisch)
            "! ",
            "? ",
            "; ",
            ", ",
            " ",          # Wort
        ]

    def chunk(self, text: str) -> list[str]:
        """
        Zerlegt Text in überlappende Chunks.
        
        Strategie:
        1. Versuche an natürlichen Grenzen (Absätze, Sätze) zu trennen
        2. Überlappung für Kontexterhalt zwischen Chunks
        3. Kein Chunk kürzer als 50 Zeichen (Mindestlänge)
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Text ist kurz genug → ein einziger Chunk
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # End-Position bestimmen
            end = start + self._chunk_size

            if end >= len(text):
                # Letzter Chunk
                chunk = text[start:].strip()
                if chunk and len(chunk) >= 50:
                    chunks.append(chunk)
                break

            # Natürliche Trennstelle finden (rückwärts suchen)
            best_split = end
            for sep in self._separators:
                # Suche rückwärts ab end-Position
                split_pos = text.rfind(sep, start, end)
                if split_pos > start:
                    best_split = split_pos + len(sep)
                    break

            chunk = text[start:best_split].strip()
            if chunk and len(chunk) >= 50:
                chunks.append(chunk)

            # Nächster Chunk-Start mit Überlappung
            start = best_split - self._overlap
            if start < 0:
                start = 0

        return chunks
